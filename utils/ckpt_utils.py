import logging
import re
from dataclasses import asdict, fields as dc_fields
from pathlib import Path

from easydict import EasyDict


def infer_delphi_config_from_state_dict(sd: dict) -> EasyDict:
    cfg = EasyDict()

    layer_indices = []
    for key in sd:
        m = re.match(r"transformer\.h\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    cfg.n_layer = max(layer_indices) + 1

    for key, val in sd.items():
        if "attn.c_attn.weight" in key:
            cfg.n_embd = val.shape[1]
            break

    for key, val in sd.items():
        if "mlp.c_fc.weight" in key:
            cfg.mlp_hidden_dim = val.shape[0]
            break

    cfg.domains = []
    for key in sd:
        m = re.match(r"transformer\.embed\.domain_embed\.(\w+)\.projector\.weight", key)
        if m:
            cfg.domains.append(m.group(1))

    cfg.use_age_embedding = any("age_embedding" in k for k in sd)
    cfg.use_final_layernorm = "transformer.ln_f.weight" in sd

    cfg.vocab_sizes = EasyDict()
    for d in cfg.domains:
        key = f"embedding_to_logits.embedding_layer_dict.domain_embed.{d}.projector.weight"
        if key in sd:
            cfg.vocab_sizes[d] = sd[key].shape[0]

    return cfg


def migrate_legacy_state_dict(weights: dict) -> dict:
    """
    Normalize old embedding key naming to the current layout.

    - transformer.embed.*   → embed.*
    - embedding_to_logits.* → embed.*
    """
    new_weights = {}
    for k, v in weights.items():
        new_key = k
        if k.startswith("transformer.embed."):
            new_key = k.replace("transformer.embed.", "embed.")
        elif k.startswith("embedding_to_logits."):
            new_key = k.replace("embedding_to_logits.embedding_layer_dict.", "embed.")
            new_key = new_key.replace("embedding_to_logits.", "embed.")
        if new_key not in new_weights:
            new_weights[new_key] = v
    return new_weights


def migrate_domain_embed_to_global_embed(weights: dict, model) -> dict:
    """
    Migrate per-domain embedding weights to the unified global_embed table.

    Old: embed.domain_embed.{domain}.projector.weight
    New: embed.global_embed.weight
    """
    old_keys = [k for k in weights
                if k.startswith("embed.domain_embed.") and k.endswith(".projector.weight")]
    if not old_keys:
        return weights

    embed = model.embed
    global_weight = embed.global_embed.weight.data.clone()

    for key in old_keys:
        domain = key.split(".")[2]
        if domain not in embed.domain_to_int:
            continue
        d_int = embed.domain_to_int[domain]
        offset = embed.domain_offsets[d_int]
        domain_weight = weights[key]
        vocab_size = domain_weight.shape[0]
        global_weight[offset: offset + vocab_size] = domain_weight

    new_weights = {k: v for k, v in weights.items()
                   if not k.startswith("embed.domain_embed.")}
    new_weights["embed.global_embed.weight"] = global_weight
    return new_weights


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove torch.compile '_orig_mod.' prefix from state dict keys."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ═══════════════════════════════════════════════════════════════════════════════
#  Model spec: everything needed to rebuild the model, stored in the checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SPEC_VERSION = 1
_REPO_DIR = Path(__file__).resolve().parent.parent
_DOMAIN_PATH_FIELDS = ("path", "pretrained_path")


def _to_repo_relative(p) -> str:
    """Store paths under the repo as repo-relative, so a checkpoint loads from any clone."""
    p = Path(p)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(_REPO_DIR))
        except ValueError:
            return str(p)
    return str(p)


def _from_repo_relative(p: str, repo_dir: Path) -> str:
    return str(p) if Path(p).is_absolute() else str(repo_dir / p)


def model_spec(model) -> dict | None:
    """
    Serialize the full, resolved construction config of a Delphi / DelphiMultiStream
    model into plain Python types (no dataclasses, no Path objects).

    Stored in every checkpoint under "model_spec" so that a checkpoint is
    self-contained: model_from_spec(ckpt["model_spec"]) rebuilds the exact
    architecture without consulting MLflow params (which MLflow truncates at
    6000 chars) or the domain config YAMLs. Returns None for model classes it
    doesn't know (e.g. a late-fusion head), so callers can store it unconditionally.
    """
    from delphi.model import Delphi
    from delphi.multi_stream_model import DelphiMultiStream

    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    if isinstance(model, DelphiMultiStream):
        spec = {
            "model_class": "DelphiMultiStream",
            "config": asdict(model._config),
            "scheme": asdict(model._scheme),
            "attention_scheme": model.attention_scheme,
        }
    elif isinstance(model, Delphi):
        spec = {"model_class": "Delphi", "config": asdict(model.config)}
    else:
        return None

    for dcfg in spec["config"]["domains"].values():
        for f in _DOMAIN_PATH_FIELDS:
            if dcfg.get(f) is not None:
                dcfg[f] = _to_repo_relative(dcfg[f])
    return {"version": MODEL_SPEC_VERSION, **spec}


def _domain_configs_from_spec(domains: dict, repo_dir: Path) -> dict:
    from delphi.model import DomainConfig

    known = {f.name for f in dc_fields(DomainConfig)}
    out = {}
    for name, d in domains.items():
        unknown = set(d) - known
        if unknown:
            logging.warning("Domain %r: ignoring fields no longer in DomainConfig: %s", name, sorted(unknown))
        d = {k: v for k, v in d.items() if k in known}
        for f in _DOMAIN_PATH_FIELDS:
            if d.get(f) is not None:
                d[f] = _from_repo_relative(d[f], repo_dir)
        out[name] = DomainConfig(**d)
    return out


def model_from_spec(spec: dict, repo_dir: Path | None = None, **config_overrides):
    """
    Rebuild an (untrained) model from a model_spec() dict.

    config_overrides replace fields of the top-level config (e.g. block_size=128).
    Repo-relative domain paths are resolved against repo_dir (default: this repo).
    """
    from delphi.model import Delphi, DelphiConfig
    from delphi.multi_stream_model import (
        DelphiMultiStream, DelphiMultiStreamConfig, EncoderSpec, MultiStreamScheme,
    )

    if spec.get("version") != MODEL_SPEC_VERSION:
        raise ValueError(f"Unsupported model_spec version {spec.get('version')!r} (expected {MODEL_SPEC_VERSION})")

    repo_dir = Path(repo_dir) if repo_dir is not None else _REPO_DIR
    cfg = dict(spec["config"]) | config_overrides
    cfg["domains"] = _domain_configs_from_spec(cfg["domains"], repo_dir)

    if spec["model_class"] == "Delphi":
        return Delphi(DelphiConfig(**cfg))
    if spec["model_class"] == "DelphiMultiStream":
        s = dict(spec["scheme"])
        scheme = MultiStreamScheme(encoders=[EncoderSpec(**e) for e in s.pop("encoders")], **s)
        return DelphiMultiStream(DelphiMultiStreamConfig(**cfg), scheme, spec["attention_scheme"])
    raise ValueError(f"Unknown model_class {spec['model_class']!r}")


def normalize_state_dict(state_dict: dict, model) -> dict:
    """Map any older key layout (torch.compile prefix, embedding_to_logits.*, per-domain
    embedding tables, attn.bias mask buffers) onto the current one, so it loads with strict=True."""
    state_dict = migrate_legacy_state_dict(strip_compiled_prefix(state_dict))
    state_dict = migrate_domain_embed_to_global_embed(state_dict, model)
    # Old nanoGPT-style blocks registered the causal mask as an `attn.bias` buffer:
    # a constant, not a learned weight, and no longer part of the model.
    expected = model.state_dict()
    return {k: v for k, v in state_dict.items() if k in expected or not k.endswith(".attn.bias")}


def load_model_from_checkpoint(ckpt: dict, device=None, repo_dir: Path | None = None, **config_overrides):
    """Rebuild the model from ckpt["model_spec"] and load ckpt["state_dict"] into it (strict)."""
    if ckpt.get("model_spec") is None:
        raise KeyError("Checkpoint has no 'model_spec' (saved before model specs were stored in checkpoints)")
    model = model_from_spec(ckpt["model_spec"], repo_dir=repo_dir, **config_overrides)
    model.load_state_dict(normalize_state_dict(ckpt["state_dict"], model), strict=True)
    if device is not None:
        model = model.to(device)
    return model
