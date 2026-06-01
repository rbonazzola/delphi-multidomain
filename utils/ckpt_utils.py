import re
from typing import TypedDict


class InferredConfig(TypedDict):
    n_layer: int
    n_embd: int
    domains: list[str]
    use_age_embedding: bool
    use_final_layernorm: bool
    vocab_sizes: dict[str, int]


def infer_delphi_config_from_state_dict(sd: dict) -> InferredConfig:
    layer_indices = [int(m.group(1)) for key in sd if (m := re.match(r"transformer\.h\.(\d+)\.", key))]
    if not layer_indices:
        raise ValueError("No transformer layer keys found in state dict")

    n_embd: int | None = next((val.shape[1] for key, val in sd.items() if "attn.c_attn.weight" in key), None)
    if n_embd is None:
        raise ValueError("Could not infer n_embd: no 'attn.c_attn.weight' key found")

    domains: list[str] = [
        m.group(1) for key in sd if (m := re.match(r"transformer\.embed\.domain_embed\.(\w+)\.projector\.weight", key))
    ]

    vocab_sizes: dict[str, int] = {}
    for d in domains:
        key = f"embedding_to_logits.embedding_layer_dict.domain_embed.{d}.projector.weight"
        if key in sd:
            vocab_sizes[d] = sd[key].shape[0]

    return InferredConfig(
        n_layer=max(layer_indices) + 1,
        n_embd=n_embd,
        domains=domains,
        use_age_embedding=any("age_embedding" in k for k in sd),
        use_final_layernorm="transformer.ln_f.weight" in sd,
        vocab_sizes=vocab_sizes,
    )


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
    old_keys = [k for k in weights if k.startswith("embed.domain_embed.") and k.endswith(".projector.weight")]
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
        global_weight[offset : offset + vocab_size] = domain_weight

    new_weights = {k: v for k, v in weights.items() if not k.startswith("embed.domain_embed.")}
    new_weights["embed.global_embed.weight"] = global_weight
    return new_weights


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove torch.compile '_orig_mod.' prefix from state dict keys."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict
