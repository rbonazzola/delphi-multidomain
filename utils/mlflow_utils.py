import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import mlflow
import torch
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = Path(os.getenv("MLFLOW_TRACKING_URI", DELPHI_DIR / "mlruns"))


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MLFLOW_TRACKING_URI


def _get_experiment_id_from_runid(run_id: str) -> str:
    return mlflow.get_run(run_id).info.experiment_id


def load_run_params(run_id: str) -> dict:
    """Load MLflow run params; parses attention_scheme from string repr."""
    run = mlflow.get_run(run_id)
    params = dict(run.data.params)
    if "attention_scheme" in params:
        try:
            params["attention_scheme"] = ast.literal_eval(params["attention_scheme"])
        except (ValueError, SyntaxError):
            params["attention_scheme"] = [params["attention_scheme"]]
    return params


def parse_domains_param(domains_str: str, run_id: Optional[str] = None) -> dict:
    """Parse the domains MLflow param (handles embedded PosixPath reprs).

    MLflow truncates param values at 6000 chars; the `domains` param for
    ESM2-per-locus configs (9 domains) routinely exceeds that, leaving the
    string with an unterminated brace. When run_id is given and the direct
    parse fails, falls back to the untruncated domain_config_*.yaml artifact
    logged alongside the run (see train.py's "config" artifact_path).
    """
    from delphi.model import DomainConfig
    s_clean = re.sub(r"PosixPath\(([^)]+)\)", r"\1", domains_str)
    try:
        domains_dict = ast.literal_eval(s_clean)
    except (ValueError, SyntaxError):
        if run_id is None:
            raise
        domains_dict = _load_domains_from_config_artifact(run_id)
    return {k: DomainConfig(**v) for k, v in domains_dict.items()}


def _load_domains_from_config_artifact(run_id: str) -> dict:
    """Load the full (untruncated) domains dict from the run's logged config artifact.

    Finetune runs are created by clone_run_to_new_experiment, which copies the
    source run's artifacts; some early clones did not end up with their own
    "config" artifact, so fall back to finetune_source_run_id when this run
    has none of its own.
    """
    try:
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config")
    except OSError:
        source_run_id = mlflow.get_run(run_id).data.params.get("finetune_source_run_id")
        if not source_run_id:
            raise FileNotFoundError(f"No 'config' artifact for run {run_id} and no finetune_source_run_id to fall back to")
        local_dir = mlflow.artifacts.download_artifacts(run_id=source_run_id, artifact_path="config")
    yaml_files = sorted(Path(local_dir).glob("domain_config_*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(
            f"No domain_config_*.yaml artifact found for run {run_id} "
            "(needed because the 'domains' param is truncated)"
        )
    # unsafe_load: the artifact is logged by our own train.py (trusted), and
    # dataclasses.asdict() leaves `path` as a PosixPath, which safe_load can't construct.
    return yaml.unsafe_load(yaml_files[-1].read_text())


@dataclass
class RunSetup:
    """Decoded HLA encoding/dropout/attention setup for one training run."""
    encoding: str                  # "1field" | "2field" | "esm2_raw" | "esm2_pca" | "none" | "other"
    hla_domains: list              # domain name(s) carrying the HLA signal; [] if encoding == "none"
    dropout_mode: Optional[str]    # DomainConfig.dropout_mode on the HLA domain, or None
    dropout_rate: float            # 0.0 if no dropout (or no HLA domain)
    attention: Optional[str]       # "bidirectional" | "causal" | None if encoding == "none"
    n_embd: Optional[int] = None
    n_head: Optional[int] = None
    n_layer: Optional[int] = None

    def label(self) -> str:
        """Composite string like 'esm2_raw__nodrop__bidir', matching the run_name
        tagging convention used in train_scripts/*.tsv (e.g. hla_esm2raw_nodrop_bidir)."""
        if self.encoding in ("none", "other"):
            return self.encoding
        drop = "nodrop" if self.dropout_rate == 0 else f"drop{int(round(self.dropout_rate * 100))}"
        attn = "bidir" if self.attention == "bidirectional" else "causal"
        return f"{self.encoding}__{drop}__{attn}"

    def __str__(self) -> str:
        return self.label()


def _extract_domain_dict(domains_str: str, domain_name: str) -> dict:
    """Extract and parse a single domain's dict from the raw `domains` param string by
    brace-matching, instead of ast.literal_eval-ing the whole string.

    MLflow truncates param values at 6000 chars — the full `domains` param routinely
    exceeds that for the 9-domain ESM2-per-locus configs, so a whole-string parse
    (parse_domains_param) raises a SyntaxError on those runs. Brace-matching just the
    domain we actually need (one HLA domain) works as long as that domain's own dict
    isn't itself past the truncation point, which holds for the current configs.
    """
    m = re.search(r"'" + re.escape(domain_name) + r"':\s*\{", domains_str)
    if not m:
        raise KeyError(f"Domain {domain_name!r} not found in domains param (string may be truncated)")
    start = m.end() - 1  # index of the opening '{'
    depth = 0
    end = None
    for i in range(start, len(domains_str)):
        if domains_str[i] == "{":
            depth += 1
        elif domains_str[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"Domain {domain_name!r} dict is truncated in domains param")
    snippet = re.sub(r"PosixPath\(([^)]+)\)", r"\1", domains_str[start:end])
    return ast.literal_eval(snippet)


def _detect_hla_encoding(domains_str: str) -> tuple[str, list, dict]:
    """Returns (encoding_label, hla_domain_names, representative_domain_dict), reading
    only domain key names + one targeted domain dict from the raw `domains` param
    string (see _extract_domain_dict for why not the full parse)."""
    keys = re.findall(r"'(\w+)':\s*\{", domains_str)

    if "hla_alleles" in keys:
        d = _extract_domain_dict(domains_str, "hla_alleles")
        encoding = "1field" if d.get("token_value_column") == "allele_1field" else "2field"
        return encoding, ["hla_alleles"], d

    hla_loci = [k for k in keys if k.startswith("hla_") and k != "hla_alleles"]
    if hla_loci:
        d = _extract_domain_dict(domains_str, hla_loci[0])
        encoding = "esm2_pca" if "pca" in str(d.get("pretrained_path") or "") else "esm2_raw"
        return encoding, hla_loci, d

    return "none", [], {}


def _hla_attention(attention_scheme, hla_domains: list) -> Optional[str]:
    """Scans the (parsed) attention_scheme for the rule covering the HLA group/domains.
    Falls back to "causal" if the HLA domains aren't named explicitly (covered only by
    a catch-all "all:..." rule, which is causal in every scheme used so far)."""
    if not hla_domains:
        return None
    schemes = attention_scheme if isinstance(attention_scheme, list) else [attention_scheme]
    for layer_scheme in schemes:
        for rule in re.split(r",(?![^\[]*\])", layer_scheme):
            domain_part, _, rule_part = rule.partition(":")
            domain_part, rule_part = domain_part.strip(), rule_part.strip()
            names = [n.strip() for n in domain_part.strip("[]").split(",")]
            if "hla_alleles" in names or any(d in names for d in hla_domains):
                return "bidirectional" if rule_part.startswith("bidirectional") else "causal"
    return "causal"


def get_run_setup(params: dict) -> RunSetup:
    """Decode a run's MLflow params (as returned by load_run_params) into a RunSetup.

    Reads `params["domains"]` (the resolved, post-override DomainConfig dict — the
    ground truth for what actually ran) rather than the domain_config_yaml path or
    run_name, so it works regardless of --dcfg overrides or naming conventions.
    """
    domains_str = params["domains"]

    attention_scheme = params.get("attention_scheme")
    if isinstance(attention_scheme, str):
        try:
            attention_scheme = ast.literal_eval(attention_scheme)
        except (ValueError, SyntaxError):
            attention_scheme = [attention_scheme]

    encoding, hla_domains, hla_cfg = _detect_hla_encoding(domains_str)
    dropout_mode = hla_cfg.get("dropout_mode")
    dropout_rate = hla_cfg.get("dropout_rate") or 0.0
    attention = _hla_attention(attention_scheme, hla_domains)

    def _int_or_none(v):
        return int(v) if v is not None else None

    return RunSetup(
        encoding=encoding,
        hla_domains=hla_domains,
        dropout_mode=dropout_mode,
        dropout_rate=dropout_rate,
        attention=attention,
        n_embd=_int_or_none(params.get("n_embd")),
        n_head=_int_or_none(params.get("n_head")),
        n_layer=_int_or_none(params.get("n_layer")),
    )


def get_run_setup_for_run(run_id: str) -> RunSetup:
    """Convenience wrapper: loads params for run_id, then decodes via get_run_setup."""
    return get_run_setup(load_run_params(run_id))


def get_checkpoint_path(run_id: str) -> Path:
    """
    Return best_model.pt for a run, falling back to the highest-epoch checkpoint.
    Prefers best_model.pt (lowest validation loss) over the latest epoch.
    """
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    ckpt_dir = Path(unquote(urlparse(unquote(artifact_uri)).path)) / "checkpoints"

    best = ckpt_dir / "best_model.pt"
    if best.exists():
        return best

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    epoch_re = re.compile(r"epoch(\d+)", re.IGNORECASE)
    best_ckpt, best_epoch = None, -1
    for ck in ckpts:
        m = epoch_re.search(ck.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_ckpt = epoch, ck

    return best_ckpt or ckpts[-1]


def _get_last_epoch_checkpoint(run_dir: str) -> tuple[Path, int]:
    """Return (path, epoch) of the highest-epoch checkpoint under run_dir/artifacts/checkpoints."""
    ckpt_dir = Path(run_dir) / "artifacts" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found.")

    epoch_re = re.compile(r"epoch(\d+)", re.IGNORECASE)
    best, best_epoch = None, -1
    for ck in ckpts:
        m = epoch_re.search(ck.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best = epoch, ck

    if best is None:
        raise RuntimeError("No checkpoint contained an epoch number.")

    return best, best_epoch



def load_checkpoint(run_id: str) -> tuple[dict, Path]:
    """Load best_model.pt, falling back to the highest-epoch checkpoint."""
    ckpt_path = get_checkpoint_path(run_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path
