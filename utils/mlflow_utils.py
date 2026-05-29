import ast
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import torch

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


def parse_domains_param(domains_str: str) -> dict:
    """Parse the domains MLflow param (handles embedded PosixPath reprs)."""
    from delphi.model import DomainConfig

    s_clean = re.sub(r"PosixPath\(([^)]+)\)", r"\1", domains_str)
    domains_dict = ast.literal_eval(s_clean)
    return {k: DomainConfig(**v) for k, v in domains_dict.items()}


def get_checkpoint_path(run_id: str) -> Path:
    """
    Return best_model.pt for a run, falling back to the highest-epoch checkpoint.
    Prefers best_model.pt (lowest validation loss) over the latest epoch.
    """
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    assert artifact_uri is not None, "MLflow returned a None artifact_uri"
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


def _get_last_epoch_checkpoint(run_dir: Path | str) -> tuple[Path, int]:
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
