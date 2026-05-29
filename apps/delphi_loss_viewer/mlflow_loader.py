import re
from pathlib import Path

import mlflow
import pandas as pd


def _tracking_root() -> Path:
    """Return the mlruns directory as a plain filesystem Path (no file:// prefix)."""
    uri = mlflow.get_tracking_uri()
    return Path(re.sub(r"^file://", "", uri))


def load_runs(exp_ids):
    """Load MLflow runs for given experiment IDs and normalize artifact_uri."""
    runs_df = mlflow.search_runs(experiment_ids=exp_ids)
    assert isinstance(runs_df, pd.DataFrame)
    runs_df = runs_df.query("experiment_id != '0'").copy()

    root = str(_tracking_root())
    runs_df["artifact_uri"] = runs_df["artifact_uri"].str.replace(r".*mlruns", root, regex=True)
    return runs_df


def filter_runs_with_loss_files(runs_df, min_files=10):
    counts = []
    for _, row in runs_df.iterrows():
        data_dir = Path(row["artifact_uri"]) / "val_loss_per_disease"
        found = list(data_dir.glob("losses_epoch*_*.csv"))
        counts.append(len(found))
    runs_df["n_loss_files"] = counts
    return runs_df[runs_df["n_loss_files"] >= min_files].copy()


def validate_loss_files(runs_df):
    """Keep only runs with at least one readable loss CSV."""
    valid_ids = []
    for _, row in runs_df.iterrows():
        data_dir = Path(row["artifact_uri"]) / "val_loss_per_disease"
        files = list(data_dir.glob("losses_epoch*_*.csv"))
        if not files:
            continue
        try:
            if not pd.read_csv(files[0], nrows=3).empty:
                valid_ids.append(row["run_id"])
        except Exception:
            continue
    return runs_df[runs_df["run_id"].isin(valid_ids)]
