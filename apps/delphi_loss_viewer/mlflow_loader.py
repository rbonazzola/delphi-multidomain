from pathlib import Path

import mlflow
import pandas as pd


def load_runs(exp_ids):
    """
    Load MLflow runs for given experiment IDs and normalize artifact_uri.
    """
    runs_df = mlflow.search_runs(experiment_ids=exp_ids, output_format="pandas")
    assert isinstance(runs_df, pd.DataFrame)
    runs_df = runs_df.query("experiment_id != '0'").copy()

    runs_df["artifact_uri"] = runs_df["artifact_uri"].str.replace(r".*mlruns", mlflow.get_tracking_uri(), regex=True)
    return runs_df


def filter_runs_with_loss_files(runs_df, min_files=10):
    """
    DEBUG VERSION — prints what it finds.
    """

    from pathlib import Path

    counts = []

    for _, row in runs_df.iterrows():
        data_dir = Path(row["artifact_uri"]) / "val_loss_per_disease"
        found = list(data_dir.glob("losses_epoch*_*.csv"))
        counts.append((row["run_id"], len(found), str(data_dir)))

    # print("\n=== DEBUG: LOSS FILE COUNTS ===")
    # for run_id, n, path in counts:
    # print(f"Run {run_id}: {n} files in {path}")
    # print("================================\n")

    # original logic
    runs_df["n_loss_files"] = [n for _, n, _ in counts]
    return runs_df[runs_df["n_loss_files"] >= min_files].copy()


def validate_loss_files(runs_df):
    """
    Try reading the first loss file of each run.
    Keep only those with valid readable CSVs.
    """
    valid_ids = []

    for _, row in runs_df.iterrows():
        data_dir = Path(row["artifact_uri"]) / "val_loss_per_disease"
        files = list(data_dir.glob("losses_epoch*_*.csv"))
        if not files:
            continue

        try:
            sample = pd.read_csv(files[0], nrows=3)
            if not sample.empty:
                valid_ids.append(row["run_id"])
        except Exception:
            continue

    return runs_df[runs_df["run_id"].isin(valid_ids)]
