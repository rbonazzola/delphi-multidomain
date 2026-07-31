"""
Standalone AUC evaluation script.

Loads a trained model from an MLflow run_id (using best_model.pt),
reconstructs the test dataloader, and computes AUCs.

By default, evaluates on the run's own stored test split (the model's
training cohort). Two optional overrides let you evaluate on a different
cohort instead:

  --tokens_dir  Load domain tokens from this folder instead of wherever the
                run's own config points -- each domain's data must live under
                <tokens_dir>/<domain_name>/ (tokens.csv + tokenizer.yaml),
                sharing the same vocabulary the model was trained with (e.g.
                a synthetic cohort, or another held-out dataset).
  --subjects    Restrict to these subject_ids instead of the run's own
                stored test_ids -- a file with one id per line (see
                utils.utils.read_ids).

Usage:
    python compute_aucs.py --runid <mlflow_run_id> [--block_size 128] [--batch_size 512] [--n_jobs 8]

    # Evaluate on a different cohort's tokens, restricted to a subject list
    python compute_aucs.py --runid <mlflow_run_id> \\
        --tokens_dir transforms/tokens_synthetic \\
        --subjects data/transforms/subject_lists/some_ids.csv \\
        --output_file aucs_custom.csv
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

if (DELPHI_DIR := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from auc.aucs import evaluate_aucs
from utils.mlflow_utils import setup_mlflow, get_checkpoint_path, load_run_params, parse_domains_param
from utils.run_loader import reconstruct_from_run, AUTO_BLOCK_SIZE

setup_mlflow()

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute AUCs for a trained Delphi model")
    parser.add_argument("--runid", required=True, help="MLflow run ID")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Fixed block size for AUC evaluation (must be an integer; 'auto' is not supported)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel jobs for AUC computation")
    parser.add_argument("--output_file", type=str, default="aucs.csv")
    parser.add_argument("--tokens_dir", type=str, default=None,
                        help="Load domain tokens from this folder instead of the run's own "
                             "config (each domain under <tokens_dir>/<domain_name>/)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Path to a subject ids file, to evaluate on instead of the "
                             "run's own stored test split")
    args = parser.parse_args()

    model, loaders, run_params = reconstruct_from_run(
        args.runid,
        split="test",
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokens_path=args.tokens_dir,
        subjects=args.subjects,
    )
    test_loader = loaders["test"]

    from utils.trainer import MLFlowLogger

    run = mlflow.get_run(args.runid)
    experiment_name = mlflow.get_experiment(run.info.experiment_id).name

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=None,
        autostart=False,
    )
    logger.start(resume_run_id=args.runid)
    try:
        auc_df = evaluate_aucs(
            model,
            test_loader,
            block_size=model.block_size,
            run_id=args.runid,
            n_jobs=args.n_jobs,
            logger=logger,
            output_file=args.output_file,
        )
        logging.info(f"Done. {len(auc_df)} AUC rows computed.")
    finally:
        logger.end()


if __name__ == "__main__":
    main()
