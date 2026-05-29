"""
Standalone AUC evaluation script.

Loads a trained model from an MLflow run_id (using best_model.pt),
reconstructs the test dataloader, and computes AUCs.

Usage:
    python compute_aucs.py --runid <mlflow_run_id> [--block_size 128] [--batch_size 512] [--n_jobs 8]
"""

import argparse
import logging
import os

import mlflow
import torch

from auc.aucs import evaluate_aucs
from utils.mlflow_utils import setup_mlflow
from utils.run_loader import reconstruct_from_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

setup_mlflow()

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Compute AUCs for a trained Delphi model")
    parser.add_argument("--runid", required=True, help="MLflow run ID")
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Fixed block size for AUC evaluation (must be an integer; 'auto' is not supported)",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel jobs for AUC computation")
    parser.add_argument("--output_file", type=str, default="aucs.csv")
    args = parser.parse_args()

    model, loaders, run_params = reconstruct_from_run(
        args.runid,
        split="test",
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    auc_df = evaluate_aucs(
        model,
        test_loader,
        block_size=model.block_size,
        run_id=args.runid,
        n_jobs=args.n_jobs,
        logger=logger,
        output_file=args.output_file,
    )

    logger.end()
    logging.info(f"Done. {len(auc_df)} AUC rows computed.")


if __name__ == "__main__":
    main()
