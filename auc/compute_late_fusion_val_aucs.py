"""
Compute val-split (default) or test-split AUCs for an existing DelphiLateFusion run, from its
saved head checkpoint, without retraining. Mirrors train_late_fusion.py's --skip_training +
--compute_aucs path, and logs aucs_val.csv (val) or aucs.csv (test) into the same MLflow run.

Usage:
    python auc/compute_late_fusion_val_aucs.py --runid <late_fusion_run_id> [--split test]
"""
import argparse
import copy
import sys
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

DELPHI_DIR = Path(__file__).resolve().parent.parent
if DELPHI_DIR not in map(Path, sys.path):
    sys.path.insert(0, str(DELPHI_DIR))

from data.late_fusion_dataset import LateFusionDataset, LateFusionCollateFn
from delphi.late_fusion import DelphiLateFusion, build_hla_head
from utils.mlflow_utils import setup_mlflow, get_checkpoint_path
from utils.run_loader import reconstruct_from_run
from utils.trainer import MLFlowLogger
from train_late_fusion import build_encoder, build_hla_features
from auc.aucs import DEFAULT_WASHOUT_DAYS
from auc.late_fusion_aucs import evaluate_aucs_late_fusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runid", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--eval_block_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--washout_days", type=float, default=DEFAULT_WASHOUT_DAYS,
                    help="Score each case at the last token at least this many days before the "
                         "diagnosis. Default: %(default)g. 0: the token right before it. "
                         "See auc/aucs.py:evaluate_aucs")
    args = ap.parse_args()

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(args.runid)
    params = run.data.params

    trunk_run_id = params["trunk_run_id"]
    hla_encoding = params["hla_encoding"]
    head_hidden_dims_str = params.get("head_hidden_dims")
    hidden_dims = None if head_hidden_dims_str in (None, "linear") else [int(x) for x in head_hidden_dims_str.split(",")]

    print(f"Loading trunk from run {trunk_run_id} ({args.split} split only)...")
    trunk, loaders, _ = reconstruct_from_run(trunk_run_id, split=[args.split], batch_size=args.eval_batch_size,
                                              num_workers=args.num_workers, device=DEVICE)
    trunk.eval()
    print(f"Trunk loaded: {sum(p.numel() for p in trunk.parameters())} params")

    print(f"Building HLA features (encoding={hla_encoding})...")
    encoder = build_encoder(hla_encoding)
    val_loader = loaders[args.split]
    val_dataset = val_loader.dataset
    subject_ids = val_dataset._subject_ids.numpy()
    hla_features = build_hla_features(subject_ids, encoder)
    lf_dataset = LateFusionDataset(val_dataset, hla_features)
    lf_collate = LateFusionCollateFn(val_loader.collate_fn)

    predicted_vocab_sizes = {dname: trunk.embed._domain_vocab_sizes[dname] for dname in trunk.predicted_domains}
    rank = params.get("head_rank")
    head = build_hla_head(encoder, predicted_vocab_sizes, hidden_dims=hidden_dims,
                          rank=None if rank in (None, "full") else int(rank),
                          factorized=params.get("head_factorized"), trunk=trunk)

    ckpt_path = get_checkpoint_path(args.runid)
    print(f"Loading head checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    print(f"Head loaded (epoch={ckpt.get('metadata', {}).get('epoch')})")

    model = DelphiLateFusion(trunk=trunk, head=head).to(DEVICE)
    model.eval()

    auc_collate = copy.copy(lf_collate)
    auc_collate.delphi_collate.block_size = args.eval_block_size
    auc_loader = DataLoader(lf_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=auc_collate)

    output_file = "aucs_val.csv" if args.split == "val" else "aucs.csv"
    logger = MLFlowLogger(experiment_name=None, run_name=None, autostart=False)
    logger.start(resume_run_id=args.runid)
    try:
        auc_df = evaluate_aucs_late_fusion(
            model=model, test_loader=auc_loader, focus_token_ids=None,
            block_size=args.eval_block_size, n_jobs=8, logger=logger, output_file=output_file,
            washout_days=args.washout_days,
        )
        print(f"AUC summary:\n{auc_df.groupby('domain')['auc_delong'].describe().to_string()}")
    finally:
        logger.end()

    print(f"Done. Run {args.runid} now has aucs/{output_file} logged.")


if __name__ == "__main__":
    main()
