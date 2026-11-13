#!/usr/bin/env python3
"""
Train DelphiLateFusion: frozen Delphi trunk + HLA late-fusion head.

Loads a pretrained no-HLA Delphi run (specified by --trunk_run_id), builds
per-subject HLA feature vectors (multi-hot 1-field/2-field or mean-pooled ESM2
embeddings), and trains a linear or MLP head to add a static per-subject HLA
contribution to the predicted domain logits.

Only the head parameters are optimised; the trunk is kept frozen.

Usage
-----
python train_late_fusion.py \\
    --trunk_run_id <mlflow-run-id> \\
    --hla_encoding 2field \\
    --experiment_name late-fusion-hla \\
    --run_name fold1_2field_linear
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

DELPHI_DIR = Path(__file__).resolve().parent
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.late_fusion_dataset import LateFusionBatch, LateFusionDataset, LateFusionCollateFn
from delphi.late_fusion import (
    DelphiLateFusion,
    HLAHead,
    MultiHotHLAEncoder,
    PretrainedHLAEncoder,
    LocusESM2HLAEncoder,
    build_hla_head,
    HLAResidualStream,
    DelphiLateFusionResidual,
)
from delphi.model import mask_seen_token_logits
from torch.utils.data import DataLoader
from utils import setup_mlflow
from utils.run_loader import reconstruct_from_run
from utils.trainer import EarlyStopping, MLFlowLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_TOKENS_ROOT = DELPHI_DIR / "data" / "transforms" / "tokens"
_HLA_METADATA = _TOKENS_ROOT / "hla_alleles" / "token_metadata.tsv"
_HLA_TOKENS_CSV = _TOKENS_ROOT / "hla_alleles" / "tokens.csv"
_ESM2_DIR = DELPHI_DIR / "embeddings" / "hla" / "esm2_t48_15B_UR50D"


# ── Encoder factory ───────────────────────────────────────────────────────────

def build_encoder(hla_encoding: str):
    if hla_encoding == "1field":
        return MultiHotHLAEncoder(_HLA_METADATA, mode="1field")
    if hla_encoding == "2field":
        return MultiHotHLAEncoder(_HLA_METADATA, mode="2field")
    if hla_encoding == "esm2":
        return PretrainedHLAEncoder(_ESM2_DIR, _HLA_METADATA)
    if hla_encoding == "esm2_locus":
        return LocusESM2HLAEncoder(_ESM2_DIR, _HLA_METADATA)
    raise ValueError(f"Unknown --hla_encoding: {hla_encoding!r}. Choose 1field / 2field / esm2.")


def build_hla_features(subject_ids, encoder):
    """Return [N, d_hla] float32 tensor for the given subject IDs."""
    import pandas as pd

    df = pd.read_csv(_HLA_TOKENS_CSV)
    subj_to_tokens = df.groupby("subject_id")["token_id"].apply(list).to_dict()
    token_ids_per_subject = [subj_to_tokens.get(int(sid), []) for sid in subject_ids]
    return encoder.encode(token_ids_per_subject)


def hla_donor_map(seed: int = 0) -> dict:
    """Fixed random permutation of the subjects in the HLA tokens file: subject -> donor subject.
    Same mapping in training and evaluation (control --shuffle_hla)."""
    import numpy as np
    import pandas as pd

    subs = np.sort(pd.read_csv(_HLA_TOKENS_CSV, usecols=["subject_id"]).subject_id.unique())
    donors = np.random.default_rng(seed).permutation(subs)
    return dict(zip(subs.tolist(), donors.tolist()))


def build_hla_features_shuffled(subject_ids, encoder, seed: int = 0):
    """HLA features of each subject's donor under hla_donor_map(seed): keeps the HLA structure
    (alleles per locus, allele frequencies), breaks the genotype-subject link."""
    m = hla_donor_map(seed)
    return build_hla_features([m.get(int(s), int(s)) for s in subject_ids], encoder)


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(model: DelphiLateFusion, batch: LateFusionBatch) -> torch.Tensor:
    """Forward pass → CE loss, replicating Trainer.shared_step logic (no time-to-event)."""
    logits_dict, _ = model(batch)
    logits_dict = mask_seen_token_logits(model, batch.delphi_batch, logits_dict)

    target_global_ids = batch.global_token_ids[:, 1:]    # [B, T-1]
    target_domain_ids = batch.domain_ids[:, 1:]          # [B, T-1]

    logits_cat = torch.cat(
        [logits_dict[d] for d in model.predicted_domains], dim=-1
    )[:, :-1, :]                                          # [B, T-1, V_total]

    predicted_ints = torch.tensor(
        [model.domain_to_int[d] for d in model.predicted_domains],
        device=batch.domain_ids.device,
    )
    predict_mask = torch.isin(target_domain_ids, predicted_ints)   # [B, T-1]
    if batch.eval_mask is not None:
        predict_mask = predict_mask & ~batch.eval_mask[:, 1:]

    f_logits = logits_cat[predict_mask]          # [N_pred, V_total]
    f_global_ids = target_global_ids[predict_mask]
    f_domain_ids = target_domain_ids[predict_mask]

    # Remap global IDs → indices within logits_cat
    f_targets = f_global_ids.clone()
    cum = 0
    for dname in model.predicted_domains:
        d_int = model.domain_to_int[dname]
        mask = f_domain_ids == d_int
        f_targets[mask] = f_global_ids[mask] - model.domain_offsets[d_int] + cum
        cum += logits_dict[dname].shape[-1]

    return model.cross_entropy_loss(f_logits, f_targets)


# ── Epoch runners ─────────────────────────────────────────────────────────────

def run_epoch(
    model: DelphiLateFusion,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> float:
    """Run one full epoch. Returns mean CE loss over all batches."""
    training = optimizer is not None
    model.head.train(training)

    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            loss = compute_loss(model, batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    setup_mlflow()

    # 1. Load frozen trunk + subject splits
    log.info("Loading trunk from run %s …", args.trunk_run_id)
    trunk, loaders, params = reconstruct_from_run(
        args.trunk_run_id,
        split=["train", "val", "test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=DEVICE,
    )
    trunk.eval()
    log.info("Trunk loaded: %d params", sum(p.numel() for p in trunk.parameters()))

    # 2. Build HLA encoder + feature matrices per split
    log.info("Building HLA features (encoding=%s) …", args.hla_encoding)
    encoder = build_encoder(args.hla_encoding)

    lf_loaders: dict[str, DataLoader] = {}
    for split, loader in loaders.items():
        dataset = loader.dataset
        subject_ids = dataset._subject_ids.numpy()
        hla_features = (build_hla_features_shuffled(subject_ids, encoder) if args.shuffle_hla
                        else build_hla_features(subject_ids, encoder))
        if args.null_hla:                       # control: same head, no HLA information
            hla_features = hla_features * 0
        lf_dataset = LateFusionDataset(dataset, hla_features)
        lf_collate = LateFusionCollateFn(loader.collate_fn)
        lf_loaders[split] = DataLoader(
            lf_dataset,
            batch_size=args.batch_size if split == "train" else args.eval_batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lf_collate,
        )
        log.info("  %s: %d subjects, d_hla=%d", split, len(lf_dataset), encoder.n_features)

    # 3. Build head + model
    predicted_vocab_sizes = {
        dname: trunk.embed._domain_vocab_sizes[dname]
        for dname in trunk.predicted_domains
    }
    hidden_dims = (
        [int(x) for x in args.head_hidden_dims.split(",")]
        if args.head_hidden_dims
        else None
    )
    if args.residual_blocks:
        # Stage 2: start from a stage-1 factorised ("separate") head and add zero-initialised blocks
        d_model = trunk.embed._get_domain_weight(next(iter(predicted_vocab_sizes))).shape[1]
        head = HLAResidualStream(encoder.n_features, predicted_vocab_sizes, d_model,
                                 n_blocks=args.residual_blocks, n_head=args.block_heads, block_type=args.block_type)
        if args.init_from:
            from utils.mlflow_utils import get_checkpoint_path
            sd = torch.load(get_checkpoint_path(args.init_from), map_location="cpu", weights_only=False)["state_dict"]
            missing, unexpected = head.load_state_dict({k: v for k, v in sd.items() if k.split(".")[0] in ("E", "U")}, strict=False)
            assert not unexpected and all(k.startswith("blocks.") for k in missing), (missing, unexpected)
            log.info("Loaded E, U from stage-1 run %s", args.init_from)
        model = DelphiLateFusionResidual(trunk=trunk, head=head).to(DEVICE)
    else:
        head = build_hla_head(encoder, predicted_vocab_sizes, hidden_dims=hidden_dims, rank=args.head_rank,
                              factorized=args.head_factorized, trunk=trunk)
        model = DelphiLateFusion(trunk=trunk, head=head).to(DEVICE)
    log.info(
        "Head: %s → %d params",
        f"ResidualStream(E, U' + {args.residual_blocks} {args.block_type} block(s), {args.block_heads} heads)" if args.residual_blocks
        else f"Factorized[{args.head_factorized} U]({encoder.n_features} → E → d_model → U → {sum(predicted_vocab_sizes.values())})" if args.head_factorized
        else f"MLP({encoder.n_features}, {hidden_dims}, {sum(predicted_vocab_sizes.values())})" if hidden_dims
        else f"PerLocusLowRank({len(encoder.loci)} loci x ESM2 → {args.head_rank} each → {sum(predicted_vocab_sizes.values())})" if args.hla_encoding == "esm2_locus"
        else f"LowRankLinear({encoder.n_features} → {args.head_rank} → {sum(predicted_vocab_sizes.values())})" if args.head_rank
        else f"Linear({encoder.n_features}, {sum(predicted_vocab_sizes.values())})",
        sum(p.numel() for p in head.parameters()),
    )

    if args.dryrun:
        log.info("--dryrun: exiting before training.")
        return

    exp_name = args.experiment_name or "late-fusion-hla"

    if args.skip_training:
        # ── Load best checkpoint from an existing MLflow run ──────────────────
        import mlflow as _mlflow, tempfile
        runs = _mlflow.search_runs(
            experiment_names=[exp_name],
            filter_string=f"tags.mlflow.runName = '{args.run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if len(runs) == 0:
            log.error("No run found with name '%s' in '%s'. Exiting.", args.run_name, exp_name)
            return
        existing_run_id = runs.iloc[0]["run_id"]
        client = _mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = client.download_artifacts(
                existing_run_id, "checkpoints/best_model.pt", tmp
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        head.load_state_dict(ckpt["state_dict"])
        log.info(
            "Loaded head checkpoint from epoch %s (val=%.4f)",
            ckpt.get("metadata", {}).get("epoch", "?"),
            ckpt.get("metadata", {}).get("val_ce_loss", float("nan")),
        )
        model = DelphiLateFusion(trunk=trunk, head=head).to(DEVICE)
        best_val_loss = ckpt.get("metadata", {}).get("val_ce_loss", float("nan"))
        mlf = MLFlowLogger(experiment_name=exp_name, run_name=args.run_name)
        mlf.start(resume_run_id=existing_run_id)

    else:
        # ── Normal training path ──────────────────────────────────────────────
        # 4. Optimiser + scheduler (head params only)
        lr = args.lr
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=lr / 10.0
        )

        # 5. MLflow
        mlf = MLFlowLogger(experiment_name=exp_name, run_name=args.run_name)
        mlf.start()
        _enc_to_setup = {"2field": "2field__nodrop__bidir", "1field": "1field__nodrop__bidir",
                         "esm2": "esm2_raw__nodrop__bidir"}
        mlf.log_params({
            "trunk_run_id":     args.trunk_run_id,
            "hla_encoding":     args.hla_encoding,
            "setup":            _enc_to_setup.get(args.hla_encoding, args.hla_encoding),
            "d_hla":            encoder.n_features,
            "head_hidden_dims": args.head_hidden_dims or "linear",
            "head_rank":        args.head_rank or "full",
            "head_factorized":  args.head_factorized or "none",
            "residual_blocks":  args.residual_blocks,
            "block_heads":      args.block_heads,
            "block_type":       args.block_type,
            "init_from":        args.init_from or "none",
            "null_hla":         args.null_hla,
            "shuffle_hla":      args.shuffle_hla,
            "predicted_domains": list(trunk.predicted_domains),
            "lr":               lr,
            "weight_decay":     args.weight_decay,
            "batch_size":       args.batch_size,
            "max_epochs":       args.max_epochs,
            "patience":         args.patience,
            "grad_clip":        args.grad_clip,
        })
        mlf.log_run_metadata(cwd=DELPHI_DIR)

        # 6. Training loop
        early_stopping = EarlyStopping(patience=args.patience, mode="min")
        best_val_loss = float("inf")

        for epoch in range(args.max_epochs):
            train_loss = run_epoch(
                model, lf_loaders["train"], DEVICE,
                optimizer=optimizer, grad_clip=args.grad_clip,
            )
            val_loss = run_epoch(model, lf_loaders["val"], DEVICE)
            scheduler.step()

            mlf.log_metrics(
                {"train_ce_loss": train_loss, "val_ce_loss": val_loss,
                 "lr": scheduler.get_last_lr()[0]},
                step=epoch,
            )
            log.info("Epoch %4d | train %.4f | val %.4f", epoch, train_loss, val_loss)

            should_stop, improved = early_stopping.step(val_loss)
            if improved:
                best_val_loss = val_loss
                mlf.save_model(
                    model=head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={
                        "epoch":        epoch,
                        "val_ce_loss":  val_loss,
                        "trunk_run_id": args.trunk_run_id,
                        "hla_encoding": args.hla_encoding,
                    },
                    filename=f"checkpoint_epoch{epoch:04d}.pt",
                    symlink_as="best_model.pt",
                )
            if should_stop:
                log.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val_loss)
                break

    # 7. AUC evaluation (shared by both paths)
    if args.compute_aucs:
        import copy
        from auc.late_fusion_aucs import evaluate_aucs_late_fusion, _load_focus_token_ids

        focus_token_ids = None
        if args.focus_list:
            disease_meta = (
                Path(trunk.config.domains["diseases"].path) / "token_metadata.tsv"
            )
            focus_token_ids = _load_focus_token_ids(args.focus_list, disease_meta)
            log.info("Focus diseases: %d token IDs from %s", len(focus_token_ids), args.focus_list)

        # Rebuild test loader with fixed block_size for AUC index arithmetic
        auc_collate = copy.copy(lf_loaders["test"].collate_fn)
        auc_collate.delphi_collate.block_size = args.eval_block_size
        auc_loader = DataLoader(
            lf_loaders["test"].dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=auc_collate,
        )

        auc_df = evaluate_aucs_late_fusion(
            model=model,
            test_loader=auc_loader,
            focus_token_ids=focus_token_ids,
            block_size=args.eval_block_size,
            n_jobs=8,
            logger=mlf,
            output_file="aucs_autoimmune.csv" if args.focus_list else "aucs.csv",
        )
        log.info("AUC summary:\n%s", auc_df.groupby("domain")["auc_delong"].describe().to_string())

    mlf.end()
    log.info("Done. Best val CE loss: %.4f", best_val_loss)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DelphiLateFusion (frozen trunk + HLA head)"
    )

    # Required
    p.add_argument(
        "--trunk_run_id", "--trunk-run-id", required=True,
        help="MLflow run ID of the pretrained no-HLA Delphi trunk",
    )

    # HLA encoding
    p.add_argument(
        "--hla_encoding", "--hla-encoding", default="2field",
        choices=["1field", "2field", "esm2", "esm2_locus"],
        help=(
            "HLA representation: 'multi-hot 1-field (138-dim), "
            "'2field' multi-hot 2-field (359-dim), "
            "'esm2' mean-pooled ESM2-15B embeddings (5120-dim). "
            "Default: 2field"
        ),
    )

    # Head architecture
    p.add_argument(
        "--head_hidden_dims", "--head-hidden-dims", default=None,
        help="Comma-separated hidden layer sizes for MLP head, e.g. '512,256'. "
             "None → single linear layer.",
    )

    p.add_argument(
        "--head_rank", "--head-rank", default=None, type=int,
        help="Low-rank linear head: d_hla → R → outputs (linear, no activation). "
             "None → full-rank linear layer. Incompatible with --head_hidden_dims.",
    )

    p.add_argument(
        "--head_factorized", "--head-factorized", default=None, choices=["tied", "separate"],
        help="Factorised head contrib = U·E·x with E: d_hla → trunk n_embd. 'tied': U is the trunk's "
             "frozen output matrix (HLA added to the trunk's final residual stream); 'separate': U learned.",
    )

    p.add_argument("--residual_blocks", "--residual-blocks", default=0, type=int,
                   help="Stage 2: number of zero-initialised attention+MLP blocks on the HLA stream (reads trunk states)")
    p.add_argument("--block_heads", "--block-heads", default=4, type=int)
    p.add_argument("--shuffle_hla", "--shuffle-hla", action="store_true",
                   help="Control: each subject gets a fixed random donor's HLA (hla_donor_map, seed 0)")
    p.add_argument("--block_type", "--block-type", default="cross", choices=["cross", "self", "allele"],
                   help="cross: HLA stream attends to trunk states; self: one stream h + E·x with causal self-attention; "
                        "allele: self-attention between allele tokens, then cross-attention from trunk states to them")
    p.add_argument("--null_hla", "--null-hla", action="store_true",
                   help="Control: zero the HLA features (with --residual_blocks: what the blocks gain from the trunk states alone)")
    p.add_argument("--init_from", "--init-from", default=None,
                   help="Stage-1 run (factorised 'separate' head, same encoding) to load E and U from")

    # Optimiser
    p.add_argument("--lr", default=1e-3, type=float, help="Peak learning rate (default: 1e-3)")
    p.add_argument("--weight_decay", "--weight-decay", default=1e-2, type=float)
    p.add_argument("--grad_clip", "--grad-clip", default=1.0, type=float)

    # Data / training loop
    p.add_argument("--batch_size", "--batch-size", default=256, type=int)
    p.add_argument("--eval_batch_size", "--eval-batch-size", default=512, type=int)
    p.add_argument("--num_workers", "--num-workers", default=4, type=int)
    p.add_argument("--max_epochs", "--max-epochs", default=500, type=int)
    p.add_argument("--patience", default=20, type=int)

    # MLflow
    p.add_argument("--experiment_name", "-x", default=None)
    p.add_argument("--run_name", default=None)

    # AUC
    p.add_argument(
        "--compute_aucs", "--compute-aucs", action="store_true",
        help="Compute AUCs on the test set after training",
    )
    p.add_argument(
        "--focus_list", "--focus-list", default=None,
        help="Path to a focus-disease list file (ICD-10 codes, one per line). "
             "If provided, AUC computation is restricted to these diseases.",
    )
    p.add_argument(
        "--eval_block_size", "--eval-block-size", default=128, type=int,
        help="Block size used for AUC evaluation collate (default: 128)",
    )

    # Misc
    p.add_argument(
        "--dryrun", action="store_true",
        help="Print config summary and exit without training",
    )
    p.add_argument(
        "--skip_training", "--skip-training", action="store_true",
        help="Skip training; load best checkpoint from an existing MLflow run "
             "(matched by --experiment_name + --run_name) and go straight to AUC evaluation.",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
