"""
Permutation SHAP — source-to-disease attribution.

Estimates the Shapley value of each source token on the model's prediction
of every disease/death token that occurs after it.

    shap_values[s, t] = Shapley contribution of source token s
                        on the logit of target token t
                        (only defined when age[s] < age[t])

Source domain is controlled by --source_domain:
  - "diseases_death" (default): disease/death → disease/death (original behaviour)
  - "hla_alleles":              HLA allele → disease/death

All tokens outside the source domain are kept fixed across all coalition
evaluations — only source tokens are permuted in/out.

Output (.pkl per run):
    {
      "run_id":           str,
      "n_permutations":   int,
      "source_domain":    str,
      "records": [
        {
          "subject_id":    int,
          "src_token_ids": np.ndarray (S,)    local token IDs of source tokens
          "src_domain_ids":np.ndarray (S,)    domain int codes of source tokens
          "src_ages":      np.ndarray (S,)    ages in days of source tokens
          "tgt_token_ids": np.ndarray (T,)    local token IDs of target tokens
          "tgt_domain_ids":np.ndarray (T,)    domain int codes of target tokens
          "tgt_ages":      np.ndarray (T,)    ages in days of target tokens
          "shap_values":   np.ndarray (S, T)  phi[s, t] where age[s] < age[t]
        }, ...
      ]
    }

Usage:
    python shap/disease_shap.py --run_id RUN_ID [RUN_ID ...]
    python shap/disease_shap.py --experiment_id EXP_ID [--test_fold 1 2 3]
    python shap/disease_shap.py --run_id RUN_ID --source_domain hla_alleles
    python shap/disease_shap.py --run_id RUN_ID --dry_run
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

DELPHI_DIR = Path("/nfs/research/birney/users/bonazzola/repos/delphis/delphi-refactor")
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

_INVOCATION_CWD = Path.cwd()
os.chdir(DELPHI_DIR)

from data.dataset import DelphiCollateFn, DelphiDataset
from utils import reconstruct_model
from utils.mlflow_utils import setup_mlflow
from utils.utils import read_ids

setup_mlflow()

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PADDING_AGE  = -10000.0
OUTPUT_DIR   = DELPHI_DIR / "shap" / "output_disease_shap"


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------

class _NullAgeSampler:
    """No-op sampler — disables no-event token insertion for clean coalitions."""
    def sample(self, max_ages: torch.Tensor):
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)


def _make_shap_collate(model) -> DelphiCollateFn:
    continuous_domains = {
        name: cfg.n_latent_tokens or 1
        for name, cfg in model.domain_cfg.items()
        if cfg.type == "continuous"
    }
    return DelphiCollateFn(
        age_sampler=_NullAgeSampler(),
        block_size=model.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
    )


# ---------------------------------------------------------------------------
# Coalition item builder
# ---------------------------------------------------------------------------

def _build_coalition_item(
    item: dict,
    src_positions: list[int],
    coalition: set[int],
    padding_domain_id: int,
) -> dict:
    """
    Copy of item keeping all non-source tokens plus only the source tokens
    whose index into src_positions is in coalition.
    """
    real_count   = int(item["real_count"])
    block_size   = item["domain_ids"].shape[0]

    keep = torch.ones(real_count, dtype=torch.bool)
    for pos in src_positions:
        keep[pos] = False
    for i in coalition:
        keep[src_positions[i]] = True

    kept_doms = item["domain_ids"][:real_count][keep]
    kept_toks = item["local_token_ids"][:real_count][keep]
    kept_ages = item["ages"][:real_count][keep]

    order = kept_ages.argsort()
    kept_doms = kept_doms[order]
    kept_toks = kept_toks[order]
    kept_ages = kept_ages[order]

    n = min(len(kept_doms), block_size)
    new = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    new["domain_ids"]      = torch.full((block_size,), padding_domain_id, dtype=item["domain_ids"].dtype)
    new["local_token_ids"] = torch.zeros(block_size, dtype=item["local_token_ids"].dtype)
    new["ages"]            = torch.full((block_size,), PADDING_AGE, dtype=item["ages"].dtype)
    new["domain_ids"][:n]      = kept_doms[:n]
    new["local_token_ids"][:n] = kept_toks[:n]
    new["ages"][:n]            = kept_ages[:n]
    new["real_count"]      = torch.tensor(n, dtype=item["real_count"].dtype)
    return new


# ---------------------------------------------------------------------------
# Model evaluation for a batch of coalition sequences
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_logits(
    model,
    batch,
    target_weights: torch.Tensor,  # [T, n_embd]
    target_ages: torch.Tensor,      # [T]
) -> torch.Tensor:                  # [B, T]
    """
    For each sequence in the batch and each target disease/death token,
    return the logit at the last context position before target_ages[t].
    """
    _, _, h = model(batch, return_embeddings=True)   # [B, seq_len, n_embd]
    B        = batch.ages.shape[0]
    T        = len(target_ages)
    result   = torch.zeros(B, T)
    ages     = batch.ages  # [B, seq_len]

    for j in range(T):
        a_t   = target_ages[j]
        valid = (ages < a_t) & (ages > PADDING_AGE + 1.0)   # [B, seq_len]
        has_valid = valid.any(dim=1)                          # [B]

        age_masked = ages.clone()
        age_masked[~valid] = -float("inf")
        last_pos = age_masked.argmax(dim=1)                  # [B]

        h_ctx  = h[torch.arange(B), last_pos].float()        # [B, n_embd]
        logit  = (h_ctx @ target_weights[j]) * has_valid.float()
        result[:, j] = logit.cpu()

    return result


# ---------------------------------------------------------------------------
# Per-subject permutation SHAP
# ---------------------------------------------------------------------------

def compute_subject_shap(
    model,
    item: dict,
    src_positions: list[int],
    src_ages: torch.Tensor,          # [S] on CPU
    target_weights: torch.Tensor,    # [T_d, n_embd] on DEVICE
    target_ages: torch.Tensor,       # [T_d] on CPU
    collate,
    n_permutations: int,
    step_batch_size: int,
    padding_domain_id: int,
) -> np.ndarray:
    """Return (S, T_d) SHAP matrix: phi[s, t] = Shapley value of source s on target t."""
    S   = len(src_positions)
    T_d = len(target_ages)
    if S == 0 or T_d == 0:
        return np.zeros((S, T_d), dtype=np.float32)

    perms = np.stack([np.random.permutation(S) for _ in range(n_permutations)])  # [n_perm, S]
    phi   = np.zeros((S, T_d), dtype=np.float32)

    # causal_mask[s, t] = True iff source s occurs before target t
    causal_mask = src_ages.numpy()[:, None] < target_ages.numpy()[None, :]  # [S, T_d]

    prev_logits = None
    for step in range(S + 1):
        items_step = [
            _build_coalition_item(item, src_positions, set(perms[i, :step].tolist()), padding_domain_id)
            for i in range(n_permutations)
        ]

        parts = []
        for start in range(0, n_permutations, step_batch_size):
            batch = collate(items_step[start : start + step_batch_size]).to(DEVICE)
            parts.append(_eval_logits(model, batch, target_weights, target_ages.to(DEVICE)))
        logits_step = torch.cat(parts, dim=0).numpy()   # [n_perm, T_d]

        if prev_logits is not None:
            marginals = logits_step - prev_logits        # [n_perm, T_d]
            s_added   = perms[:, step - 1]               # [n_perm]

            for s in range(S):
                perm_mask = s_added == s
                if not perm_mask.any():
                    continue
                phi[s] += marginals[perm_mask].sum(axis=0) * causal_mask[s]

        prev_logits = logits_step

    phi /= n_permutations
    return phi


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(
    run_id: str,
    n_permutations: int,
    step_batch_size: int,
    output_path: Path,
    subjects_include: Optional[set] = None,
    partition_idx: int = 0,
    n_partitions: int = 1,
    source_domain: str = "diseases_death",
    target_disease_ids: Optional[set] = None,
) -> None:
    logger.info(f"Loading model  run_id={run_id[:12]}  partition={partition_idx}/{n_partitions}  source={source_domain}")
    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to(DEVICE).eval()

    if subjects_include is not None:
        test_ids = [sid for sid in test_ids if int(sid) in subjects_include]
        logger.info(f"  {len(test_ids)} subjects after filtering")

    if n_partitions > 1:
        test_ids = sorted(test_ids)
        test_ids = np.array_split(test_ids, n_partitions)[partition_idx].tolist()
        logger.info(f"  partition {partition_idx}/{n_partitions}: {len(test_ids)} subjects")

    disease_int = model.domain_to_int.get("diseases")
    death_int   = model.domain_to_int.get("death")
    if disease_int is None:
        raise ValueError("Model has no 'diseases' domain.")
    tgt_ints = {d for d in [disease_int, death_int] if d is not None}

    if source_domain == "hla_alleles":
        hla_int = model.domain_to_int.get("hla_alleles")
        if hla_int is None:
            raise ValueError("Model has no 'hla_alleles' domain.")
        src_ints = {hla_int}
    else:
        src_ints = tgt_ints

    padding_domain_id = model.domain_to_int["padding"]

    continuous_domains = {
        name: cfg.n_latent_tokens or 1
        for name, cfg in model.domain_cfg.items()
        if cfg.type == "continuous"
    }
    dataset = DelphiDataset(
        root=DELPHI_DIR / "data" / "transforms",
        domains_cfg=model.domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=model.block_size,
        subjects=test_ids,
        required_domains=["diseases"],
        no_event_token_rate=model.config.no_event_token_rate,
        no_event_insertion_mode=model.config.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
    )
    collate = _make_shap_collate(model)

    records = []
    for subj_idx in tqdm(range(len(dataset)), desc=f"Subjects ({run_id[:8]})"):
        item       = dataset[subj_idx]
        real_count = int(item["real_count"])

        domain_ids = item["domain_ids"][:real_count]
        token_ids  = item["local_token_ids"][:real_count]
        ages       = item["ages"][:real_count]

        src_mask = torch.zeros(real_count, dtype=torch.bool)
        for d_int in src_ints:
            src_mask |= (domain_ids == d_int)
        src_positions = src_mask.nonzero(as_tuple=True)[0].tolist()

        tgt_mask = torch.zeros(real_count, dtype=torch.bool)
        for d_int in tgt_ints:
            tgt_mask |= (domain_ids == d_int)
        if target_disease_ids is not None:
            tgt_mask &= (domain_ids == disease_int) & torch.isin(
                token_ids, torch.tensor(sorted(target_disease_ids))
            )
        tgt_positions = tgt_mask.nonzero(as_tuple=True)[0].tolist()

        if len(src_positions) == 0 or len(tgt_positions) == 0:
            continue
        # For disease→disease mode, require at least 2 tokens (need a causal pair)
        if source_domain == "diseases_death" and len(src_positions) < 2:
            continue

        src_domain_ids = domain_ids[src_positions]
        src_token_ids  = token_ids[src_positions]
        src_ages       = ages[src_positions]

        tgt_domain_ids = domain_ids[tgt_positions]
        tgt_token_ids  = token_ids[tgt_positions]
        tgt_ages       = ages[tgt_positions]

        # Embedding weights for target tokens
        weights = [
            model.embed._get_domain_weight(model.int_to_domain[int(dom_id)])[int(tok_id)].float()
            for dom_id, tok_id in zip(tgt_domain_ids.tolist(), tgt_token_ids.tolist())
        ]
        target_weights = torch.stack(weights).to(DEVICE)   # [T_d, n_embd]

        phi = compute_subject_shap(
            model=model,
            item=item,
            src_positions=src_positions,
            src_ages=src_ages,
            target_weights=target_weights,
            target_ages=tgt_ages,
            collate=collate,
            n_permutations=n_permutations,
            step_batch_size=step_batch_size,
            padding_domain_id=padding_domain_id,
        )

        records.append({
            "subject_id":    int(dataset._subject_ids[subj_idx].item()),
            "src_token_ids": src_token_ids.numpy(),
            "src_domain_ids":src_domain_ids.numpy(),
            "src_ages":      src_ages.numpy(),
            "tgt_token_ids": tgt_token_ids.numpy(),
            "tgt_domain_ids":tgt_domain_ids.numpy(),
            "tgt_ages":      tgt_ages.numpy(),
            "shap_values":   phi,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "run_id": run_id,
            "n_permutations": n_permutations,
            "source_domain": source_domain,
            "records": records,
        }, f)
    logger.info(f"Saved {len(records)} records → {output_path}")


# ---------------------------------------------------------------------------
# Related-run discovery
# ---------------------------------------------------------------------------

# Params that identify configuration vs. params that vary per fold/seed/run
_IGNORE_FOR_CONFIG = {
    "test_fold", "seed", "n_train", "n_val", "n_test", "n_params",
    "hostname", "batch_size", "ema_alpha", "attn_pdrop", "embd_pdrop",
    "resid_pdrop", "dropout", "bias", "max_epochs", "min_epochs",
    "patience", "optim_config", "attention_scheme_alias",
    "no_event_token_insertion_mode", "slurm_job_id",
}


def _config_key(params: dict) -> str:
    return "|".join(
        f"{k}={v}" for k, v in sorted(params.items())
        if k not in _IGNORE_FOR_CONFIG and not k.startswith("mlflow.")
    )


def find_related_runs(reference_run_id: str) -> pd.DataFrame:
    """
    Return a DataFrame of FINISHED runs in the same experiment that share the
    same configuration as reference_run_id (same params except fold/seed/metadata).
    Fetches params per-run via the MLflow API to avoid column-stripping issues.
    """
    ref = mlflow.get_run(reference_run_id)
    experiment_id = ref.info.experiment_id
    ref_key = _config_key(dict(ref.data.params))

    run_infos = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=500,
    )

    rows = []
    for run_id in run_infos["run_id"]:
        run = mlflow.get_run(run_id)
        params = dict(run.data.params)
        if _config_key(params) == ref_key:
            rows.append({
                "run_id":    run_id,
                "test_fold": params.get("test_fold"),
                "seed":      params.get("seed"),
            })

    df = pd.DataFrame(rows)
    if not df.empty and "test_fold" in df.columns:
        df["test_fold"] = pd.to_numeric(df["test_fold"], errors="coerce")
        df = df.sort_values("test_fold").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation SHAP — disease-to-disease.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_id",        nargs="+", metavar="RUN_ID",
                     help="One or more MLflow run IDs to process directly.")
    src.add_argument("--experiment_id", type=str,
                     help="MLflow experiment ID; selects all FINISHED runs.")
    src.add_argument("--generate_submission", type=str, metavar="RUN_ID",
                     help="Find all runs with the same config as this run_id "
                          "and write a sarray_params TSV.")

    parser.add_argument("--test_fold",      type=int, nargs="+", default=None,
                        help="With --experiment_id: restrict to these fold(s).")
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--batch_size",     type=int, default=256,
                        help="Permutations per model call.")
    parser.add_argument("--n_partitions",   type=int, default=1,
                        help="Split subjects into this many partitions (default: 1 = no split).")
    parser.add_argument("--partition_idx",  type=int, default=0,
                        help="Which partition to process (0-based).")
    parser.add_argument("--subjects",       type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR}).")
    parser.add_argument("--tsv_output",     type=str, default=None,
                        help="With --generate_submission: path for the TSV file.")
    parser.add_argument("--source_domain",  type=str, default="diseases_death",
                        choices=["diseases_death", "hla_alleles"],
                        help="Domain of tokens to permute as SHAP sources.")
    parser.add_argument("--target_disease_ids", type=int, nargs="+", default=None,
                        metavar="TOKEN_ID",
                        help="Restrict target tokens to these 'diseases' token IDs "
                             "(drops 'death' from targets). E.g. the 17 autoimmune "
                             "disease IDs used in the ablation/epistasis analyses.")
    parser.add_argument("--dry_run", "--dryrun", action="store_true")
    args = parser.parse_args()

    output_dir = Path(_INVOCATION_CWD / args.output_dir) if args.output_dir else OUTPUT_DIR

    # ------------------------------------------------------------------ #
    #  GENERATE SUBMISSION MODE                                            #
    # ------------------------------------------------------------------ #
    if args.generate_submission:
        ref_id = args.generate_submission
        logger.info(f"Finding runs with same config as {ref_id[:12]}…")
        related = find_related_runs(ref_id)

        if related.empty:
            raise SystemExit("No related runs found.")

        folds = related["test_fold"].dropna().astype(int).tolist() if "test_fold" in related.columns else ["?"]
        print(f"Found {len(related)} related run(s) across fold(s): {folds}")
        for _, row in related.iterrows():
            print(f"  fold={row.get('test_fold', '?')}  run_id={row['run_id']}")

        n_part = args.n_partitions
        rows = [
            {"run_id": row["run_id"], "partition_idx": p, "n_partitions": n_part,
             "n_permutations": args.n_permutations}
            for _, row in related.iterrows()
            for p in range(n_part)
        ]
        tsv_df = pd.DataFrame(rows)
        tsv_path = Path(args.tsv_output) if args.tsv_output else \
            DELPHI_DIR / "shap" / f"disease_shap_submission_{ref_id[:8]}.tsv"
        tsv_df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\nTSV ({len(tsv_df)} jobs) → {tsv_path}")
        print(f"\nSubmit with:")
        print(f"  source ~/repos/codon_helpers/slurm_functions.sh")
        print(f"  sarray_params shap/disease_shap.py {tsv_path} \\")
        print(f"    --output_dir {output_dir} \\")
        print(f"    --time=4:00:00 --mem=16G --cpus=2 --max-parallel=1000")
        return

    # ------------------------------------------------------------------ #
    #  NORMAL PROCESSING MODE                                              #
    # ------------------------------------------------------------------ #
    if args.run_id:
        run_ids = args.run_id
    else:
        runs = mlflow.search_runs(
            experiment_ids=[args.experiment_id],
            filter_string="status = 'FINISHED'",
        )
        if args.test_fold:
            runs = runs[runs["params.test_fold"].astype(float).isin(args.test_fold)]
        run_ids = runs["run_id"].tolist()
        if not run_ids:
            raise SystemExit("No FINISHED runs found.")

    subjects_include = None
    if args.subjects:
        subjects_include = set(read_ids(args.subjects))

    if args.dry_run:
        n_jobs = len(run_ids) * args.n_partitions
        print(f"Would submit {n_jobs} job(s): {len(run_ids)} run(s) × {args.n_partitions} partition(s)")
        for rid in run_ids:
            print(f"  {rid}")
        return

    target_disease_ids = set(args.target_disease_ids) if args.target_disease_ids else None

    src_tag = f"_src-{args.source_domain}" if args.source_domain != "diseases_death" else ""
    tgt_tag = "_tgt-autoimmune17" if target_disease_ids is not None else ""
    suffix = f"_part{args.partition_idx:04d}of{args.n_partitions}" if args.n_partitions > 1 else ""
    for run_id in run_ids:
        process_run(
            run_id=run_id,
            n_permutations=args.n_permutations,
            step_batch_size=args.batch_size,
            output_path=output_dir / f"{run_id}{src_tag}{tgt_tag}{suffix}.pkl",
            subjects_include=subjects_include,
            partition_idx=args.partition_idx,
            n_partitions=args.n_partitions,
            source_domain=args.source_domain,
            target_disease_ids=target_disease_ids,
        )


if __name__ == "__main__":
    main()
