"""
AUC evaluation for Delphi.

Computes per-disease, per-sex, per-age-range AUCs using the output
embeddings from the model and the tied embedding weights.

Usage (from training script):

    from evaluation.aucs import evaluate_aucs
    auc_df = evaluate_aucs(model, test_loader, block_size=96)

Or with MLflow logging:

    evaluate_aucs(model, test_loader, block_size=96, run_id=run_id, logger=logger)
"""

from __future__ import annotations

import gc
import warnings
from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from delphi.model import Delphi, DelphiBatch

warnings.filterwarnings("ignore")

DAYS_PER_YEAR = 365.25
SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a + 5) for a in range(0, 85, 5)]
EMPTY = (np.array([], dtype=int), np.array([], dtype=int))


# ═══════════════════════════════════════════════════════════════════════════════
#  Token DataFrame construction
# ═══════════════════════════════════════════════════════════════════════════════


def batch_to_tokens_df(batch: DelphiBatch, int_to_domain: dict[int, str], domain_offsets: dict[int, int]):
    """
    Convert a DelphiBatch into a flat DataFrame of tokens.
    Uses global_token_ids and recovers local IDs via offsets.
    """
    B, T = batch.global_token_ids.shape

    subject_idx = torch.arange(B).unsqueeze(1).expand(B, T)
    seq_idx = torch.arange(T).unsqueeze(0).expand(B, T)
    subject_id = batch.subject_ids.unsqueeze(1).expand(B, T)

    domain_ids: np.ndarray = batch.domain_ids.reshape(-1).cpu().numpy()
    global_ids: np.ndarray = batch.global_token_ids.reshape(-1).cpu().numpy()

    # Recover local token IDs
    local_ids = global_ids.copy()
    for d_int, offset in domain_offsets.items():
        mask = domain_ids == d_int
        local_ids[mask] -= offset

    domains = [int_to_domain.get(d, f"unknown_{d}") for d in domain_ids]

    df = pd.DataFrame(
        {
            "subject_idx": subject_idx.reshape(-1).cpu().numpy(),
            "seq_idx": seq_idx.reshape(-1).cpu().numpy(),
            "subject_id": subject_id.reshape(-1).cpu().numpy(),
            "age": batch.ages.reshape(-1).cpu().numpy(),
            "token_id": local_ids,
            "global_token_id": global_ids,
            "domain_id": domain_ids,
            "domain": domains,
        }
    )

    return df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)


def get_subject_sex(df):
    return df.query('domain == "sex"').groupby("subject_idx")["token_id"].first().to_dict()


def add_sex_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(sex=lambda df: df["subject_idx"].map(get_subject_sex(df)))


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sex_column(df)
    df["previous_idx"] = df.global_idx.apply(lambda x: x - 1)
    df["age_previous"] = df.age.shift(1)
    age_bins = [DAYS_PER_YEAR * i - 1 for i in range(0, 90, 5)]
    df["age_bin"] = pd.cut(df["age_previous"], age_bins)
    df["age_bin"] = (df.age_bin.cat.codes + 1) * 5
    df = df[df.age_bin.notna()]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Case/Control extraction
# ═══════════════════════════════════════════════════════════════════════════════

CaseCtrlKey = tuple[str, tuple[int, int], str, int]


def extract_case_ctrl_for_sex_age(
    sex: str,
    sex_id: int,
    a0: int,
    a1: int,
    all_sids: dict[int, np.ndarray],
    tok_subj: np.ndarray,
    tok_sex: np.ndarray,
    tok_gidx: np.ndarray,
    age_masks_sa: Any,
    ctrl_subjects_sex: dict[tuple[str, int], np.ndarray],
    disease_idx_sex: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[CaseCtrlKey, pd.Series], dict[CaseCtrlKey, np.ndarray]]:
    mask = age_masks_sa & (tok_sex == sex_id)
    subj = tok_subj[mask]
    gidx = tok_gidx[mask]

    df = pd.DataFrame({"subject_id": subj, "global_idx": gidx})
    subset = (
        df.groupby("subject_id", group_keys=False)
        .sample(n=1, random_state=42)
        .set_index("subject_id")
        .reindex(all_sids[sex_id])
        .sort_index()
    )

    local_case: dict[CaseCtrlKey, np.ndarray] = {}
    local_ctrl: dict[CaseCtrlKey, pd.Series] = {}

    for dd in ctrl_subjects_sex:
        local_ctrl[(sex, (a0, a1), *dd)] = subset.loc[ctrl_subjects_sex[dd]].dropna().global_idx

        gidx_arr, age_arr = disease_idx_sex[dd]
        age_mask = (age_arr > a0 * DAYS_PER_YEAR) & (age_arr <= a1 * DAYS_PER_YEAR)
        local_case[(sex, (a0, a1), *dd)] = gidx_arr[age_mask]

    return local_ctrl, local_case


# ═══════════════════════════════════════════════════════════════════════════════
#  AUC computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_auc_from_indices(
    logits_for_token: Any, cases: Any, controls: Any, block_size: int
) -> dict[str, float | np.ndarray | None]:
    """Slice subject logits by flat indices and compute AUC stats.

    Args:
        logits_for_token: 2D array-like [n_subjects, block_size] of logit scores
        cases: array-like of flat indices for case subjects
        controls: array-like of flat indices for control subjects
        block_size: sequence length used to convert flat to (subject, position) indices
    """
    from auc.scripts.auc_utils import compute_all_stats

    return compute_all_stats(
        logits_for_token[cases // block_size, cases % block_size - 1],
        logits_for_token[controls // block_size, controls % block_size],
    )


def process_disease(logits_for_token, token_id, lookup_disease, block_size):
    rows = []
    for age_range in AGE_RANGES:
        for sex in ["female", "male"]:
            case, ctrl = lookup_disease.get((sex, age_range), EMPTY)

            stats: dict[str, Any] = compute_auc_from_indices(logits_for_token, case, ctrl, block_size)
            stats.update(
                {
                    "domain": token_id[0],
                    "token_id": token_id[1],
                    "age_start": age_range[0],
                    "age_end": age_range[1],
                    "sex": sex,
                    "n_case": len(case),
                    "n_ctrl": len(ctrl),
                }
            )
            rows.append(stats)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation function
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_aucs(
    model: Delphi,
    test_loader,
    block_size: int | None = None,
    n_jobs: int = -1,
    run_id: str | None = None,
    logger=None,
    output_file: str = "aucs.csv",
) -> pd.DataFrame:
    """
    Evaluate AUCs for all predicted diseases.

    Parameters
    ----------
    model : Delphi
        Trained model (already on device).
    test_loader : DataLoader
        Test dataloader with DelphiCollateFn.
    block_size : int, optional
        Override model block_size. If None, uses model.block_size.
    n_jobs : int
        Number of parallel jobs for AUC computation.
    run_id : str, optional
        MLflow run ID for logging artifacts.
    logger : MLFlowLogger, optional
        Logger for saving results.
    output_file : str
        Filename for the AUC results.

    Returns
    -------
    auc_df : pd.DataFrame
    """

    device = model.device
    block_size = block_size or model.block_size

    int_to_domain = model.int_to_domain
    domain_offsets = model.domain_offsets

    # ── 1. Forward pass: collect embeddings and token info ────────────
    print("Extracting output embeddings...")
    all_tokens_dfs: list[pd.DataFrame] = []
    _all_output_embeddings: list[torch.Tensor] = []
    batch_offset = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forward pass"):
            batch = batch.to(device)

            _, _, h = model(batch, return_embeddings=True)
            _all_output_embeddings.append(h.half())

            tokens_df = batch_to_tokens_df(batch, int_to_domain, domain_offsets)
            # Offset subject_idx by batch position
            tokens_df["subject_idx"] += batch_offset
            batch_offset += batch.batch_size

            all_tokens_dfs.append(tokens_df)

            del h
            gc.collect()

    all_output_embeddings = torch.cat(_all_output_embeddings, dim=0)  # [N_subj, T, n_embd]
    tokens_df = pd.concat(all_tokens_dfs, ignore_index=True)
    tokens_df = tokens_df.reset_index().rename(columns={"index": "global_idx"})
    tokens_df["subject_idx"] = tokens_df.global_idx // block_size
    tokens_df = add_sex_column(tokens_df)
    tokens_df = add_age_bin(tokens_df)

    del all_tokens_dfs

    # ── 2. Build case/control indices ─────────────────────────────────
    print("Building case/control indices...")

    # Precompute age masks
    age_masks = {}
    for a0, a1 in AGE_RANGES:
        age_masks[(a0, a1)] = tokens_df.age.between(
            a0 * DAYS_PER_YEAR, a1 * DAYS_PER_YEAR, inclusive="right"
        ).to_numpy()

    # List all predicted (domain, token_id) pairs
    predicted_tokens = [
        (dname, tid) for dname in model.predicted_domains for tid in range(model.embed._domain_vocab_sizes[dname])
    ]

    # Group tokens by sex and disease
    by_disease_dfs: dict[int, dict[tuple[str, int], pd.DataFrame]] = {}
    case_subjects: dict[str, dict[tuple[str, int], pd.Series]] = dict(female={}, male={})
    ctrl_subjects: dict[str, dict[tuple[str, int], np.ndarray]] = dict(female={}, male={})

    for sex, sex_id in SEX_TOKENS.items():
        tokens_sex = tokens_df.query("sex == @sex_id")
        _by_disease = cast(dict[tuple[str, int], pd.DataFrame], dict(list(tokens_sex.groupby(["domain", "token_id"]))))
        by_disease_dfs[sex_id] = _by_disease

        unique_subjects = tokens_sex.subject_id.unique()

        for domain_name, token_id in predicted_tokens:
            if (domain_name, token_id) not in _by_disease:
                continue
            case_subjects[sex][domain_name, token_id] = _by_disease[(domain_name, token_id)].subject_id
            cases_ids = set(case_subjects[sex][(domain_name, token_id)])
            ctrl_subjects[sex][(domain_name, token_id)] = unique_subjects[~pd.Series(unique_subjects).isin(cases_ids)]

    # Precompute arrays for parallel extraction
    all_sids = {sex_id: tokens_df.loc[tokens_df.sex == sex_id, "subject_id"].unique() for sex_id in SEX_TOKENS.values()}

    tok_subj = tokens_df["subject_id"].to_numpy()
    tok_sex = tokens_df["sex"].to_numpy()
    tok_gidx = tokens_df["global_idx"].to_numpy()

    disease_idx = {
        sex_id: {
            dd: (df["global_idx"].to_numpy(), df["age_previous"].to_numpy())
            for dd, df in by_disease_dfs[sex_id].items()
        }
        for sex_id in SEX_TOKENS.values()
    }

    del by_disease_dfs, tokens_df

    # Parallel extraction
    print("Extracting case/control indices...")
    tasks = [(sex, sex_id, a0, a1) for sex, sex_id in SEX_TOKENS.items() for (a0, a1) in AGE_RANGES]

    _cc_args = (
        (
            sex,
            sex_id,
            a0,
            a1,
            all_sids,
            tok_subj,
            tok_sex,
            tok_gidx,
            age_masks[(a0, a1)],
            ctrl_subjects[sex],
            disease_idx[sex_id],
        )
        for sex, sex_id, a0, a1 in tqdm(tasks, desc="Case/ctrl indices")
    )
    if n_jobs == 1:
        results = [extract_case_ctrl_for_sex_age(*a) for a in _cc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(extract_case_ctrl_for_sex_age)(*a) for a in _cc_args
        )

    cc_results = cast(list[tuple[dict[Any, Any], dict[Any, Any]]], results)
    ctrl_indices, case_indices = {}, {}
    for local_ctrl, local_case in cc_results:
        ctrl_indices.update(local_ctrl)
        case_indices.update(local_case)
    del results

    # Build lookup: (domain_name, token_id) -> {(sex, age_range) -> (cases, controls)}
    case_ctrl_lookup: defaultdict[tuple[str, int], dict[tuple[str, tuple[int, int]], tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(dict)
    )
    for key, cases in case_indices.items():
        if key not in ctrl_indices or len(cases) == 0:
            continue
        sex, age_range, domain_name, token_id = key
        case_ctrl_lookup[(domain_name, token_id)][(sex, age_range)] = (cases, ctrl_indices[key].to_numpy(dtype=int))

    del case_indices, ctrl_indices
    gc.collect()

    # ── 3. Compute AUCs ──────────────────────────────────────────────
    print("Computing AUCs...")

    _auc_args = (
        (
            (all_output_embeddings @ model.embed._get_domain_weight(domain_name)[token_id].half())
            .detach()
            .cpu()
            .numpy(),
            (domain_name, token_id),
            case_ctrl_lookup.get((domain_name, token_id), {}),
        )
        for domain_name, token_id in tqdm(predicted_tokens, desc="AUC computation")
    )
    if n_jobs == 1:
        results = [process_disease(*a, block_size=block_size) for a in _auc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_disease)(*a, block_size=block_size) for a in _auc_args
        )

    # ── 4. Assemble results ──────────────────────────────────────────
    auc_results = cast(list[list[dict[str, Any]]], results)
    auc_data = [row for sublist in auc_results for row in sublist]
    auc_df = pd.DataFrame(auc_data)

    if "auc_bootstrap_mean" in auc_df.columns:
        auc_df = auc_df.drop(["auc_bootstrap_mean", "auc_bootstrap_std"], axis=1)

    if run_id is not None:
        auc_df = auc_df.assign(runid=run_id, block_size=block_size)

    columns_order = [
        c
        for c in [
            "runid",
            "domain",
            "token_id",
            "sex",
            "age_start",
            "age_end",
            "n_case",
            "n_ctrl",
            "auc_delong",
            "auc_delong_var",
            "mann_u",
            "mann_p",
            "block_size",
        ]
        if c in auc_df.columns
    ]
    auc_df = auc_df[columns_order]

    # ── 5. Log if logger provided ────────────────────────────────────
    if logger is not None:
        logger.log_df_as_artifact(
            auc_df,
            filename=output_file,
            artifact_path="aucs",
        )
        print(f"AUC results logged as artifact: aucs/{output_file}")

    print(
        f"AUC evaluation complete: {len(auc_df)} rows, "
        f"{auc_df['domain'].nunique()} domains, "
        f"{len(predicted_tokens)} tokens"
    )

    return auc_df
