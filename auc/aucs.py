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
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if (DELPHI_DIR := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

DAYS_PER_YEAR = 365.25
SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a + 5) for a in range(0, 85, 5)]
EMPTY = (np.array([], dtype=int), np.array([], dtype=int))


# ═══════════════════════════════════════════════════════════════════════════════
#  Token DataFrame construction
# ═══════════════════════════════════════════════════════════════════════════════

def batch_to_tokens_df(batch, int_to_domain: Dict[int, str], domain_offsets: Dict[int, int]):
    """
    Convert a DelphiBatch into a flat DataFrame of tokens.
    Uses global_token_ids and recovers local IDs via offsets.
    """
    B, T = batch.global_token_ids.shape

    subject_idx = torch.arange(B).unsqueeze(1).expand(B, T)
    seq_idx = torch.arange(T).unsqueeze(0).expand(B, T)
    subject_id = batch.subject_ids.unsqueeze(1).expand(B, T)

    domain_ids = batch.domain_ids.reshape(-1).cpu().numpy()
    global_ids = batch.global_token_ids.reshape(-1).cpu().numpy()

    # Recover local token IDs
    local_ids = global_ids.copy()
    for d_int, offset in domain_offsets.items():
        mask = domain_ids == d_int
        local_ids[mask] -= offset

    domains = [int_to_domain.get(d, f"unknown_{d}") for d in domain_ids]

    df = pd.DataFrame({
        "subject_idx": subject_idx.reshape(-1).cpu().numpy(),
        "seq_idx": seq_idx.reshape(-1).cpu().numpy(),
        "subject_id": subject_id.reshape(-1).cpu().numpy(),
        "age": batch.ages.reshape(-1).cpu().numpy(),
        "token_id": local_ids,
        "global_token_id": global_ids,
        "domain_id": domain_ids,
        "domain": domains,
    })

    if batch.cutoff_ages is not None:
        cutoff_age = batch.cutoff_ages.unsqueeze(1).expand(B, T)
        df["cutoff_age"] = cutoff_age.reshape(-1).cpu().numpy()

    return df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)


def get_subject_sex(df):
    return (
        df.query('domain == "sex"')
        .groupby("subject_idx")["token_id"]
        .first()
        .to_dict()
    )


def add_sex_column(df):
    return df.assign(sex=lambda df: df["subject_idx"].map(get_subject_sex(df)))


def add_age_bin(df):
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

def extract_case_ctrl_for_sex_age(
    sex, sex_id, a0, a1,
    all_sids, tok_subj, tok_sex, tok_gidx, age_masks_sa,
    ctrl_subjects_sex, disease_idx_sex,
):
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

    local_case, local_ctrl = {}, {}

    for dd in ctrl_subjects_sex.keys():
        local_ctrl[(sex, (a0, a1), *dd)] = (
            subset.loc[ctrl_subjects_sex[dd]].dropna().global_idx
        )

        gidx_arr, age_arr = disease_idx_sex[dd]
        age_mask = (age_arr > a0 * DAYS_PER_YEAR) & (age_arr <= a1 * DAYS_PER_YEAR)
        local_case[(sex, (a0, a1), *dd)] = gidx_arr[age_mask]

    return local_ctrl, local_case


# ═══════════════════════════════════════════════════════════════════════════════
#  AUC computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auc_from_indices(logits_for_token, cases, controls, block_size):

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

            stats = compute_auc_from_indices(logits_for_token, case, ctrl, block_size)
            stats.update({
                "domain": token_id[0],
                "token_id": token_id[1],
                "age_start": age_range[0],
                "age_end": age_range[1],
                "sex": sex,
                "n_case": len(case),
                "n_ctrl": len(ctrl),
            })
            rows.append(stats)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation function
# ═══════════════════════════════════════════════════════════════════════════════

def _domain_softmax_logsumexp(
    h: torch.Tensor, domain_weight: torch.Tensor, chunk_size: int = 2000
) -> torch.Tensor:
    """Per-(subject, position) log-normalizer (logsumexp over the full domain vocab), computed
    once per domain and reused for every token's softmax score in that domain.

    Materializing the full [N_subj, T, vocab_size] logits tensor for a domain like 'diseases'
    (~1250 tokens) at full test-set size would be tens of GB even in half precision -- chunk
    over the subject dimension so peak memory is bounded to one chunk's [chunk, T, vocab_size]
    tensor instead. `domain_weight` is [vocab_size, n_embd] (from
    model.embed._get_domain_weight); softmax(logit_t) = exp(logit_t - this normalizer) is then
    an O(1) extra step per token on top of the existing per-token dot product -- see
    `evaluate_aucs(score_transform="softmax")`, which is why the normalizer is shared across
    tokens rather than recomputed each time.
    """
    n_subj = h.shape[0]
    out = torch.empty(h.shape[0], h.shape[1], dtype=torch.float32, device=h.device)
    w_t = domain_weight.T.half()
    for start in range(0, n_subj, chunk_size):
        end = min(start + chunk_size, n_subj)
        chunk_logits = (h[start:end] @ w_t).float()  # [chunk, T, vocab_size]
        out[start:end] = torch.logsumexp(chunk_logits, dim=-1)
        del chunk_logits
    return out


def _forward_pass_embeddings_and_tokens(model, test_loader, device, block_size, int_to_domain, domain_offsets):
    """Shared step 1 for both evaluate_aucs and evaluate_aucs_dynamic: one forward
    pass over test_loader, collecting per-token output embeddings and a flat
    tokens_df (with age_bin, sex, and cutoff_age columns already attached).
    """
    print("Extracting output embeddings...")
    all_tokens_dfs = []
    all_output_embeddings = []
    batch_offset = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forward pass"):
            batch = batch.to(device)

            _, _, h = model(batch, return_embeddings=True)
            all_output_embeddings.append(h.half())

            tokens_df = batch_to_tokens_df(batch, int_to_domain, domain_offsets)
            tokens_df["subject_idx"] += batch_offset
            batch_offset += batch.batch_size

            all_tokens_dfs.append(tokens_df)

            del h
            gc.collect()

    all_output_embeddings = torch.cat(all_output_embeddings, dim=0)  # [N_subj, T, n_embd]
    tokens_df = pd.concat(all_tokens_dfs, ignore_index=True)
    tokens_df = tokens_df.reset_index().rename(columns={"index": "global_idx"})
    tokens_df["subject_idx"] = tokens_df.global_idx // block_size
    tokens_df = add_sex_column(tokens_df)
    tokens_df = add_age_bin(tokens_df)

    return all_output_embeddings, tokens_df


def evaluate_aucs(
    model,
    test_loader,
    block_size: Optional[int] = None,
    n_jobs: int = -1,
    run_id: Optional[str] = None,
    logger=None,
    output_file: str = "aucs.csv",
    score_transform: str = "logit",
    tokens: Optional[List[Tuple[str, int]]] = None,
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
    score_transform : str
        "logit" (default, original behavior): use the raw per-token logit
        `h @ W[token_id]` as the AUC score.
        "softmax": use softmax(logits)[token_id] -- the logit normalized against every other
        token in that domain's full vocabulary (same normalization set the cross-entropy
        training loss uses) -- as the AUC score instead. This is NOT guaranteed to give the
        same AUC as "logit": softmax(logit_t) = exp(logit_t - logsumexp(all domain logits for
        that subject/position)), and the subtracted normalizer varies by subject/position, so
        ranking by softmax can differ from ranking by the raw logit whenever the normalizer
        correlates with case/control status for token_id (e.g. subjects with an generally
        "busier"/more-confident predicted distribution at that timestep).
    tokens : list of (domain_name, token_id), optional
        Restrict evaluation to these tokens. If None, every token of every predicted
        domain is evaluated.

    Returns
    -------
    auc_df : pd.DataFrame
    """
    if score_transform not in ("logit", "softmax"):
        raise ValueError(f"score_transform must be 'logit' or 'softmax', got {score_transform!r}")
    import logging

    device = model.device
    block_size = block_size or model.block_size

    int_to_domain = model.int_to_domain
    domain_offsets = model.domain_offsets

    # ── 1. Forward pass: collect embeddings and token info ────────────
    all_output_embeddings, tokens_df = _forward_pass_embeddings_and_tokens(
        model, test_loader, device, block_size, int_to_domain, domain_offsets
    )

    # ── 2. Build case/control indices ─────────────────────────────────
    print("Building case/control indices...")

    # Precompute age masks
    age_masks = {}
    for (a0, a1) in AGE_RANGES:
        age_masks[(a0, a1)] = tokens_df.age.between(
            a0 * DAYS_PER_YEAR, a1 * DAYS_PER_YEAR, inclusive="right"
        ).to_numpy()

    # List all predicted (domain, token_id) pairs
    predicted_tokens = [
        (dname, tid)
        for dname in model.predicted_domains
        for tid in range(model.embed._domain_vocab_sizes[dname])
    ]
    if tokens is not None:
        requested = set(tokens)
        unknown = requested - set(predicted_tokens)
        if unknown:
            raise ValueError(f"tokens not among the model's predicted tokens: {sorted(unknown)}")
        predicted_tokens = [t for t in predicted_tokens if t in requested]

    # Group tokens by sex and disease
    by_disease_dfs = {}
    case_subjects = dict(female={}, male={})
    ctrl_subjects = dict(female={}, male={})

    for sex, sex_id in SEX_TOKENS.items():
        tokens_sex = tokens_df.query("sex == @sex_id")
        _by_disease = dict(list(tokens_sex.groupby(["domain", "token_id"])))
        by_disease_dfs[sex_id] = _by_disease

        unique_subjects = tokens_sex.subject_id.unique()

        for (domain_name, token_id) in predicted_tokens:
            if (domain_name, token_id) not in _by_disease:
                continue
            case_subjects[sex][domain_name, token_id] = _by_disease[(domain_name, token_id)].subject_id
            cases_ids = set(case_subjects[sex][(domain_name, token_id)])
            ctrl_subjects[sex][(domain_name, token_id)] = unique_subjects[
                ~pd.Series(unique_subjects).isin(cases_ids)
            ]

    # Precompute arrays for parallel extraction
    all_sids = {
        sex_id: tokens_df.loc[tokens_df.sex == sex_id, "subject_id"].unique()
        for sex_id in SEX_TOKENS.values()
    }

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
    tasks = [
        (sex, sex_id, a0, a1)
        for sex, sex_id in SEX_TOKENS.items()
        for (a0, a1) in AGE_RANGES
    ]

    _cc_args = (
        (sex, sex_id, a0, a1,
         all_sids, tok_subj, tok_sex, tok_gidx,
         age_masks[(a0, a1)],
         ctrl_subjects[sex],
         disease_idx[sex_id])
        for sex, sex_id, a0, a1 in tqdm(tasks, desc="Case/ctrl indices")
    )
    if n_jobs == 1:
        results = [extract_case_ctrl_for_sex_age(*a) for a in _cc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(extract_case_ctrl_for_sex_age)(*a) for a in _cc_args
        )

    ctrl_indices, case_indices = {}, {}
    for local_ctrl, local_case in results:
        ctrl_indices.update(local_ctrl)
        case_indices.update(local_case)
    del results

    # Build lookup
    case_indices_df = pd.Series(case_indices).to_frame()
    case_indices_df = case_indices_df.loc[case_indices_df.apply(lambda x: len(x[0]) > 0, axis=1)]

    ctrl_indices_df = pd.Series(ctrl_indices).to_frame()

    case_ctrl_df = (
        case_indices_df
        .merge(ctrl_indices_df, left_index=True, right_index=True)
        .reset_index()
        .rename({"0_x": "cases", "0_y": "controls"}, axis=1)
    )

    case_ctrl_lookup = defaultdict(dict)
    for row in case_ctrl_df.itertuples(index=False):
        case_ctrl_lookup[(row.level_2, row.level_3)][
            (row.level_0, row.level_1)
        ] = (row.cases, row.controls.astype(int))

    del case_ctrl_df, case_indices, ctrl_indices
    gc.collect()

    # ── 3. Compute AUCs ──────────────────────────────────────────────
    print("Computing AUCs...")

    domain_logsumexp = {}
    if score_transform == "softmax":
        print("Precomputing per-domain softmax normalizers...")
        for dname in tqdm(model.predicted_domains, desc="Softmax normalizers"):
            domain_logsumexp[dname] = _domain_softmax_logsumexp(
                all_output_embeddings, model.embed._get_domain_weight(dname)
            )

    def _score(domain_name, token_id):
        logit = all_output_embeddings @ model.embed._get_domain_weight(domain_name)[token_id].half()
        if score_transform == "softmax":
            return torch.exp(logit.float() - domain_logsumexp[domain_name]).detach().cpu().numpy()
        return logit.detach().cpu().numpy()

    _auc_args = (
        (
            _score(domain_name, token_id),
            (domain_name, token_id),
            case_ctrl_lookup.get((domain_name, token_id), {}),
        )
        for domain_name, token_id in tqdm(predicted_tokens, desc="AUC computation")
    )
    if n_jobs == 1:
        results = [process_disease(*a, block_size=block_size) for a in _auc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_disease)(*a, block_size=block_size) for a in _auc_args
        )

    # ── 4. Assemble results ──────────────────────────────────────────
    auc_data = [row for sublist in results for row in sublist]
    auc_df = pd.DataFrame(auc_data)

    if "auc_bootstrap_mean" in auc_df.columns:
        auc_df = auc_df.drop(["auc_bootstrap_mean", "auc_bootstrap_std"], axis=1)

    if run_id is not None:
        auc_df = auc_df.assign(runid=run_id, block_size=block_size)

    columns_order = [
        c for c in [
            "runid", "domain", "token_id",
            "sex", "age_start", "age_end",
            "n_case", "n_ctrl",
            "auc_delong", "auc_delong_var", "mann_u", "mann_p",
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

    print(f"AUC evaluation complete: {len(auc_df)} rows, "
          f"{auc_df['domain'].nunique()} domains, "
          f"{len(predicted_tokens)} tokens")

    return auc_df


# ═══════════════════════════════════════════════════════════════════════════════
#  Dynamic (time-dependent, incident/dynamic) AUC(t)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dynamic_disease_arrays(tokens_df, sex_id, all_sids_sex, predicted_tokens_set):
    """Per (domain, token_id), returns (diagnosis_age_arr, diagnosis_gidx_arr), both
    aligned to `all_sids_sex` (position i = all_sids_sex[i]), NaN / -1 for subjects
    who never have that disease.

    `diagnosis_age` is the *age_previous* of a subject's first occurrence of the
    token (age of the position right before the diagnosis) -- same convention
    `evaluate_aucs` already uses for bin membership, kept here so a case's
    scoring position is defined identically: one step before the event.
    """
    sid_to_pos = {int(sid): i for i, sid in enumerate(all_sids_sex)}
    N = len(all_sids_sex)

    tokens_sex = tokens_df[tokens_df.sex == sex_id]
    out = {}
    for (domain_name, token_id), df in tokens_sex.groupby(["domain", "token_id"]):
        if (domain_name, token_id) not in predicted_tokens_set:
            continue
        first = df.loc[df.groupby("subject_id")["age_previous"].idxmin()]
        diag_age = np.full(N, np.nan, dtype=np.float64)
        diag_gidx = np.full(N, -1, dtype=np.int64)
        pos = first["subject_id"].map(sid_to_pos)
        valid = pos.notna().to_numpy()
        pos = pos[valid].to_numpy().astype(int)
        diag_age[pos] = first["age_previous"].to_numpy()[valid]
        diag_gidx[pos] = first["global_idx"].to_numpy()[valid]
        out[(domain_name, token_id)] = (diag_age, diag_gidx)
    return out


def extract_dynamic_case_ctrl_for_sex_window(
    sex, sex_id, win0, win1,
    all_sids, tok_subj, tok_sex, tok_gidx, age_mask_window,
    censoring_age_arr_sex, disease_arrays_sex,
    case_mode="incident",
):
    """One sliding window (win0, win1] (days), one sex. For each disease:
      - case    = "incident" (default): diagnosed *within* the window
                  (age_previous in (win0, win1]) -- Heagerty-Zheng I/D.
                  "cumulative": diagnosed *at any time* up to win1
                  (age_previous <= win1), i.e. once a subject becomes a case
                  they remain one for every later window -- Heagerty-Zheng
                  C/D. Either way the case is scored at the position right
                  before their own diagnosis (never re-scored at a later,
                  post-diagnosis position -- that would make the AUC trivial,
                  since the model has already seen the diagnosis token).
      - control = NOT diagnosed by win1 (never, or diagnosis_age > win1) AND
                  still under observation through win1 (censoring_age >= win1),
                  scored at one of their own token positions inside the window.
                  Same definition for both case modes -- Heagerty-Zheng's I/D
                  and C/D estimators share the "dynamic" control side; only
                  the case side differs.
    A subject already diagnosed before win0 (incident mode only), or who left
    observation before win1, is excluded entirely for this window (neither
    case nor control) -- this is what makes it a *dynamic* risk set instead
    of a fixed cohort split.
    """
    mask = age_mask_window & (tok_sex == sex_id)
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
    ctrl_gidx_at_window = subset["global_idx"].to_numpy(dtype=float)  # NaN = no token in window
    has_token_in_window = ~np.isnan(ctrl_gidx_at_window)

    local_case, local_ctrl = {}, {}
    for dd, (diag_age, diag_gidx) in disease_arrays_sex.items():
        if case_mode == "incident":
            case_mask = (diag_age > win0) & (diag_age <= win1)
        elif case_mode == "cumulative":
            case_mask = diag_age <= win1
        else:
            raise ValueError(f"case_mode must be 'incident' or 'cumulative', got {case_mode!r}")
        local_case[(sex, (win0, win1), *dd)] = diag_gidx[case_mask]

        not_yet_a_case = np.isnan(diag_age) | (diag_age > win1)
        still_at_risk = censoring_age_arr_sex >= win1
        eligible = not_yet_a_case & still_at_risk & has_token_in_window
        local_ctrl[(sex, (win0, win1), *dd)] = ctrl_gidx_at_window[eligible].astype(int)

    return local_ctrl, local_case


def process_disease_dynamic(logits_for_token, token_id, lookup_disease, block_size, windows):
    rows = []
    for (win0, win1) in windows:
        for sex in ["female", "male"]:
            case, ctrl = lookup_disease.get((sex, (win0, win1)), EMPTY)
            stats = compute_auc_from_indices(logits_for_token, case, ctrl, block_size)
            stats.update({
                "domain": token_id[0],
                "token_id": token_id[1],
                "age_start": win0,
                "age_end": win1,
                "t": (win0 + win1) / 2,
                "sex": sex,
                "n_case": len(case),
                "n_ctrl": len(ctrl),
            })
            rows.append(stats)
    return rows


def evaluate_aucs_dynamic(
    model,
    test_loader,
    block_size: Optional[int] = None,
    window_width_years: float = 5.0,
    step_years: float = 1.0,
    t_min_years: float = 20.0,
    t_max_years: float = 80.0,
    n_jobs: int = -1,
    run_id: Optional[str] = None,
    logger=None,
    output_file: str = "aucs_dynamic.csv",
    case_mode: str = "incident",
) -> pd.DataFrame:
    """Time-dependent, dynamic AUC(t) (Heagerty & Zheng, 2005), evaluated on a
    continuous, overlapping sliding window instead of evaluate_aucs's fixed
    5-year bins. Supports both estimators from the paper, selected via
    `case_mode`:

      - "incident" (I/D, default): case = diagnosed *within* the window
        (win0, win1] -- "who's about to get disease X". A case contributes to
        exactly one window.
      - "cumulative" (C/D): case = diagnosed *at any time* up to win1 --
        "who will have disease X by age t". Once diagnosed, a subject
        contributes as a case to every later window too, always scored at the
        same position (right before their own diagnosis -- never re-scored at
        a later, post-diagnosis position, which would make the task trivial
        since the model has already seen the diagnosis token by then).

    Both modes share the same *dynamic* control definition (not yet diagnosed
    as of win1, and still under observation through win1) -- per
    Heagerty-Zheng, I/D and C/D differ only in the case side.

    Requires the dataset to have been built with `date_cutoff` + `birth_dates_file`
    (so `DelphiBatch.cutoff_ages` is populated) -- that, combined with each subject's
    death age (if any), gives the true administrative right-censoring age needed for
    a correct risk set. Without it this raises rather than silently falling back to
    the (informatively-censored) "last observed token" proxy.

    Case/control definition per window (win0, win1] centered at t:
      - case    = see `case_mode` above (scored one position before the
                  diagnosis token, same convention as evaluate_aucs)
      - control = *dynamic*: not yet diagnosed as of win1 (may still be
                  diagnosed later -- that's what makes this "dynamic" rather
                  than a fixed never-vs-ever-case split) AND still under
                  observation through win1 (censoring_age >= win1), scored at
                  one of their own token positions inside the window.
    A subject diagnosed before win0 (incident mode only), or censored/dead
    before win1, is excluded from that window entirely.

    Consecutive windows overlap when step_years < window_width_years, so
    resulting points are NOT independent (a case near a window boundary
    contributes to multiple neighboring windows) -- keep that in mind before
    treating adjacent points' AUCs as independent replicates.

    Output schema matches evaluate_aucs's aucs.csv (same age_start/age_end/n_case/
    n_ctrl/auc_delong columns) plus a window-midpoint `t` column, so most existing
    downstream aggregation code can be reused as-is -- just remember windows
    overlap here, unlike evaluate_aucs's disjoint bins.
    """
    if case_mode not in ("incident", "cumulative"):
        raise ValueError(f"case_mode must be 'incident' or 'cumulative', got {case_mode!r}")

    device = model.device
    block_size = block_size or model.block_size
    int_to_domain = model.int_to_domain
    domain_offsets = model.domain_offsets

    all_output_embeddings, tokens_df = _forward_pass_embeddings_and_tokens(
        model, test_loader, device, block_size, int_to_domain, domain_offsets
    )

    if "cutoff_age" not in tokens_df.columns:
        raise ValueError(
            "evaluate_aucs_dynamic requires DelphiBatch.cutoff_ages to be populated "
            "(build the DelphiDataset with date_cutoff + birth_dates_file) -- without "
            "a real administrative censoring age there is no correct way to know who "
            "was still under observation at a given t, and falling back to 'last "
            "observed token' is informatively censored (tied to disease occurrence)."
        )

    predicted_tokens = [
        (dname, tid)
        for dname in model.predicted_domains
        for tid in range(model.embed._domain_vocab_sizes[dname])
    ]
    predicted_tokens_set = set(predicted_tokens)

    # ── Per-subject censoring age = min(administrative cutoff, death age) ──
    cutoff_age_by_subject = tokens_df.groupby("subject_id")["cutoff_age"].first()
    death_age_by_subject = (
        tokens_df[tokens_df.domain == "death"].groupby("subject_id")["age"].min()
    )
    censoring_age_by_subject = pd.concat(
        [cutoff_age_by_subject, death_age_by_subject.reindex(cutoff_age_by_subject.index)],
        axis=1,
    ).min(axis=1, skipna=True)

    n_missing_birth = int(np.isinf(cutoff_age_by_subject).sum())
    if n_missing_birth:
        print(f"WARNING: {n_missing_birth}/{len(cutoff_age_by_subject)} subjects have no "
              f"birth date -- their censoring age defaults to +inf (never administratively "
              f"censored), same convention as DelphiDataset._build_eval_mask.")

    all_sids = {
        sex_id: tokens_df.loc[tokens_df.sex == sex_id, "subject_id"].unique()
        for sex_id in SEX_TOKENS.values()
    }
    censoring_age_arr = {
        sex_id: censoring_age_by_subject.reindex(all_sids[sex_id]).to_numpy()
        for sex_id in SEX_TOKENS.values()
    }

    tok_subj = tokens_df["subject_id"].to_numpy()
    tok_sex = tokens_df["sex"].to_numpy()
    tok_gidx = tokens_df["global_idx"].to_numpy()
    tok_age = tokens_df["age"].to_numpy()

    disease_arrays = {
        sex_id: _build_dynamic_disease_arrays(tokens_df, sex_id, all_sids[sex_id], predicted_tokens_set)
        for sex_id in SEX_TOKENS.values()
    }
    del tokens_df
    gc.collect()

    # ── Build the sliding-window grid (days) and per-window "has a token
    #    here" masks (mirrors evaluate_aucs's age_masks, just on a finer,
    #    overlapping grid instead of disjoint AGE_RANGES) ─────────────────
    half_width = window_width_years * DAYS_PER_YEAR / 2
    t_grid_years = np.arange(t_min_years, t_max_years + 1e-9, step_years)
    windows = [
        (round(t * DAYS_PER_YEAR - half_width), round(t * DAYS_PER_YEAR + half_width))
        for t in t_grid_years
    ]
    age_masks = {
        (w0, w1): (tok_age > w0) & (tok_age <= w1)
        for (w0, w1) in windows
    }

    # ── Parallel case/control extraction, one task per (sex, window) ──────
    print("Extracting dynamic case/control indices...")
    tasks = [
        (sex, sex_id, w0, w1)
        for sex, sex_id in SEX_TOKENS.items()
        for (w0, w1) in windows
    ]
    _cc_args = (
        (sex, sex_id, w0, w1,
         all_sids, tok_subj, tok_sex, tok_gidx,
         age_masks[(w0, w1)],
         censoring_age_arr[sex_id],
         disease_arrays[sex_id],
         case_mode)
        for sex, sex_id, w0, w1 in tqdm(tasks, desc="Dynamic case/ctrl indices")
    )
    if n_jobs == 1:
        results = [extract_dynamic_case_ctrl_for_sex_window(*a) for a in _cc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(extract_dynamic_case_ctrl_for_sex_window)(*a) for a in _cc_args
        )

    ctrl_indices, case_indices = {}, {}
    for local_ctrl, local_case in results:
        ctrl_indices.update(local_ctrl)
        case_indices.update(local_case)
    del results

    case_indices_df = pd.Series(case_indices).to_frame()
    case_indices_df = case_indices_df.loc[case_indices_df.apply(lambda x: len(x[0]) > 0, axis=1)]
    ctrl_indices_df = pd.Series(ctrl_indices).to_frame()

    case_ctrl_df = (
        case_indices_df
        .merge(ctrl_indices_df, left_index=True, right_index=True)
        .reset_index()
        .rename({"0_x": "cases", "0_y": "controls"}, axis=1)
    )

    case_ctrl_lookup = defaultdict(dict)
    for row in case_ctrl_df.itertuples(index=False):
        case_ctrl_lookup[(row.level_2, row.level_3)][
            (row.level_0, row.level_1)
        ] = (row.cases, row.controls.astype(int))

    del case_ctrl_df, case_indices, ctrl_indices
    gc.collect()

    # ── Compute AUCs per (disease, sex, window) ────────────────────────
    print("Computing dynamic AUCs...")

    def _score(domain_name, token_id):
        logit = all_output_embeddings @ model.embed._get_domain_weight(domain_name)[token_id].half()
        return logit.detach().cpu().numpy()

    _auc_args = (
        (
            _score(domain_name, token_id),
            (domain_name, token_id),
            case_ctrl_lookup.get((domain_name, token_id), {}),
        )
        for domain_name, token_id in tqdm(predicted_tokens, desc="Dynamic AUC computation")
    )
    if n_jobs == 1:
        results = [process_disease_dynamic(*a, block_size=block_size, windows=windows) for a in _auc_args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_disease_dynamic)(*a, block_size=block_size, windows=windows) for a in _auc_args
        )

    auc_data = [row for sublist in results for row in sublist]
    auc_df = pd.DataFrame(auc_data)

    if "auc_bootstrap_mean" in auc_df.columns:
        auc_df = auc_df.drop(["auc_bootstrap_mean", "auc_bootstrap_std"], axis=1)
    auc_df = auc_df.assign(case_mode=case_mode)
    if run_id is not None:
        auc_df = auc_df.assign(runid=run_id, block_size=block_size)

    columns_order = [
        c for c in [
            "runid", "domain", "token_id",
            "sex", "age_start", "age_end", "t",
            "n_case", "n_ctrl",
            "auc_delong", "auc_delong_var", "mann_u", "mann_p",
            "case_mode", "block_size",
        ]
        if c in auc_df.columns
    ]
    auc_df = auc_df[columns_order]

    if logger is not None:
        logger.log_df_as_artifact(auc_df, filename=output_file, artifact_path="aucs")
        print(f"Dynamic AUC results logged as artifact: aucs/{output_file}")

    print(f"Dynamic AUC evaluation complete ({case_mode}): {len(auc_df)} rows, "
          f"{auc_df['domain'].nunique()} domains, {len(windows)} windows "
          f"(width={window_width_years}y, step={step_years}y)")

    return auc_df


def compute_time_dependent_cindex(auc_df: pd.DataFrame) -> pd.DataFrame:
    """Antolini et al. (2005) time-dependent concordance index, per (domain, token_id),
    computed from a discretized set of pairwise case/control comparisons.

    Antolini's C^td is a single global summary of discrimination across the whole
    trajectory, defined over ALL comparable pairs at once:

        C^td = P( R_j(T_i) < R_i(T_i) | T_i < T_j, event_i = 1 )

    i.e. for every subject i who gets the disease at time T_i, and every other
    subject j still at risk at T_i (not yet a case, not censored), check whether
    i's own predicted risk (scored right before their diagnosis) exceeds j's
    predicted risk *at time T_i specifically* (not at j's own event/censoring
    time) -- the risk score is time-varying, unlike Harrell's/Uno's C which use
    one fixed baseline score per subject.

    This is computed here as a *discretized* approximation: `auc_df` must be
    `evaluate_aucs_dynamic`'s output run with case_mode="incident" and
    NON-OVERLAPPING windows (window_width_years == step_years) -- each window's
    case/control comparison already IS a set of (i, j) pairs with i's window-of-
    diagnosis standing in for T_i and j scored at a token within that same
    (narrow) window standing in for "j's risk at T_i". Pooling the Mann-Whitney
    numerator (`mann_u` = count of concordant pairs, ties counted as 0.5) and
    denominator (`n_case * n_ctrl`) across ALL windows, rather than averaging
    each window's AUC, gives exactly the pooled U-statistic over the union of
    all windows' pairs -- this converges to the exact (continuous-time)
    Antolini estimator as the window width shrinks to 0; overlapping windows
    would double-count pairs and bias it, hence the hard requirement below.

    Returns one row per (domain, token_id, sex) plus a pooled "F+M" row per
    disease (summed counts across sexes, i.e. a single pooled U-statistic --
    NOT a mean of the two sexes' C-index values).
    """
    # Window boundaries are round()ed to whole days in evaluate_aucs_dynamic, so a
    # "nominal" width of e.g. 365.25 days alternates between 365 and 366 whole-day
    # widths depending on t -- a rounding artifact, not a real gap/overlap. Check
    # adjacency (age_end of one window == age_start of the next, within a couple of
    # days) on the shared window grid instead of demanding bit-exact widths.
    windows = auc_df[["age_start", "age_end"]].drop_duplicates().sort_values("age_start")
    tol_days = 2
    gaps = windows["age_start"].to_numpy()[1:] - windows["age_end"].to_numpy()[:-1]
    if len(gaps) > 0 and np.abs(gaps).max() > tol_days:
        raise ValueError(
            f"windows are not adjacent/non-overlapping within {tol_days} days "
            f"(gaps={gaps[np.abs(gaps) > tol_days]}) -- pass evaluate_aucs_dynamic output "
            "computed with window_width_years == step_years."
        )

    df = auc_df.dropna(subset=["mann_u"]).copy()
    df["n_pairs"] = df["n_case"] * df["n_ctrl"]

    rows = []
    for (domain, token_id), grp in df.groupby(["domain", "token_id"]):
        for sex, sub in grp.groupby("sex"):
            n_pairs = sub["n_pairs"].sum()
            rows.append({
                "domain": domain, "token_id": token_id, "sex": sex,
                "n_events": int(sub["n_case"].sum()),
                "n_pairs": int(n_pairs),
                "c_index_td": (sub["mann_u"].sum() / n_pairs) if n_pairs > 0 else np.nan,
            })
        n_pairs_pooled = grp["n_pairs"].sum()
        rows.append({
            "domain": domain, "token_id": token_id, "sex": "F+M (pooled U-statistic)",
            "n_events": int(grp["n_case"].sum()),
            "n_pairs": int(n_pairs_pooled),
            "c_index_td": (grp["mann_u"].sum() / n_pairs_pooled) if n_pairs_pooled > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def _fast_mann_whitney_u(case_scores: np.ndarray, ctrl_scores: np.ndarray) -> float:
    """Mann-Whitney U (case stochastically > control), via ranks. Exactly matches
    `scipy.stats.mannwhitneyu(case, ctrl, alternative="greater").statistic` (validated
    against it, incl. with duplicated/repeated points -- see module tests) but ~10-50x
    faster with no per-call scipy overhead, needed since the bootstrap below calls this
    thousands of times.
    """
    from scipy.stats import rankdata
    n1 = len(case_scores)
    ranks = rankdata(np.concatenate([case_scores, ctrl_scores]), method="average")
    return ranks[:n1].sum() - n1 * (n1 + 1) / 2


def _prepare_dynamic_risk_set(
    model, test_loader, block_size, window_width_years, step_years, t_min_years, t_max_years,
):
    """Shared, disease-independent setup for the bootstrap CI functions below: one
    forward pass, censoring ages, and the window grid. Kept separate so
    `compute_dynamic_cindex_bootstrap_ci_batch` can amortize this (the expensive part,
    one GPU pass over the whole split) across many diseases instead of repeating it.
    """
    device = model.device
    block_size = block_size or model.block_size
    int_to_domain = model.int_to_domain
    domain_offsets = model.domain_offsets

    all_output_embeddings, tokens_df = _forward_pass_embeddings_and_tokens(
        model, test_loader, device, block_size, int_to_domain, domain_offsets
    )
    if "cutoff_age" not in tokens_df.columns:
        raise ValueError(
            "compute_dynamic_cindex_bootstrap_ci requires DelphiBatch.cutoff_ages to be "
            "populated (build the DelphiDataset with date_cutoff + birth_dates_file)."
        )

    cutoff_age_by_subject = tokens_df.groupby("subject_id")["cutoff_age"].first()
    death_age_by_subject = (
        tokens_df[tokens_df.domain == "death"].groupby("subject_id")["age"].min()
    )
    censoring_age_by_subject = pd.concat(
        [cutoff_age_by_subject, death_age_by_subject.reindex(cutoff_age_by_subject.index)],
        axis=1,
    ).min(axis=1, skipna=True)

    all_sids = {
        sex_id: tokens_df.loc[tokens_df.sex == sex_id, "subject_id"].unique()
        for sex_id in SEX_TOKENS.values()
    }
    censoring_age_arr = {
        sex_id: censoring_age_by_subject.reindex(all_sids[sex_id]).to_numpy()
        for sex_id in SEX_TOKENS.values()
    }

    tok_subj = tokens_df["subject_id"].to_numpy()
    tok_sex = tokens_df["sex"].to_numpy()
    tok_gidx = tokens_df["global_idx"].to_numpy()
    tok_age = tokens_df["age"].to_numpy()

    half_width = window_width_years * DAYS_PER_YEAR / 2
    t_grid_years = np.arange(t_min_years, t_max_years + 1e-9, step_years)
    windows = [
        (round(t * DAYS_PER_YEAR - half_width), round(t * DAYS_PER_YEAR + half_width))
        for t in t_grid_years
    ]
    age_masks = {(w0, w1): (tok_age > w0) & (tok_age <= w1) for (w0, w1) in windows}
    N = all_output_embeddings.shape[0]

    return dict(
        all_output_embeddings=all_output_embeddings, tokens_df=tokens_df,
        all_sids=all_sids, censoring_age_arr=censoring_age_arr,
        tok_subj=tok_subj, tok_sex=tok_sex, tok_gidx=tok_gidx,
        windows=windows, age_masks=age_masks, N=N, block_size=block_size,
    )


def _prepare_disease_windows(model, domain_name, token_id, disease_arrays, setup):
    """Per-disease step: raw case/control global indices -> subject rows + scores per
    window, kept un-pooled (unlike evaluate_aucs_dynamic) so bootstrap weights can be
    applied. `disease_arrays` must already be restricted to (domain_name, token_id)
    (per sex) -- see `_build_dynamic_disease_arrays`.
    """
    block_size = setup["block_size"]
    logit = model.embed._get_domain_weight(domain_name)[token_id].half()
    logit = (setup["all_output_embeddings"] @ logit).detach().cpu().numpy()  # [N, T]

    prepared = []
    for sex, sex_id in SEX_TOKENS.items():
        for (w0, w1) in setup["windows"]:
            local_ctrl, local_case = extract_dynamic_case_ctrl_for_sex_window(
                sex, sex_id, w0, w1,
                setup["all_sids"], setup["tok_subj"], setup["tok_sex"], setup["tok_gidx"],
                setup["age_masks"][(w0, w1)], setup["censoring_age_arr"][sex_id],
                disease_arrays[sex_id],
                case_mode="incident",
            )
            key = (sex, (w0, w1), domain_name, token_id)
            cases = local_case.get(key, EMPTY[0])
            ctrls = local_ctrl.get(key, EMPTY[1])
            if len(cases) == 0 or len(ctrls) == 0:
                continue
            prepared.append((
                logit[cases // block_size, cases % block_size - 1], cases // block_size,
                logit[ctrls // block_size, ctrls % block_size], ctrls // block_size,
            ))
    return prepared


def _pooled_cindex_from_prepared(prepared, weights: Optional[np.ndarray]) -> float:
    num, den = 0.0, 0.0
    for case_scores, case_rows, ctrl_scores, ctrl_rows in prepared:
        wc = np.ones(len(case_rows), dtype=int) if weights is None else weights[case_rows]
        wk = np.ones(len(ctrl_rows), dtype=int) if weights is None else weights[ctrl_rows]
        cs, ks = np.repeat(case_scores, wc), np.repeat(ctrl_scores, wk)
        if len(cs) == 0 or len(ks) == 0:
            continue
        num += _fast_mann_whitney_u(cs, ks)
        den += len(cs) * len(ks)
    return num / den if den > 0 else np.nan


def compute_dynamic_cindex_bootstrap_ci(
    model,
    test_loader,
    domain_name: str,
    token_id: int,
    block_size: Optional[int] = None,
    window_width_years: float = 1.0,
    step_years: float = 1.0,
    t_min_years: float = 20.0,
    t_max_years: float = 80.0,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Subject-level (cluster) bootstrap confidence interval for ONE disease's pooled
    time-dependent C-index (`compute_time_dependent_cindex`'s pooled U-statistic).

    Scoped to a single disease token, not the whole vocabulary: the pooled C-index
    already has within-subject correlation across windows (a never-diagnosed subject
    is a control in many windows before their own diagnosis/censoring), so a valid CI
    needs a bootstrap that resamples *subjects*, not pairs or windows independently.
    Recomputing this for ~700 disease tokens x n_bootstrap replicates would be far more
    compute than is warranted before we even know which diseases' effects are worth a
    CI for -- run this on demand for diseases of interest (e.g. after
    `compute_time_dependent_cindex` flags a large delta), or use
    `compute_dynamic_cindex_bootstrap_ci_batch` for a handful of diseases at once (one
    shared forward pass instead of one per disease).

    Mechanics: draw subject-row multiplicities via a standard nonparametric bootstrap
    (N draws with replacement from N subjects), then recompute the SAME pooled
    U-statistic (sum of Mann-Whitney U across non-overlapping windows / sum of
    n_case*n_ctrl) but weighting each case/control's contribution by its subject's
    bootstrap multiplicity -- implemented via literal repetition (`np.repeat`) into
    `_fast_mann_whitney_u`, which was validated to reproduce scipy's Mann-Whitney U
    exactly, including under repeated/duplicated points (so weighting-via-repetition is
    just as exact, not an approximation). A subject drawn twice contributes to EVERY
    window they'd normally appear in, doubled, together -- this is what correctly
    propagates their within-subject correlation into the resampled statistic, unlike a
    naive per-pair or per-window bootstrap.

    Returns {"c_index_td": point estimate, "ci_low", "ci_high", "boots": array of
    n_bootstrap replicates} using case_mode="incident" only (mirrors
    compute_time_dependent_cindex's requirement of non-overlapping windows).
    """
    setup = _prepare_dynamic_risk_set(
        model, test_loader, block_size, window_width_years, step_years, t_min_years, t_max_years
    )
    disease_arrays = {
        sex_id: _build_dynamic_disease_arrays(setup["tokens_df"], sex_id, setup["all_sids"][sex_id], {(domain_name, token_id)})
        for sex_id in SEX_TOKENS.values()
    }
    del setup["tokens_df"]
    gc.collect()

    prepared = _prepare_disease_windows(model, domain_name, token_id, disease_arrays, setup)
    N = setup["N"]

    point = _pooled_cindex_from_prepared(prepared, None)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for b in tqdm(range(n_bootstrap), desc=f"Bootstrap CI ({domain_name}, {token_id})"):
        weights = np.bincount(rng.integers(0, N, size=N), minlength=N)
        boots[b] = _pooled_cindex_from_prepared(prepared, weights)

    alpha = 1 - ci
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"c_index_td": point, "ci_low": lo, "ci_high": hi, "ci": ci, "boots": boots}


def compute_dynamic_cindex_bootstrap_ci_batch(
    model,
    test_loader,
    disease_tokens: List[Tuple[str, int]],
    block_size: Optional[int] = None,
    window_width_years: float = 1.0,
    step_years: float = 1.0,
    t_min_years: float = 20.0,
    t_max_years: float = 80.0,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Same as `compute_dynamic_cindex_bootstrap_ci`, but for several diseases at once,
    sharing ONE forward pass (the expensive part) across all of them instead of
    repeating it per disease. Use this instead of calling the single-disease version
    in a loop whenever checking more than one or two diseases.

    Returns one row per (domain, token_id) with columns: c_index_td, ci_low, ci_high,
    ci, n_bootstrap.
    """
    setup = _prepare_dynamic_risk_set(
        model, test_loader, block_size, window_width_years, step_years, t_min_years, t_max_years
    )
    tokens_set = set(disease_tokens)
    disease_arrays_all = {
        sex_id: _build_dynamic_disease_arrays(setup["tokens_df"], sex_id, setup["all_sids"][sex_id], tokens_set)
        for sex_id in SEX_TOKENS.values()
    }
    del setup["tokens_df"]
    gc.collect()

    N = setup["N"]
    rows = []
    for domain_name, token_id in tqdm(disease_tokens, desc="Bootstrap CI (batch)"):
        # A disease absent from one sex entirely (e.g. a sex-specific condition) simply
        # never appears as a key in _build_dynamic_disease_arrays's output for that sex
        # (its groupby only yields observed combos) -- default to "nobody in this sex
        # ever gets it" (all-NaN/-1) rather than KeyError.
        disease_arrays = {
            sex_id: {
                (domain_name, token_id): disease_arrays_all[sex_id].get(
                    (domain_name, token_id),
                    (np.full(len(setup["all_sids"][sex_id]), np.nan),
                     np.full(len(setup["all_sids"][sex_id]), -1, dtype=np.int64)),
                )
            }
            for sex_id in SEX_TOKENS.values()
        }
        prepared = _prepare_disease_windows(model, domain_name, token_id, disease_arrays, setup)
        point = _pooled_cindex_from_prepared(prepared, None)

        rng = np.random.default_rng(seed)
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            weights = np.bincount(rng.integers(0, N, size=N), minlength=N)
            boots[b] = _pooled_cindex_from_prepared(prepared, weights)

        alpha = 1 - ci
        lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({
            "domain": domain_name, "token_id": token_id,
            "c_index_td": point, "ci_low": lo, "ci_high": hi,
            "ci": ci, "n_bootstrap": n_bootstrap,
        })

    return pd.DataFrame(rows)


def compute_disease_cox_snell_residuals(
    model, test_loader, domain_name: str, token_id: int, block_size: Optional[int] = None,
) -> pd.DataFrame:
    """Cause-specific Cox-Snell residuals for one disease token, testing whether
    exp(logit_k) is itself a calibrated instantaneous hazard for that specific
    disease -- NOT the same question as "is the competing-risks race well
    calibrated overall" (a plain single-step residual using the total rate
    logsumexp(logits) would test that instead; see this module's
    `time_to_event_loss`-matching residual, and the 2026-09-23 discussion of
    why a single-step per-disease residual is mis-specified: the winning time
    in a race of independent exponentials is Exponential(total rate), not
    Exponential(the winning type's own rate), regardless of which type wins).

    For each subject: walk their sequence in age order, accumulating
    exp(logit_k at position j) * (age[j+1] - age[j]) at every position up to
    (not including) the position where token_id is first observed. That
    accumulated value H is the cause-specific cumulative hazard "used up"
    before the event -- the standard competing-risks generalized (Cox-Snell)
    residual. If exp(logit_k) is a correctly calibrated hazard, H ~
    Exponential(1) for subjects who get the disease, and censored draws from
    that same Exponential(1) for subjects who don't (right-censored at the
    end of their observed record).

    Returns one row per subject: [subject_id, sex, H, event] (event=1 if the
    disease occurred in this record, 0 if censored).

    Caveat: "end of the observed record" is capped by whatever no_event
    tokens the dataset generated, which are only inserted up to a subject's
    own max(disease, death) age (DelphiDataset's age_domains) -- a subject who
    never has ANY disease can have an artificially short observed record,
    understating their true follow-up (the same informative-censoring gap
    flagged for evaluate_aucs_dynamic). This biases censored H values
    downward (too small -- makes calibration look better than it is for the
    censored group), not the event=1 residuals, which are exact regardless.
    """
    device = model.device
    block_size = block_size or model.block_size
    int_to_domain = model.int_to_domain
    domain_offsets = model.domain_offsets

    all_output_embeddings, tokens_df = _forward_pass_embeddings_and_tokens(
        model, test_loader, device, block_size, int_to_domain, domain_offsets
    )

    N, T = all_output_embeddings.shape[0], all_output_embeddings.shape[1]
    W_k = model.embed._get_domain_weight(domain_name)[token_id].half()
    logit_k = (all_output_embeddings @ W_k).float().detach().cpu().numpy()  # [N, T]

    ages = tokens_df["age"].to_numpy().reshape(N, T)
    domain_ids = tokens_df["domain_id"].to_numpy().reshape(N, T)
    token_ids = tokens_df["token_id"].to_numpy().reshape(N, T)
    subject_ids = tokens_df["subject_id"].to_numpy().reshape(N, T)[:, 0]
    sex_arr = tokens_df["sex"].to_numpy().reshape(N, T)[:, 0] if "sex" in tokens_df.columns else None

    PADDING_AGE_THRESHOLD = -1000.0  # DelphiDataset.PADDING_AGE = -10000.0
    domain_to_int = {v: k for k, v in int_to_domain.items()}
    domain_int = domain_to_int[domain_name]

    valid = ages > PADDING_AGE_THRESHOLD  # [N, T]
    dt = np.diff(ages, axis=1)  # [N, T-1] -- gap held constant at position j's logit
    pair_valid = valid[:, :-1] & valid[:, 1:]
    contribution = np.where(pair_valid, np.exp(logit_k[:, :-1]) * np.clip(dt, 0, None), 0.0)
    cum = np.cumsum(contribution, axis=1)  # cum[:, j] = hazard accumulated up to age[:, j+1]

    is_event_token = valid & (domain_ids == domain_int) & (token_ids == token_id)
    has_event = is_event_token.any(axis=1)
    event_idx = np.where(has_event, is_event_token.argmax(axis=1), T - 1)

    stop_idx = np.clip(event_idx - 1, 0, T - 2)
    H = np.where(event_idx - 1 >= 0, cum[np.arange(N), stop_idx], 0.0)

    out = pd.DataFrame({"subject_id": subject_ids, "H": H, "event": has_event.astype(int)})
    if sex_arr is not None:
        out["sex"] = sex_arr
    return out


def kaplan_meier_survival(H: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Product-limit (Kaplan-Meier) survival estimate of H, treating event==0
    rows as right-censored. Returns (times, survival), starting at (0, 1.0).
    Used to check Cox-Snell residuals against the theoretical Exponential(1)
    survival curve S(h) = exp(-h) without needing a survival-analysis library.
    """
    order = np.argsort(H)
    H_sorted = np.asarray(H)[order]
    event_sorted = np.asarray(event)[order]
    n = len(H_sorted)
    at_risk = n
    surv = 1.0
    times, survs = [0.0], [1.0]
    i = 0
    while i < n:
        t = H_sorted[i]
        j = i
        d = 0
        while j < n and H_sorted[j] == t:
            d += int(event_sorted[j] == 1)
            j += 1
        if d > 0:
            surv *= (1 - d / at_risk)
        at_risk -= (j - i)
        times.append(t)
        survs.append(surv)
        i = j
    return np.array(times), np.array(survs)


def compute_freqbin_consensus(
    auc_df: pd.DataFrame,
    domain: str = "diseases",
    n_bins: int = 20,
    min_n_case: int = 20,
) -> pd.DataFrame:
    """Consensus AUC by prevalence bin, for a single run's auc_df (one fold, one split).

    Same convention as auc/analysis/simple_mean_auc_panels.py's build_freqbin_panel,
    but for a single run: diseases are qcut into `n_bins` bins by their total n_case
    (rarest -> bin 1), and each bin's consensus AUC is a plain (unweighted) mean over
    the diseases in it — no fold-averaging step here since there's only one fold.

    Returns a DataFrame with columns: freq_bin, n_diseases, n_case_min, n_case_max,
    auc_simple_mean, auc_std_across_diseases.
    """
    d = (
        auc_df[auc_df["domain"] == domain]
        .dropna(subset=["auc_delong"])
        .query("n_case >= @min_n_case")
        .copy()
    )
    if d.empty:
        return pd.DataFrame(columns=[
            "freq_bin", "n_diseases", "n_case_min", "n_case_max",
            "auc_simple_mean", "auc_std_across_diseases",
        ])

    # Step 1: mean over age bins within (token_id, sex) — same as the "mean over
    # bins within a fold" step used elsewhere, just without a fold axis here.
    per_disease_sex = d.groupby(["token_id", "sex"])["auc_delong"].mean().reset_index()

    total_n_case = d.groupby("token_id")["n_case"].sum().reset_index().rename(
        columns={"n_case": "total_n_case"}
    )
    merged = per_disease_sex.merge(total_n_case, on="token_id", how="inner")
    merged["freq_bin"] = pd.qcut(
        merged["total_n_case"], q=n_bins, labels=False, duplicates="drop"
    ) + 1  # 1 = rarest, n_bins = most common

    return (
        merged.groupby("freq_bin")
        .agg(
            n_diseases=("token_id", "nunique"),
            n_case_min=("total_n_case", "min"),
            n_case_max=("total_n_case", "max"),
            auc_simple_mean=("auc_delong", "mean"),
            auc_std_across_diseases=("auc_delong", "std"),
        )
        .reset_index()
        .sort_values("freq_bin")
    )
