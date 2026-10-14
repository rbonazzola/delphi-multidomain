"""
AUC evaluation for DelphiLateFusion.

Mirrors the logic of auc/aucs.py but accounts for the HLA head contribution:

    logit[s, t, d] = h[s, t] @ W_d  +  hla_contrib[s, d]

where h comes from the frozen trunk and hla_contrib from the trained head.
This differs from the standard evaluate_aucs which only uses h @ W_d.
"""
from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from auc.aucs import (
    batch_to_tokens_df,
    add_sex_column,
    add_age_bin,
    extract_case_ctrl_for_sex_age,
    process_disease,
    SEX_TOKENS,
    AGE_RANGES,
    DAYS_PER_YEAR,
)


def _load_focus_token_ids(focus_list_path: str | Path, disease_metadata_path: str | Path) -> List[int]:
    """Map ICD-10 codes from a focus list file → disease local token IDs."""
    meta = pd.read_csv(disease_metadata_path, sep="\t")
    icd_to_tid = dict(zip(meta["icd_code"], meta["token_id"]))

    token_ids = []
    with open(focus_list_path) as f:
        for line in f:
            code = line.split("#")[0].strip()
            if not code:
                continue
            if code in icd_to_tid:
                token_ids.append(int(icd_to_tid[code]))
            else:
                print(f"[late_fusion_aucs] ICD code {code!r} not found in tokenizer — skipped.")
    return token_ids


def evaluate_aucs_late_fusion(
    model,                       # DelphiLateFusion
    test_loader,                 # LateFusion DataLoader
    focus_token_ids: Optional[List[int]] = None,
    block_size: Optional[int] = None,
    n_jobs: int = -1,
    logger=None,
    output_file: str = "aucs.csv",
) -> pd.DataFrame:
    """
    Compute per-disease AUCs for a DelphiLateFusion model.

    Parameters
    ----------
    model : DelphiLateFusion
    test_loader : DataLoader yielding LateFusionBatch
    focus_token_ids : list of disease local token IDs to evaluate.
        If None, evaluates all predicted disease tokens.
    block_size : override model block_size for index arithmetic.
    n_jobs : joblib parallelism for AUC computation.
    logger : MLFlowLogger; if provided, saves results as artifact.
    output_file : artifact filename.
    """
    device = model.device
    trunk = model.trunk
    block_size = block_size or trunk.config.block_size

    int_to_domain = trunk.int_to_domain
    domain_offsets = trunk.domain_offsets

    diseases_domain = "diseases"
    assert diseases_domain in trunk.predicted_domains, (
        "Trunk must predict 'diseases' for AUC computation."
    )

    # ── 1. Forward pass: collect trunk hidden states + HLA contributions ──────
    print("Extracting trunk hidden states and HLA contributions …")
    all_h: list[torch.Tensor] = []           # list of [B, T, n_embd] (fp16, on device — GPU)
    all_hla_contrib: list[torch.Tensor] = [] # list of [B, n_diseases] (fp16, on CPU)
    all_tokens_dfs: list[pd.DataFrame] = []
    batch_offset = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forward pass"):
            batch = batch.to(device)

            # Trunk hidden states — keep on GPU so the per-disease matmul is fast
            _, _, h = trunk(batch.delphi_batch, return_embeddings=True)
            all_h.append(h.half())

            # HLA head contribution — move to CPU (small: [B, n_disease_tokens])
            hla_full = model.head(batch.hla_features)           # {domain: [B, vocab]}
            hla_disease = hla_full[diseases_domain].half().cpu() # [B, n_disease_tokens]
            all_hla_contrib.append(hla_disease)

            tokens_df = batch_to_tokens_df(batch.delphi_batch, int_to_domain, domain_offsets)
            tokens_df["subject_idx"] += batch_offset
            batch_offset += batch.batch_size
            all_tokens_dfs.append(tokens_df)

            del h, hla_full
            gc.collect()

    all_h = torch.cat(all_h, dim=0)                    # [N, T, n_embd] on GPU
    all_hla_contrib = torch.cat(all_hla_contrib, dim=0) # [N, n_disease_tokens] on CPU
    tokens_df = pd.concat(all_tokens_dfs, ignore_index=True)
    tokens_df = tokens_df.reset_index().rename(columns={"index": "global_idx"})
    tokens_df["subject_idx"] = tokens_df.global_idx // block_size
    tokens_df = add_sex_column(tokens_df)
    tokens_df = add_age_bin(tokens_df)
    del all_tokens_dfs

    # ── 2. Decide which disease tokens to evaluate ────────────────────────────
    n_disease_tokens = trunk.embed._domain_vocab_sizes[diseases_domain]
    if focus_token_ids is not None:
        predicted_tokens = [(diseases_domain, tid) for tid in focus_token_ids
                            if 0 <= tid < n_disease_tokens]
    else:
        predicted_tokens = [(diseases_domain, tid) for tid in range(n_disease_tokens)]

    # ── 3. Build case/control indices (same logic as evaluate_aucs) ───────────
    print("Building case/control indices …")
    age_masks = {
        (a0, a1): tokens_df.age.between(
            a0 * DAYS_PER_YEAR, a1 * DAYS_PER_YEAR, inclusive="right"
        ).to_numpy()
        for (a0, a1) in AGE_RANGES
    }

    case_subjects: Dict = dict(female={}, male={})
    ctrl_subjects: Dict = dict(female={}, male={})
    by_disease_dfs: Dict = {}

    for sex, sex_id in SEX_TOKENS.items():
        tokens_sex = tokens_df.query("sex == @sex_id")
        _by_disease = dict(list(tokens_sex.groupby(["domain", "token_id"])))
        by_disease_dfs[sex_id] = _by_disease
        unique_subjects = tokens_sex.subject_id.unique()

        for (dname, token_id) in predicted_tokens:
            if (dname, token_id) not in _by_disease:
                continue
            cases_ids = set(_by_disease[(dname, token_id)].subject_id)
            case_subjects[sex][(dname, token_id)] = _by_disease[(dname, token_id)].subject_id
            ctrl_subjects[sex][(dname, token_id)] = unique_subjects[
                ~pd.Series(unique_subjects).isin(cases_ids)
            ]

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
        case_ctrl_lookup[(row.level_2, row.level_3)][(row.level_0, row.level_1)] = (
            row.cases, row.controls.astype(int)
        )
    del case_ctrl_df, case_indices, ctrl_indices, tokens_df
    gc.collect()

    # ── 4. Compute AUCs: logit = h @ W_d + hla_contrib[:, d] ────────────────
    print("Computing AUCs (late fusion logits) …")

    # Weight matrix stays on GPU — matmul is [N, T, n_embd] @ [n_embd] on GPU (fast)
    W_diseases = trunk.embed._get_domain_weight(diseases_domain).half()  # [n_disease, n_embd] on GPU

    def _get_logits_for_token(token_id: int) -> np.ndarray:
        # GPU matmul → move result to CPU only after (result is small: [N, T])
        trunk_logit = (all_h @ W_diseases[token_id]).cpu().numpy()   # [N, T]
        # HLA head contribution is already on CPU: [N, 1] broadcast
        hla_add = all_hla_contrib[:, token_id].unsqueeze(1).numpy()  # [N, 1]
        return trunk_logit + hla_add  # [N, T]

    _auc_args = (
        (
            _get_logits_for_token(token_id),
            (diseases_domain, token_id),
            case_ctrl_lookup.get((diseases_domain, token_id), {}),
        )
        for _, token_id in tqdm(predicted_tokens, desc="AUC computation")
    )
    if n_jobs == 1:
        auc_results = [process_disease(*a, block_size=block_size) for a in _auc_args]
    else:
        auc_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_disease)(*a, block_size=block_size) for a in _auc_args
        )

    # ── 5. Assemble and log ───────────────────────────────────────────────────
    auc_data = [row for sublist in auc_results for row in sublist]
    auc_df = pd.DataFrame(auc_data)

    columns_order = [c for c in [
        "domain", "token_id", "sex", "age_start", "age_end",
        "n_case", "n_ctrl", "auc_delong", "auc_delong_var", "mann_u", "mann_p",
    ] if c in auc_df.columns]
    auc_df = auc_df[columns_order]

    if logger is not None:
        logger.log_df_as_artifact(auc_df, filename=output_file, artifact_path="aucs")
        print(f"AUC results logged as artifact: aucs/{output_file}")

    return auc_df
