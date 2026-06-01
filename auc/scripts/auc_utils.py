from typing import Any

import numpy as np
import torch
from scipy.stats import mannwhitneyu


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # +1 → ranks start from 1 in DeLong's formula
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast implementation of DeLong's algorithm for computing
    covariance of AUC.

    predictions_sorted_transposed: 2D numpy array
                                   [n_classifiers, n_examples]
                                   sorted s.t positives come first
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx, sy = np.cov(v01), np.cov(v10)

    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_auc(case: np.ndarray, ctrl: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Compute AUC + variance using DeLong."""
    if len(case) == 0 or len(ctrl) == 0:
        return None, None

    labels = np.array([1] * len(case) + [0] * len(ctrl))
    scores = np.concatenate([case, ctrl])

    order = (-labels).argsort()
    m = labels.sum()

    preds_sorted = scores[np.newaxis, order]
    auc, cov = fastDeLong(preds_sorted, m)
    assert len(auc) == 1

    return auc[0], cov


def compute_all_stats(
    case: Any, ctrl: Any, do_bootstrap: bool = False, n_bootstrap: int = 200
) -> dict[str, float | np.ndarray | None]:
    """Compute AUC and Mann-Whitney stats for case/control logit arrays.

    Args:
        case: array-like of logit scores for positive cases
        ctrl: array-like of logit scores for controls
        do_bootstrap: whether to compute bootstrapped AUC (requires CUDA)
        n_bootstrap: number of bootstrap replicates
    """
    case = np.asarray(case, float)
    ctrl = np.asarray(ctrl, float)

    if len(case) == 0 or len(ctrl) == 0:
        return {
            "auc_delong": None,
            "auc_delong_var": None,
            "mann_u": None,
            "mann_p": None,
            "auc_bootstrap_mean": None,
            "auc_bootstrap_std": None,
        }

    auc_d, auc_var = delong_auc(case, ctrl)

    u, p = mannwhitneyu(case, ctrl, alternative="greater")

    if do_bootstrap and torch.cuda.is_available():
        boots = optimized_bootstrapped_auc_gpu(case, ctrl, n_bootstrap)
        auc_b_mean = float(np.mean(boots))
        auc_b_std = float(np.std(boots))
    else:
        auc_b_mean = None
        auc_b_std = None

    return {
        "auc_delong": auc_d,
        "auc_delong_var": auc_var,
        "mann_u": float(u),
        "mann_p": float(p),
        "auc_bootstrap_mean": auc_b_mean,
        "auc_bootstrap_std": auc_b_std,
    }


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1):
    """
    Computes bootstrapped AUC estimates using PyTorch on CUDA.

    Parameters:
        case: 1D tensor of scores for positive cases
        control: 1D tensor of scores for controls
        n_bootstrap: Number of bootstrap replicates

    Returns:
        Tensor of shape (n_bootstrap,) containing AUC for each bootstrap replicate
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Convert inputs to CUDA tensors
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)

    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)

    total = (n_case := case.size(0)) + (n_control := control.size(0))

    # Generate bootstrap samples
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")

    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]

    combined = torch.cat([boot_case, boot_control], dim=1)

    # Mask to identify case entries
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True

    # Compute ranks and AUC
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()
