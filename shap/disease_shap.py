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
    python shap/disease_shap.py --run_id RUN_ID --subjects ids.txt --tokens_dir /path/to/tokens
    python shap/disease_shap.py --ckpt <run_dir>/artifacts/checkpoints/best_model.pt \
        --subjects ids.txt --tokens_dir /path/to/tokens --output_dir shap_out

--ckpt: load the model from a .pt file, without MLflow, as auc/compute_aucs_unified.py does
(the same code): from ckpt["model_spec"] if present, else from the params/ (and, for
MultiStream, the domain config yaml in artifacts/) of the exported run directory the .pt
is in, e.g. gel/<experiment>/<run_id>/artifacts/checkpoints/best_model.pt. The run id is
taken from the path. Late-fusion checkpoints are not supported (see load_model).

--subjects: file with one subject ID per line (a non-numeric first line is a header and is
ignored; for a CSV, the first column). The listed subjects are used as they are, instead of
the checkpoint's test split.
--tokens_dir: the directory with one folder per domain (diseases/, death/, sex/, ...), each
with tokens.csv and tokenizer.yaml. Default: <repo>/data/transforms/tokens.
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

# The repo that contains this script (shap/disease_shap.py), e.g. /app in the Docker image.
DELPHI_DIR = Path(__file__).resolve().parent.parent
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

_INVOCATION_CWD = Path.cwd()
os.chdir(DELPHI_DIR)

from data.dataset import DelphiCollateFn, DelphiDataset
from auc.compute_aucs_unified import (
    _block_size, build_model, get_run, mlflow_client, run_id_from_path,
)
from utils import reconstruct_model
from utils.run_loader import _model_config
from utils.mlflow_utils import setup_mlflow

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


def read_subject_ids(path) -> list[int]:
    """Subject IDs, one per line (or the first column of a CSV). A non-numeric line (header) is skipped."""
    ids = []
    for line in Path(path).read_text().splitlines():
        field = line.strip().split(",")[0].strip()
        if not field:
            continue
        try:
            ids.append(int(float(field)))
        except ValueError:
            continue  # header
    if not ids:
        raise SystemExit(f"Could not read any subject ID from {path}")
    return list(dict.fromkeys(ids))


def resolve_disease_tokens(values, tokens_dir: Path) -> dict[int, str]:
    """{token_id: label} for a list of ICD-10 codes (e.g. "E10") and/or 'diseases' token IDs.

    Codes are looked up in <tokens_dir>/diseases/tokenizer.yaml (the entry's first 3 characters,
    ignoring case). An integer is taken as a token ID.
    """
    import yaml
    vocab = yaml.safe_load((tokens_dir / "diseases" / "tokenizer.yaml").read_text())
    by_code = {}
    for token_id, entry in enumerate(vocab):
        by_code.setdefault(str(entry)[:3].upper(), token_id)
    resolved = {}
    for value in values:
        value = str(value).strip()
        if value.isdigit():
            token_id = int(value)
            if token_id >= len(vocab):
                raise SystemExit(f"Token ID {token_id} is out of the diseases vocabulary ({len(vocab)} tokens)")
        elif value.upper() in by_code:
            token_id = by_code[value.upper()]
        else:
            raise SystemExit(f"{value!r} is not an ICD-10 code of {tokens_dir / 'diseases' / 'tokenizer.yaml'}")
        resolved[token_id] = str(vocab[token_id])[:3]
        logger.info(f"  {value} -> token {token_id}: {vocab[token_id]}")
    return resolved


def _make_shap_collate(model, domain_cfg) -> DelphiCollateFn:
    continuous_domains = {
        name: cfg.n_latent_tokens or 1
        for name, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }
    return DelphiCollateFn(
        age_sampler=_NullAgeSampler(),
        block_size=model.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        # No no-event tokens are inserted (_NullAgeSampler), so this id is never used.
        no_event_domain_id=model.domain_to_int.get("no_event", model.domain_to_int["padding"]),
        continuous_domains=continuous_domains,
        domain_dropout={},
        training=False,
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

    # Stable sort: tokens with the same age keep their order whatever the coalition is.
    order = kept_ages.argsort(stable=True)
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

    If several tokens share the last age before target_ages[t], the one with the largest
    global token id is used. The collate sorts by (age, domain) with an unstable sort, so the
    order of tied tokens can change between coalitions; with mask_ties each tied token has its
    own hidden state. This rule does not depend on that order, so a token removed from the
    coalition cannot change the context of an earlier target through a reordering.
    """
    _, _, h = model(batch, return_embeddings=True)   # [B, seq_len, n_embd]
    B        = batch.ages.shape[0]
    T        = len(target_ages)
    result   = torch.zeros(B, T)
    ages     = batch.ages  # [B, seq_len]
    token_ids = batch.global_token_ids  # [B, seq_len]

    for j in range(T):
        a_t   = target_ages[j]
        valid = (ages < a_t) & (ages > PADDING_AGE + 1.0)   # [B, seq_len]
        has_valid = valid.any(dim=1)                          # [B]

        last_age = torch.where(valid, ages, -float("inf")).max(dim=1, keepdim=True).values  # [B, 1]
        at_last_age = valid & (ages == last_age)
        last_pos = torch.where(at_last_age, token_ids, -1).argmax(dim=1)   # [B]

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

def load_model(run_id: str, ckpt_path: Optional[Path], tokens_dir: Path):
    """(model, domain_cfg, test_ids). From the .pt file if ckpt_path is given (no MLflow needed),
    else from the MLflow run (utils.reconstruct_model: Delphi or DelphiMultiStream)."""
    if ckpt_path is None:
        model, test_ids, _, _ = reconstruct_model(run_id)
        return model, _model_config(model).domains, test_ids

    client = mlflow_client()
    _, params = get_run(client, run_id, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "trunk_run_id" in params:
        # DelphiLateFusion: logits = trunk logits + an HLA term that is constant per subject.
        raise SystemExit(
            f"{ckpt_path} is a late-fusion model. Its HLA term is the same at every position, so "
            f"disease->disease SHAP values are those of its trunk (run {params['trunk_run_id']}): "
            f"pass the trunk's checkpoint. HLA attribution for late fusion is not implemented.")

    # build_model (as in auc/compute_aucs_unified.py) looks for the tokens in <data_root>/tokens.
    if tokens_dir.name == "tokens":
        data_root = tokens_dir.parent
    else:
        import tempfile
        data_root = Path(tempfile.mkdtemp(prefix="shap_data_root_"))
        (data_root / "tokens").symlink_to(tokens_dir)
    options = argparse.Namespace(
        runid=run_id, block_size=None, dcfg=[], domains=None, domain_config_yaml=None)
    block_size = _block_size(options, params, ckpt.get("model_spec"))
    model, domain_cfg = build_model(options, client, params, ckpt, data_root, block_size, ckpt_path)
    test_ids = (ckpt.get("metadata") or {}).get("test_ids") or []
    logger.info(f"  {type(model).__name__} from {ckpt_path} (block_size={block_size})")
    return model, domain_cfg, test_ids


def process_run(
    run_id: str,
    n_permutations: int,
    step_batch_size: int,
    output_path: Path,
    subject_ids: Optional[list] = None,
    tokens_dir: Optional[Path] = None,
    ckpt_path: Optional[Path] = None,
    partition_idx: int = 0,
    n_partitions: int = 1,
    source_domain: str = "diseases_death",
    target_disease_ids: Optional[set] = None,
    source_disease_ids: Optional[set] = None,
) -> None:
    logger.info(f"Loading model  run_id={run_id[:12]}  partition={partition_idx}/{n_partitions}  source={source_domain}")
    tokens_dir = Path(tokens_dir) if tokens_dir is not None else DELPHI_DIR / "data" / "transforms" / "tokens"
    model, domain_cfg, test_ids = load_model(run_id, ckpt_path, tokens_dir)
    model = model.to(DEVICE).eval()
    if subject_ids is None and not test_ids:
        raise SystemExit("The checkpoint has no test split: pass --subjects.")

    if subject_ids is not None:
        in_test = len(set(subject_ids) & {int(sid) for sid in test_ids})
        logger.info(f"  {len(subject_ids)} subjects from --subjects "
                    f"({in_test} of them in the checkpoint's test split)")
        test_ids = subject_ids

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

    # Token files: <tokens_dir>/<domain folder>/ (the folder name is the one the model was trained with)
    for dname, dcfg in domain_cfg.items():
        if dname in ("padding", "no_event") or not getattr(dcfg, "path", None):
            continue
        dcfg.path = str(tokens_dir / Path(str(dcfg.path)).name)
        if not Path(dcfg.path).is_dir():
            raise FileNotFoundError(f"Domain '{dname}': {dcfg.path} does not exist (check --tokens_dir)")
    logger.info(f"  tokens from {tokens_dir}")

    continuous_domains = {
        name: cfg.n_latent_tokens or 1
        for name, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }
    dataset = DelphiDataset(
        root=tokens_dir.parent,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=model.block_size,
        subjects=test_ids,
        required_domains=["diseases"],
        # DelphiMultiStreamConfig has no no-event settings; no no-event token is inserted anyway.
        no_event_token_rate=getattr(model.config, "no_event_token_rate", 5.0),
        no_event_insertion_mode=getattr(model.config, "no_event_token_insertion_mode", "random"),
        continuous_domains=continuous_domains,
    )
    collate = _make_shap_collate(model, domain_cfg)

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
        if source_disease_ids is not None:
            # Only these 'diseases' tokens are permuted; all other tokens stay fixed (context).
            src_mask &= (domain_ids == disease_int) & torch.isin(
                token_ids, torch.tensor(sorted(source_disease_ids))
            )
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
        # Need at least one source before a target (a causal pair)
        if not (ages[src_positions].min() < ages[tgt_positions].max()):
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
    src.add_argument("--ckpt",          type=str, metavar="PATH",
                     help="Model .pt file (no MLflow needed); see the module docstring.")
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
    parser.add_argument("--subjects",       type=str, default=None,
                        help="File with subject IDs, one per line (header optional). "
                             "Used instead of the checkpoint's test split.")
    parser.add_argument("--tokens_dir",     type=str, default=None,
                        help="Directory with one folder per domain (diseases/, death/, ...). "
                             "Default: <repo>/data/transforms/tokens.")
    parser.add_argument("--output_dir",     type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR}).")
    parser.add_argument("--tsv_output",     type=str, default=None,
                        help="With --generate_submission: path for the TSV file.")
    parser.add_argument("--source_domain",  type=str, default="diseases_death",
                        choices=["diseases_death", "hla_alleles"],
                        help="Domain of tokens to permute as SHAP sources.")
    parser.add_argument("--target_disease_ids", type=str, nargs="+", default=None,
                        metavar="CODE_OR_ID",
                        help="Restrict target tokens to these diseases: ICD-10 codes (E10) "
                             "or 'diseases' token IDs (201); drops 'death' from targets.")
    parser.add_argument("--source_token_ids", type=str, nargs="+", default=None,
                        metavar="CODE_OR_ID",
                        help="Only with --source_domain diseases_death: permute only these "
                             "diseases (ICD-10 codes or token IDs); all other tokens stay "
                             "fixed as context. Fewer sources = much faster.")
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
    ckpt_path = None
    if args.ckpt:
        ckpt_path = (_INVOCATION_CWD / args.ckpt).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"{ckpt_path} does not exist")
        run_ids = [run_id_from_path(ckpt_path) or ckpt_path.stem]
    elif args.run_id:
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

    subject_ids = None
    if args.subjects:
        subject_ids = read_subject_ids(_INVOCATION_CWD / args.subjects)
    tokens_dir = (_INVOCATION_CWD / args.tokens_dir).resolve() if args.tokens_dir else None

    if args.dry_run:
        n_jobs = len(run_ids) * args.n_partitions
        print(f"Would submit {n_jobs} job(s): {len(run_ids)} run(s) × {args.n_partitions} partition(s)")
        for rid in run_ids:
            print(f"  {rid}")
        return

    codes_dir = tokens_dir or DELPHI_DIR / "data" / "transforms" / "tokens"
    targets = resolve_disease_tokens(args.target_disease_ids, codes_dir) if args.target_disease_ids else None
    sources = resolve_disease_tokens(args.source_token_ids, codes_dir) if args.source_token_ids else None
    target_disease_ids = set(targets) if targets else None

    source_disease_ids = set(sources) if sources else None
    if source_disease_ids is not None and args.source_domain != "diseases_death":
        raise SystemExit("--source_token_ids needs --source_domain diseases_death")
    src_tag = f"_src-{args.source_domain}" if args.source_domain != "diseases_death" else ""
    def _codes_tag(prefix, chosen):
        labels = sorted(set(chosen.values()))
        return f"_{prefix}-{'-'.join(labels)}" if len(labels) <= 5 else f"_{prefix}-{len(chosen)}diseases"
    if sources:
        src_tag += _codes_tag("src", sources)
    tgt_tag = _codes_tag("tgt", targets) if targets else ""
    suffix = f"_part{args.partition_idx:04d}of{args.n_partitions}" if args.n_partitions > 1 else ""
    for run_id in run_ids:
        process_run(
            run_id=run_id,
            n_permutations=args.n_permutations,
            step_batch_size=args.batch_size,
            output_path=output_dir / f"{run_id}{src_tag}{tgt_tag}{suffix}.pkl",
            subject_ids=subject_ids,
            tokens_dir=tokens_dir,
            ckpt_path=ckpt_path,
            partition_idx=args.partition_idx,
            n_partitions=args.n_partitions,
            source_domain=args.source_domain,
            target_disease_ids=target_disease_ids,
            source_disease_ids=source_disease_ids,
        )


if __name__ == "__main__":
    main()
