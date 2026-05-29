"""
Refactored Delphi dataset & collate.

Key changes vs. the original dataset.py
----------------------------------------
1.  All cached data lives on **CPU**.  The DataLoader with num_workers > 0
    does the heavy lifting; pin_memory=True handles the transfer.
2.  The Dataset stores three pre-built tensors of shape
        [N_subjects, block_size]
    for domain_ids, local_token_ids and ages, plus metadata vectors
    real_counts and max_ages (both [N_subjects]).
3.  __getitem__ returns *copies* of a single subject's row (safe for workers).
4.  A standalone CollateFn class (no model dependency) performs:
        collate → insert no-event tokens → sort → compute global IDs
    and returns tensors ready for the model.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset

DAYS_PER_YEAR = 365.25
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DelphiBatch
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DelphiBatch:
    """
    Everything the model needs for a single forward pass.
    Produced by DelphiCollateFn, consumed by Delphi.forward.
    """

    global_token_ids: torch.Tensor  # [B, T]  int
    domain_ids: torch.Tensor  # [B, T]  int
    ages: torch.Tensor  # [B, T]  float
    subject_ids: torch.Tensor  # [B]     int
    continuous_data: dict[str, torch.Tensor]  # {name: [B, dim]}
    continuous_positions: dict[str, torch.Tensor]  # {name: [B, n_latent]}

    def to(self, device: str | torch.device) -> DelphiBatch:
        """Move all tensors to device."""
        return DelphiBatch(
            global_token_ids=self.global_token_ids.to(device),
            domain_ids=self.domain_ids.to(device),
            ages=self.ages.to(device),
            subject_ids=self.subject_ids.to(device),
            continuous_data={k: v.to(device) for k, v in self.continuous_data.items()},
            continuous_positions={k: v.to(device) for k, v in self.continuous_positions.items()},
        )

    @property
    def batch_size(self) -> int:
        return self.global_token_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.global_token_ids.shape[1]

    def __repr__(self) -> str:
        return f"DelphiBatch(B={self.batch_size}, T={self.seq_len}, continuous={list(self.continuous_data.keys())})"


# ═══════════════════════════════════════════════════════════════════════════════
#  TokenDomain  (unchanged from original, kept here for self-containedness)
# ═══════════════════════════════════════════════════════════════════════════════


class TokenDomain:
    """
    Single token domain: loads tokenizer.yaml + tokens.csv.

    If `subdomain` is specified, loads token_metadata.csv from the same
    directory to filter tokens by a metadata column (e.g. locus) and
    remaps token IDs to a contiguous sub-vocabulary.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        predict: bool,
        age_jitter: bool,
        type: str,
        subjects: set | None = None,
        at_birth: bool = False,
        aggregation_strategy: Callable | None = None,
        subdomain: str | None = None,
        subdomain_column: str = "locus",
        metadata_file: str = "token_metadata.csv",
    ):
        self.name = name
        self.path = Path(path)
        self.predict = predict
        self.age_jitter = age_jitter
        self.at_birth = at_birth
        self.type = type
        self.subdomain = subdomain
        self._old_to_new: dict[int, int] | None

        assert type in ("categorical", "continuous"), f"Domain type must be 'categorical' or 'continuous', got {type}"

        # Load full tokenizer first
        full_tokenizer = self._load_tokenizer(self.path / "tokenizer.yaml")

        # If subdomain, filter tokenizer and build remap
        if subdomain is not None:
            metadata_path = self.path / metadata_file
            assert metadata_path.exists(), (
                f"subdomain='{subdomain}' requires {metadata_path} to exist. Generate it with generate_hla_metadata.py."
            )
            meta_df = pd.read_csv(metadata_path)
            assert subdomain_column in meta_df.columns, (
                f"Column '{subdomain_column}' not found in {metadata_path}. Available: {meta_df.columns.tolist()}"
            )

            # Filter metadata to this subdomain
            sub_meta = meta_df[meta_df[subdomain_column] == subdomain]
            assert len(sub_meta) > 0, (
                f"No tokens found for subdomain='{subdomain}' in column '{subdomain_column}'. "
                f"Available values: {meta_df[subdomain_column].unique().tolist()}"
            )

            # old_id -> new_id (contiguous from 0)
            old_ids = sorted(sub_meta["token_id"].tolist())
            self._old_to_new = {old: new for new, old in enumerate(old_ids)}
            self._new_to_old = {new: old for old, new in self._old_to_new.items()}

            # Filter tokenizer
            self.tokenizer = {self._old_to_new[old_id]: full_tokenizer[old_id] for old_id in old_ids}

            logger.info(
                "TokenDomain '%s': subdomain '%s' → %d/%d tokens",
                name,
                subdomain,
                len(self.tokenizer),
                len(full_tokenizer),
            )
        else:
            self._old_to_new = None
            self.tokenizer = full_tokenizer

        # Load tokens (with subdomain filtering and remapping)
        self.tokens = self._load_tokens(self.path / "tokens.csv", subjects)

        if aggregation_strategy is not None:
            raise NotImplementedError

    # ---- properties ----------------------------------------------------------

    @property
    def vocab_len(self) -> int:
        if not hasattr(self, "_vocab_len"):
            self._vocab_len = len(self.tokenizer)
        return self._vocab_len

    @property
    def subject_ids(self):
        return self.tokens[:, 0].numpy()

    # ---- loaders -------------------------------------------------------------

    def _load_tokenizer(self, path: Path) -> dict[int, str]:
        with path.open() as f:
            tokenizer = yaml.safe_load(f)
        return {idx: token for idx, token in enumerate(tokenizer)}

    def _load_tokens(self, path: Path, subjects: set | None) -> torch.Tensor:
        assert path.exists(), f"Tokens file {path} does not exist"

        df = pd.read_csv(path)

        if "subject_id" not in df.columns:
            raise ValueError(f"tokens file {path} must contain subject_id")
        if self.type == "categorical" and "token_id" not in df.columns:
            raise ValueError(f"tokens file {path} must contain token_id (categorical)")
        if self.type == "continuous":
            for col in ("token_id", "value"):
                if col not in df.columns:
                    raise ValueError(f"tokens file {path} must contain {col} (continuous)")

        # Filter by subdomain if applicable
        if self._old_to_new is not None:
            valid_old_ids = set(self._old_to_new.keys())
            df = df[df["token_id"].isin(valid_old_ids)].copy()
            df["token_id"] = df["token_id"].map(self._old_to_new)

        df = df.sort_values("subject_id")

        if self.at_birth:
            if "age" in df.columns and (df["age"] != 0).any():
                warnings.warn(
                    "at_birth=True but non-zero ages found. All ages set to 0.",
                    UserWarning,
                    stacklevel=2,
                )
            df["age"] = 0

        if subjects is not None:
            df = df.query("subject_id in @subjects")

        self._as_dataframe = df
        # Always store on CPU
        return torch.tensor(df.values)

    # ---- filtering -----------------------------------------------------------

    def filter_subjects(self, subjects: set) -> TokenDomain:
        isin = self._as_dataframe.subject_id.isin(subjects).to_numpy(dtype=bool)
        self.tokens = self.tokens[isin]
        self._as_dataframe = self._as_dataframe[isin]
        return self

    def __len__(self):
        return len(self._as_dataframe)

    def __repr__(self):
        return str(self._as_dataframe)


# ═══════════════════════════════════════════════════════════════════════════════
#  AgeSampler  (standalone, no model dependency)
# ═══════════════════════════════════════════════════════════════════════════════


class AgeSampler:
    """
    Vectorized no-event-token age sampler.
    Modes: "regular", "random", "grid_jitter".
    Always operates on CPU.
    """

    def __init__(
        self,
        insertion_mode: str = "regular",
        token_rate: float = 1.0,
        seed: int | None = None,
        jitter_sigma: float | None = None,
    ):
        self.insertion_mode = insertion_mode
        self.no_event_token_rate = float(token_rate)
        self.seed = seed
        self.jitter_sigma = jitter_sigma

        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(seed)

    @property
    def step(self) -> float:
        if self.no_event_token_rate <= 0:
            return 0.0
        return self.no_event_token_rate * DAYS_PER_YEAR

    def _counts(self, max_ages: torch.Tensor) -> torch.Tensor:
        step = self.step
        if step <= 0:
            return torch.zeros_like(max_ages, dtype=torch.long)
        return torch.floor(max_ages / step).to(torch.long).clamp_min(0)

    def sample(self, max_ages: torch.Tensor):
        """
        Parameters
        ----------
        max_ages : (B,) float tensor (CPU)

        Returns
        -------
        ages_flat : (total,) float tensor | None
        subj_idx  : (total,) long tensor  | None   (indices into max_ages)
        """
        if max_ages is None or max_ages.numel() == 0:
            return None, None

        step = self.step
        if step <= 0:
            return None, None

        counts = self._counts(max_ages)
        total = int(counts.sum().item())
        if total == 0:
            return None, None

        subj_idx = torch.repeat_interleave(torch.arange(max_ages.numel()), counts)
        max_rep = torch.repeat_interleave(max_ages, counts)

        mode = self.insertion_mode

        if mode == "random":
            u = torch.rand(total, generator=self._generator, dtype=max_ages.dtype)
            return u * max_rep, subj_idx

        if mode == "regular":
            offsets = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=counts.dtype), counts[:-1]]),
                dim=0,
            )
            within = torch.arange(total) - torch.repeat_interleave(offsets, counts)
            ages = (within + 1).to(max_ages.dtype) * step
            ages = torch.minimum(ages, max_rep)
            return ages, subj_idx

        if mode == "grid_jitter":
            sigma = self.jitter_sigma if self.jitter_sigma is not None else step / 5.0
            offsets = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=counts.dtype), counts[:-1]]),
                dim=0,
            )
            within = torch.arange(total) - torch.repeat_interleave(offsets, counts)
            base = (within + 1).to(max_ages.dtype) * step
            noise = torch.randn(total, generator=self._generator, dtype=max_ages.dtype) * sigma
            ages = (base + noise).clamp(min=0.0)
            ages = torch.minimum(ages, max_rep)
            return ages, subj_idx

        raise NotImplementedError(f"Unknown insertion_mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DelphiDataset
# ═══════════════════════════════════════════════════════════════════════════════


class DelphiDatasetItem(TypedDict):
    """
    Uncollated cached tensors for a single subject.
    Produced by the Dataset, consumed by DelphiCollateFn.
    """

    domain_ids: torch.Tensor
    local_token_ids: torch.Tensor
    ages: torch.Tensor
    real_count: torch.Tensor
    max_age: torch.Tensor
    subject_id: torch.Tensor
    continuous: dict[str, torch.Tensor]
    eval_mask: NotRequired[torch.Tensor]  # Marks this key as optional


class DelphiDataset(Dataset):
    """
    Caches all data as CPU tensors of shape [N_subjects, block_size].

    Three parallel tensors:
        domain_ids       [N, block_size]  int
        local_token_ids  [N, block_size]  int
        ages             [N, block_size]  float

    Plus metadata:
        real_counts      [N]              int   (number of real tokens per subject)
        max_ages         [N]              float (max age per subject, for no-event sampling)
        subject_ids      [N]              int   (actual subject IDs, for traceability)

    Padding slots are filled with:
        domain_id  = padding_domain_id
        token_id   = PADDING_TOKEN (0)
        age        = PADDING_AGE (-10000)
    """

    PADDING_TOKEN = 0
    PADDING_AGE = -10000.0

    def __init__(
        self,
        root: str,
        domains_cfg: dict,
        domain_to_int: dict[str, int],
        block_size: int,
        subjects: str | list[str] | None = None,
        exclusions: list[str] | None = None,
        n_samples: int | None = None,
        required_domains: list[str] | None = None,
        # no-event token config (for validation at setup time)
        no_event_token_rate: float = 5.0,
        no_event_insertion_mode: str = "random",
        # continuous domain metadata
        continuous_domains: dict[str, int] | None = None,  # {name: n_latent_tokens}
        age_domains: list[str] | None = None,  # domains to consider for max_age
        # date cutoff for longitudinal evaluation
        date_cutoff: str | None = None,  # "YYYY-MM-DD"; None = no cutoff
        birth_dates_file: str | None = None,  # path to year_and_month_of_birth.txt
    ):
        super().__init__()

        exclusions = exclusions or []
        required_domains = required_domains or ["sex", "diseases"]

        self.root = Path(root)
        self.block_size = block_size
        self.domain_configs = domains_cfg
        self.domain_to_int = domain_to_int
        self.int_to_domain = {v: k for k, v in domain_to_int.items()}
        self.padding_domain_id = domain_to_int["padding"]
        self.continuous_domains = continuous_domains or {}
        self.age_domains = age_domains or ["diseases", "death"]

        # ── Load raw domains ──────────────────────────────────────────────
        self.domains = EasyDict()
        for dname, dinfo in domains_cfg.items():
            if dname == "padding":
                continue
            # Use path from config (allows multiple domains to share the same data dir)
            datafile = Path(dinfo.path) if Path(dinfo.path).is_absolute() else self.root / "tokens" / dinfo.path
            self.domains[dname] = TokenDomain(
                dname,
                datafile,
                predict=dinfo.predict,
                age_jitter=dinfo.age_jitter,
                type=dinfo.type,
                at_birth=dinfo.at_birth,
                subdomain=getattr(dinfo, "subdomain", None),
            )

        logger.info(
            "DelphiDataset | root=%s | domains=%s | required=%s",
            self.root,
            list(domains_cfg.keys()),
            required_domains,
        )

        # ── Resolve subject set ───────────────────────────────────────────
        if subjects is None:
            raise ValueError("Must provide subject IDs or paths to subject lists.")

        if isinstance(subjects, (str, Path)):
            subjects = [subjects]

        if all(isinstance(s, (str, Path)) and Path(s).exists() for s in subjects):
            included = pd.concat([pd.read_csv(self.root / f, names=["subject_id"]) for f in subjects])
        else:
            included = pd.DataFrame({"subject_id": subjects})

        excluded = self._get_excluded_subjects(exclusions)
        if exclusions:
            logger.info("Excluding %d subjects from %s", len(excluded), exclusions)

        subject_df = included[~included["subject_id"].astype(str).isin(excluded)]
        subject_df = self._filter_for_required_domains(subject_df, required_domains)
        logger.info("Subjects after required-domain filter: %d", len(subject_df))

        if n_samples is not None:
            logger.info("Subsampling to n_samples=%d", n_samples)
            subject_df = subject_df.sample(n_samples)

        subject_set = set(subject_df.subject_id.tolist())

        # ── Filter domains ────────────────────────────────────────────────
        for dname in self.domains:
            self.domains[dname].filter_subjects(subject_set)
            assert len(self.domains[dname]) > 0, f"Domain '{dname}' has no tokens after filtering."

        # ── Build subject index ───────────────────────────────────────────
        self._subject_indices = self._precompute_subject_indices()

        # ── Collect tokenizers for friendly view ──────────────────────────
        self.tokenizers = {dname: dom.tokenizer for dname, dom in self.domains.items()}

        # ── Build cache ───────────────────────────────────────────────────
        self._build_cache(subject_set, block_size, no_event_token_rate)

        # ── Date cutoff (longitudinal eval mask) ──────────────────────────
        self._eval_mask: torch.Tensor | None = None
        self._cutoff_ages: torch.Tensor | None = None
        self._cutoff_date: pd.Timestamp | None = None
        if date_cutoff is not None and birth_dates_file is not None:
            self._build_eval_mask(date_cutoff, birth_dates_file)

        print(f"DelphiDataset: {len(self)} subjects, block_size={block_size}, domains={list(self.domains.keys())}")

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_excluded_subjects(self, exclusion_files: list[str]) -> set:
        excluded = set()
        for excl in exclusion_files:
            excl_path = self.root / excl
            if excl_path.exists():
                ids = excl_path.read_text().strip().splitlines()
                excluded |= set(map(str, ids))
        return excluded

    def _filter_for_required_domains(self, subjects: pd.DataFrame, required: list[str]) -> pd.DataFrame:
        for dname in required:
            if dname not in self.domains:
                logger.debug("Required domain '%s' not in self.domains", dname)
                return subjects.iloc[0:0]
            domain_sids = self.domains[dname]._as_dataframe["subject_id"].astype(str).unique()
            subjects = subjects[subjects["subject_id"].astype(str).isin(domain_sids)]
        return subjects

    def _precompute_subject_indices(self) -> dict[str, dict[int, tuple[int, int]]]:
        """
        {domain: {subject_id: (start_idx, count)}}
        """
        indices = {}
        for dname, dom in self.domains.items():
            ids, counts = np.unique(dom.subject_ids, return_counts=True)
            cumcounts = np.concatenate([[0], np.cumsum(counts)])
            indices[dname] = {int(sid): (int(cumcounts[i]), int(counts[i])) for i, sid in enumerate(ids)}
        return indices

    def _get_subject_events(self, subject_id: int) -> dict[str, torch.Tensor]:
        """
        Return per-domain tensors for one subject.
        Each tensor has columns matching the raw tokens.csv layout.
        """
        events = {}
        for dname, dom in self.domains.items():
            try:
                start, count = self._subject_indices[dname][subject_id]
                events[dname] = dom.tokens[start : start + count]
            except KeyError:
                if self.domain_configs[dname].type == "continuous":
                    dim = self.domain_configs[dname].input_size
                    events[dname] = torch.empty(0, dim + 2, dtype=torch.float32)
                else:
                    events[dname] = torch.empty(0, 3, dtype=torch.float32)
        return events

    def _build_cache_slow(self, subject_set: set, block_size: int, no_event_token_rate: float):
        """
        Original Python-loop implementation of _build_cache (kept for reference/benchmarking).
        """
        sorted_subjects = sorted(subject_set)
        N = len(sorted_subjects)

        domain_ids = torch.full((N, block_size), self.padding_domain_id, dtype=torch.long)
        local_token_ids = torch.full((N, block_size), self.PADDING_TOKEN, dtype=torch.long)
        ages = torch.full((N, block_size), self.PADDING_AGE, dtype=torch.float32)
        real_counts = torch.zeros(N, dtype=torch.long)
        max_ages_vec = torch.zeros(N, dtype=torch.float32)
        subject_id_vec = torch.tensor(sorted_subjects, dtype=torch.long)

        # For continuous domains: store raw values separately
        continuous_cache = {}
        for cd_name, _n_latent in self.continuous_domains.items():
            continuous_cache[cd_name] = torch.zeros(N, self.domain_configs[cd_name].input_size, dtype=torch.float32)

        step = no_event_token_rate * DAYS_PER_YEAR

        # Pre-compute column indices per domain (avoid repeated lookups)
        domain_col_indices = {}
        for dname, dom in self.domains.items():
            cols = dom._as_dataframe.columns.tolist()
            domain_col_indices[dname] = {
                "age": cols.index("age"),
                "token_id": cols.index("token_id") if "token_id" in cols else None,
                "value": cols.index("value") if "value" in cols else None,
            }

        truncation_count = 0
        min_real_tokens = float("inf")
        max_real_tokens = 0

        for i, sid in enumerate(sorted_subjects):
            events = self._get_subject_events(sid)

            # Collect all tokens for this subject into a flat list
            tokens_list = []  # list of (domain_id, local_token_id, age)

            for dname, ev in events.items():
                if ev.numel() == 0:
                    continue
                d_id = self.domain_to_int[dname]
                dcfg = self.domain_configs[dname]
                cidx = domain_col_indices[dname]

                if dcfg.type == "continuous":
                    # ev has shape [input_size, n_cols] where each row is
                    # one component: (subject_id, token_id, value, age)
                    if ev.shape[0] > 0:
                        continuous_cache[dname][i] = ev[:, cidx["value"]]
                        age_val = float(ev[0, cidx["age"]])
                        n_latent = self.continuous_domains.get(dname, 1)
                        for _ in range(n_latent):
                            tokens_list.append((d_id, 0, age_val))
                else:
                    for row_idx in range(ev.shape[0]):
                        age_val = float(ev[row_idx, cidx["age"]])
                        if age_val < 0:
                            continue  # skip tokens with negative ages
                        tok_val = int(ev[row_idx, cidx["token_id"]])
                        tokens_list.append((d_id, tok_val, age_val))

            n_real = len(tokens_list)

            # Compute max age from age_domains
            max_age = -float("inf")
            for ad in self.age_domains:
                if ad in events and events[ad].numel() > 0:
                    age_col = domain_col_indices[ad]["age"]
                    max_age = max(max_age, float(events[ad][:, age_col].max()))
            if max_age == -float("inf"):
                max_age = 0.0

            # Track truncation stats
            if step > 0:
                max_noevents = int(max_age // step)
                total_needed = n_real + max_noevents
                if total_needed > block_size:
                    truncation_count += 1
                    min_real_tokens = min(min_real_tokens, n_real)
                    max_real_tokens = max(max_real_tokens, n_real)

            # Sort by (age, domain_id)
            tokens_list.sort(key=lambda t: (t[2], t[0]))

            # Fill cache (truncate to block_size if necessary)
            n_fill = min(n_real, block_size)
            for j in range(n_fill):
                domain_ids[i, j] = tokens_list[j][0]
                local_token_ids[i, j] = tokens_list[j][1]
                ages[i, j] = tokens_list[j][2]

            real_counts[i] = n_fill
            max_ages_vec[i] = max_age

        # Summary warning for truncated subjects
        if truncation_count > 0:
            warnings.warn(
                f"{truncation_count}/{N} subjects will have no-event tokens truncated "
                f"(block_size={block_size}, real tokens range: "
                f"{min_real_tokens}–{max_real_tokens}).",
                UserWarning,
                stacklevel=2,
            )

        # Store everything
        self._domain_ids = domain_ids
        self._local_token_ids = local_token_ids
        self._ages = ages
        self._real_counts = real_counts
        self._max_ages = max_ages_vec
        self._subject_ids = subject_id_vec
        self._continuous_cache = continuous_cache

        # Build subject_id -> index mapping
        self._sid_to_idx = {int(sid): i for i, sid in enumerate(sorted_subjects)}

    def _build_cache(
        self,
        subject_set: set,
        block_size: int,
        no_event_token_rate: float,
    ) -> None:
        """
        Vectorised cache build using NumPy/PyTorch ops (replaces the Python loop).

        Strategy
        --------
        1. Concatenate every categorical domain's data into one flat table
           (subject_id, domain_id, token_id, age) — all already in memory as tensors.
        2. Sort the table globally with np.lexsort (subject_idx, age, domain_id).
        3. Compute each token's position-within-subject via a single np.searchsorted.
        4. Scatter into the [N, block_size] result tensors in one assignment.
        5. Compute per-subject max_age with np.maximum.at (one pass per age-domain).

        Continuous domains are still handled per-subject (same as _build_cache_slow)
        because they are typically absent or very small.
        """
        sorted_subjects = sorted(subject_set)
        N = len(sorted_subjects)
        sorted_subjects_np = np.array(sorted_subjects, dtype=np.int64)

        # ── Pre-filled padding tensors ────────────────────────────────────────
        domain_ids_out = torch.full((N, block_size), self.padding_domain_id, dtype=torch.long)
        local_token_ids_out = torch.full((N, block_size), self.PADDING_TOKEN, dtype=torch.long)
        ages_out = torch.full((N, block_size), self.PADDING_AGE, dtype=torch.float32)

        # Continuous cache (filled per-subject below, same as original)
        continuous_cache = {
            cd_name: torch.zeros(N, self.domain_configs[cd_name].input_size, dtype=torch.float32)
            for cd_name in self.continuous_domains
        }

        # ── Column-index lookup (avoid repeated .index() calls) ───────────────
        domain_col_idx: dict = {}
        for dname, dom in self.domains.items():
            cols = dom._as_dataframe.columns.tolist()
            domain_col_idx[dname] = {
                "age": cols.index("age"),
                "token_id": cols.index("token_id") if "token_id" in cols else None,
                "value": cols.index("value") if "value" in cols else None,
            }

        # ── Build global token table for categorical domains ──────────────────
        cat_sids: list = []
        cat_dids: list = []
        cat_tids: list = []
        cat_ages: list = []

        for dname, dom in self.domains.items():
            if dom.tokens.shape[0] == 0:
                continue
            dcfg = self.domain_configs[dname]
            cidx = domain_col_idx[dname]
            d_id = self.domain_to_int[dname]

            if dcfg.type == "continuous":
                continue  # handled separately

            sids_t = dom.tokens[:, 0].long()
            ages_t = dom.tokens[:, cidx["age"]].float()
            tids_t = dom.tokens[:, cidx["token_id"]].long()

            valid = ages_t >= 0
            if valid.sum() == 0:
                continue

            cat_sids.append(sids_t[valid].numpy().astype(np.int64))
            cat_dids.append(np.full(valid.sum().item(), d_id, dtype=np.int64))
            cat_tids.append(tids_t[valid].numpy().astype(np.int64))
            cat_ages.append(ages_t[valid].numpy().astype(np.float32))

        # ── Handle continuous domains (short per-subject loop) ────────────────
        cont_sids: list = []
        cont_dids: list = []
        cont_ages: list = []

        for cd_name in self.continuous_domains:
            if cd_name not in self.domains:
                continue
            dom = self.domains[cd_name]
            cidx = domain_col_idx[cd_name]
            d_id = self.domain_to_int[cd_name]
            n_lat = self.continuous_domains[cd_name]

            for sid, (start, count) in self._subject_indices[cd_name].items():
                if sid not in subject_set:
                    continue
                idx = int(np.searchsorted(sorted_subjects_np, np.int64(sid)))
                ev = dom.tokens[start : start + count]
                continuous_cache[cd_name][idx] = ev[:, cidx["value"]]
                age_val = float(ev[0, cidx["age"]])
                cont_sids.extend([sid] * n_lat)
                cont_dids.extend([d_id] * n_lat)
                cont_ages.extend([age_val] * n_lat)

        # Merge categorical + continuous placeholder tokens
        all_sids_parts = cat_sids.copy()
        all_dids_parts = cat_dids.copy()
        all_tids_parts = cat_tids.copy()
        all_ages_parts = cat_ages.copy()

        if cont_sids:
            all_sids_parts.append(np.array(cont_sids, dtype=np.int64))
            all_dids_parts.append(np.array(cont_dids, dtype=np.int64))
            all_tids_parts.append(np.zeros(len(cont_sids), dtype=np.int64))
            all_ages_parts.append(np.array(cont_ages, dtype=np.float32))

        if not all_sids_parts:
            # No tokens at all — store empty cache
            self._domain_ids = domain_ids_out
            self._local_token_ids = local_token_ids_out
            self._ages = ages_out
            self._real_counts = torch.zeros(N, dtype=torch.long)
            self._max_ages = torch.zeros(N, dtype=torch.float32)
            self._subject_ids = torch.tensor(sorted_subjects, dtype=torch.long)
            self._continuous_cache = continuous_cache
            self._sid_to_idx = {int(s): i for i, s in enumerate(sorted_subjects)}
            return

        all_sids = np.concatenate(all_sids_parts)
        all_dids = np.concatenate(all_dids_parts)
        all_tids = np.concatenate(all_tids_parts)
        all_ages = np.concatenate(all_ages_parts)

        # ── Map subject_id → 0-based subject index ────────────────────────────
        all_sidx = np.searchsorted(sorted_subjects_np, all_sids)

        # ── Sort by (subject_idx, age, domain_id) ─────────────────────────────
        # np.lexsort: last key = primary sort key
        order = np.lexsort((all_dids, all_ages, all_sidx))
        all_sidx = all_sidx[order]
        all_dids = all_dids[order]
        all_tids = all_tids[order]
        all_ages = all_ages[order]

        # ── Compute position of each token within its subject ─────────────────
        subject_starts = np.searchsorted(all_sidx, np.arange(N, dtype=np.int64))
        pos_within = np.arange(len(all_sidx), dtype=np.int64) - subject_starts[all_sidx]

        # ── Scatter into result tensors ───────────────────────────────────────
        mask = pos_within < block_size
        row = all_sidx[mask]
        col = pos_within[mask]

        domain_ids_out[row, col] = torch.from_numpy(all_dids[mask])
        local_token_ids_out[row, col] = torch.from_numpy(all_tids[mask])
        ages_out[row, col] = torch.from_numpy(all_ages[mask])

        # ── real_counts = min(tokens_per_subject, block_size) ─────────────────
        tokens_per_subject = np.bincount(all_sidx, minlength=N).astype(np.int64)
        real_counts = torch.from_numpy(np.minimum(tokens_per_subject, block_size))

        # ── max_age per subject from age_domains ──────────────────────────────
        max_ages_np = np.full(N, -np.inf, dtype=np.float32)
        for ad in self.age_domains:
            if ad not in self.domains:
                continue
            dom = self.domains[ad]
            if dom.tokens.shape[0] == 0:
                continue
            age_col = domain_col_idx[ad]["age"]
            sids_ad = dom.tokens[:, 0].numpy().astype(np.int64)
            ages_ad = dom.tokens[:, age_col].numpy().astype(np.float32)
            sidx_ad = np.searchsorted(sorted_subjects_np, sids_ad)
            in_range = sidx_ad < N
            matches = in_range & (sorted_subjects_np[np.minimum(sidx_ad, N - 1)] == sids_ad)
            np.maximum.at(max_ages_np, sidx_ad[matches], ages_ad[matches])

        max_ages_np = np.where(np.isneginf(max_ages_np), 0.0, max_ages_np)

        # ── Truncation warning ────────────────────────────────────────────────
        step = no_event_token_rate * DAYS_PER_YEAR
        if step > 0:
            max_noevents = (max_ages_np // step).astype(np.int64)
            total_needed = tokens_per_subject + max_noevents
            truncation_count = int((total_needed > block_size).sum())
            if truncation_count > 0:
                trunc_real = tokens_per_subject[total_needed > block_size]
                warnings.warn(
                    f"{truncation_count}/{N} subjects will have no-event tokens truncated "
                    f"(block_size={block_size}, real tokens range: "
                    f"{trunc_real.min()}–{trunc_real.max()}).",
                    UserWarning,
                    stacklevel=2,
                )

        # ── Store ─────────────────────────────────────────────────────────────
        self._domain_ids = domain_ids_out
        self._local_token_ids = local_token_ids_out
        self._ages = ages_out
        self._real_counts = real_counts
        self._max_ages = torch.from_numpy(max_ages_np)
        self._subject_ids = torch.tensor(sorted_subjects, dtype=torch.long)
        self._continuous_cache = continuous_cache
        self._sid_to_idx = {int(s): i for i, s in enumerate(sorted_subjects)}

    def _build_eval_mask(self, date_cutoff: str, birth_dates_file: str) -> None:
        """
        Build a boolean mask [N, block_size] where True marks tokens that fall
        after date_cutoff for each subject.  Used for longitudinal evaluation:
        the model sees the full sequence as context but is only evaluated on
        post-cutoff positions.

        Subjects with no birth-date record are treated as having no post-cutoff
        tokens (mask is all False).
        """
        from datetime import datetime

        cutoff = datetime.strptime(date_cutoff, "%Y-%m-%d")

        birth_df = pd.read_csv(birth_dates_file, sep="\t").set_index("eid")
        sids = self._subject_ids.numpy()
        birth_info = birth_df.reindex(sids)[["year", "month"]]

        # Vectorised computation of per-subject cutoff age in days
        valid = birth_info["year"].notna().to_numpy(dtype=bool)
        cutoff_ages = np.full(len(sids), np.inf, dtype=np.float32)

        if valid.any():
            bi_valid = birth_info[valid]
            birth_dates = pd.to_datetime(
                {
                    "year": bi_valid["year"].astype(int),
                    "month": bi_valid["month"].astype(int),
                    "day": 1,
                }
            )
            cutoff_ages[valid] = (pd.Timestamp(cutoff) - birth_dates).dt.days.values.astype(np.float32)

        self._cutoff_date = pd.Timestamp(cutoff)  # stored for date reconstruction
        self._cutoff_ages = torch.from_numpy(cutoff_ages)  # [N]
        cutoff_ages_t = self._cutoff_ages.unsqueeze(1)  # [N, 1]

        self._eval_mask = (self._ages > cutoff_ages_t) & (self._ages > self.PADDING_AGE)

        n_post = int(self._eval_mask.sum().item())
        n_real = int((self._ages > self.PADDING_AGE).sum().item())
        logger.info(
            "date_cutoff=%s: %d/%d tokens are post-cutoff (longitudinal targets)",
            date_cutoff,
            n_post,
            n_real,
        )

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def subject_list(self) -> list[int]:
        return self._subject_ids.tolist()

    def __len__(self):
        return self._subject_ids.shape[0]

    def __getitem__(self, index: int) -> DelphiDatasetItem:
        """
        Returns copies of the cached tensors for one subject.
        Safe for multi-worker DataLoader.
        """
        item: DelphiDatasetItem = {
            "domain_ids": self._domain_ids[index].clone(),
            "local_token_ids": self._local_token_ids[index].clone(),
            "ages": self._ages[index].clone(),
            "real_count": self._real_counts[index].clone(),
            "max_age": self._max_ages[index].clone(),
            "subject_id": self._subject_ids[index].clone(),
            # continuous data
            "continuous": {dname: self._continuous_cache[dname][index].clone() for dname in self._continuous_cache},
        }
        if self._eval_mask is not None:
            item["eval_mask"] = self._eval_mask[index].clone()
        return item


# ═══════════════════════════════════════════════════════════════════════════════
#  CollateFn
# ═══════════════════════════════════════════════════════════════════════════════


class DelphiCollateFn:
    """
    Standalone collate function (no model dependency).

    Performs:
    1. Stack subjects into [B, block_size] tensors
    2. Sample & insert no-event tokens (overwriting padding slots)
    3. Sort by (age, domain_id) per subject
    4. Compute global token IDs using domain offsets

    Parameters
    ----------
    age_sampler : AgeSampler
    block_size : int
    domain_to_int : dict[str, int]
    domain_offsets : dict[str, int]
        {domain_name: offset} for computing global IDs.
        For projected domains, the offset points to placeholder (zero) embeddings.
    padding_domain_id : int
    no_event_token_id : int
        The local token ID for no-event tokens (typically 1).
    continuous_domains : dict[str, int]
        {domain_name: n_latent_tokens}
    """

    PADDING_AGE = -10000.0
    PADDING_TOKEN = 0

    def __init__(
        self,
        age_sampler: AgeSampler,
        block_size: int | str,
        domain_to_int: dict[str, int],
        domain_offsets: dict[int, int],  # domain_int -> global offset
        padding_domain_id: int,
        no_event_token_id: int = 1,
        continuous_domains: dict[str, int] | None = None,
        domain_dropout: dict[int, tuple] | None = None,
        training: bool = True,
    ):
        self.age_sampler = age_sampler
        self.block_size = block_size
        self.domain_to_int = domain_to_int
        self.domain_offsets = domain_offsets
        self.padding_domain_id = padding_domain_id
        self.no_event_token_id = no_event_token_id
        self.continuous_domains = continuous_domains or {}
        # {domain_int: (mode, rate)}  mode in {"token", "block"}
        self.domain_dropout = domain_dropout or {}
        self.training = training

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, batch: list[DelphiDatasetItem]) -> DelphiBatch:

        B = len(batch)

        # ── 1. Stack ──────────────────────────────────────────────────────
        domain_ids = torch.stack([item["domain_ids"] for item in batch])  # [B, T_cache]
        local_token_ids = torch.stack([item["local_token_ids"] for item in batch])  # [B, T_cache]
        ages = torch.stack([item["ages"] for item in batch])  # [B, T_cache]
        T = domain_ids.shape[1]
        real_counts = torch.stack([item["real_count"] for item in batch])  # [B]
        max_ages = torch.stack([item["max_age"] for item in batch])  # [B]
        subject_ids = torch.stack([item["subject_id"] for item in batch])  # [B]

        # Stack continuous data
        continuous_data = {}
        for dname in self.continuous_domains:
            continuous_data[dname] = torch.stack([item["continuous"][dname] for item in batch])  # [B, dim]

        # ── 2. Insert no-event tokens ─────────────────────────────────────
        noev_ages, noev_subj_idx = self.age_sampler.sample(max_ages)

        if noev_ages is not None and noev_ages.numel() > 0:
            # Count per subject
            noev_counts = torch.zeros(B, dtype=torch.long)
            for b_idx in range(B):
                mask = noev_subj_idx == b_idx
                count = mask.sum().item()
                available = T - int(real_counts[b_idx].item())

                if count > available:
                    # Truncate: keep the ones with lowest ages
                    subj_ages = noev_ages[mask]
                    _, keep_idx = subj_ages.topk(available, largest=False)
                    # Zero out the ones we're dropping
                    all_indices = mask.nonzero(as_tuple=True)[0]
                    drop_indices = all_indices[~torch.isin(torch.arange(all_indices.numel()), keep_idx)]
                    noev_ages[drop_indices] = -1  # sentinel, will be skipped
                    count = available

                noev_counts[b_idx] = count

            # Now write the surviving no-event tokens into padding slots
            for b_idx in range(B):
                mask = (noev_subj_idx == b_idx) & (noev_ages >= 0)
                b_ages = noev_ages[mask]
                n = b_ages.numel()
                if n == 0:
                    continue

                start = int(real_counts[b_idx].item())
                end = start + n
                # These slots are currently padding; overwrite them
                ages[b_idx, start:end] = b_ages
                local_token_ids[b_idx, start:end] = self.no_event_token_id
                domain_ids[b_idx, start:end] = self.padding_domain_id

        # ── 2.5. Domain dropout (training only) ──────────────────────────
        if self.training and self.domain_dropout:
            for d_int, (mode, rate) in self.domain_dropout.items():
                if rate <= 0.0:
                    continue
                d_mask = domain_ids == d_int  # [B, T]
                if mode == "token":
                    drop = torch.bernoulli(torch.full((B, T), rate, dtype=torch.float)) > 0
                    drop_mask = d_mask & drop
                elif mode == "block":
                    block_drop = torch.bernoulli(torch.full((B,), rate, dtype=torch.float)).bool()
                    drop_mask = d_mask & block_drop.unsqueeze(1)
                else:
                    continue
                ages[drop_mask] = self.PADDING_AGE
                domain_ids[drop_mask] = self.padding_domain_id
                local_token_ids[drop_mask] = self.PADDING_TOKEN

        # ── 3. Sort by (age, domain_id) per subject ──────────────────────
        DOMAIN_SCALE = 0.001  # small enough to not affect age ordering
        sort_key = ages + domain_ids.float() * DOMAIN_SCALE
        _, sort_indices = sort_key.sort(dim=1)

        domain_ids = domain_ids.gather(1, sort_indices)
        local_token_ids = local_token_ids.gather(1, sort_indices)
        ages = ages.gather(1, sort_indices)

        # ── 4. Compute global token IDs ───────────────────────────────────
        global_token_ids = local_token_ids.clone()
        for d_int, offset in self.domain_offsets.items():
            mask = domain_ids == d_int
            global_token_ids[mask] += offset

        # ── 5. Compute continuous positions ───────────────────────────────
        continuous_positions = {}
        for cd_name, n_latent in self.continuous_domains.items():
            cd_int = self.domain_to_int[cd_name]
            # For each subject, find the positions of this domain's tokens
            cd_mask = domain_ids == cd_int  # [B, T]
            # Gather positions per subject
            positions = torch.zeros(B, n_latent, dtype=torch.long)
            for b_idx in range(B):
                pos = cd_mask[b_idx].nonzero(as_tuple=True)[0]
                n_found = min(pos.numel(), n_latent)
                positions[b_idx, :n_found] = pos[:n_found]
            continuous_positions[cd_name] = positions

        # ── 6. Trim or keep to target length ─────────────────────────────
        # After sorting, padding (age=PADDING_AGE) collects at the front of every
        # sequence.
        #
        # block_size="auto": trim to the longest real sequence in this batch.
        # block_size=N (int): keep the full [B, N] — padding stays at the front.
        if self.block_size == "auto":
            n_real = (ages > self.PADDING_AGE).sum(dim=1)  # [B]
            trim_start = int((T - n_real.max()).item())
            if trim_start > 0:
                global_token_ids = global_token_ids[:, trim_start:]
                domain_ids = domain_ids[:, trim_start:]
                ages = ages[:, trim_start:]
                for k in continuous_positions:
                    continuous_positions[k] = (continuous_positions[k] - trim_start).clamp(min=0)

        return DelphiBatch(
            global_token_ids=global_token_ids,
            domain_ids=domain_ids,
            ages=ages,
            subject_ids=subject_ids,
            continuous_data=continuous_data,
            continuous_positions=continuous_positions,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Friendly view
# ═══════════════════════════════════════════════════════════════════════════════


def create_friendly_view(
    batch: DelphiBatch,
    *,
    int_to_domain_name: dict[int, str],
    tokenizers: dict[str, dict[int, str]],
    domain_offsets: dict[int, int],
):
    """
    Human-readable DataFrame for inspecting trajectories.
    Works with DelphiBatch — no model or dataset dependency.
    """
    global_token_ids = batch.global_token_ids
    ages = batch.ages
    domain_ids = batch.domain_ids
    subject_ids = batch.subject_ids

    B, L = global_token_ids.shape

    # Invert offsets: build (domain_int, offset) sorted by offset descending
    # to recover local IDs from global IDs
    rows = []
    for b in range(B):
        subject_id = int(subject_ids[b])
        for pos in range(L):
            g_id = int(global_token_ids[b, pos])
            age_days = float(ages[b, pos])
            d_id = int(domain_ids[b, pos])
            dname = int_to_domain_name.get(d_id, f"unknown_{d_id}")

            # Recover local token ID
            offset = domain_offsets.get(d_id, 0)
            local_id = g_id - offset

            tokenizer = tokenizers.get(dname, {})
            rows.append(
                {
                    "subject_id": subject_id,
                    "age": round(age_days / DAYS_PER_YEAR, 2),
                    "domain_id": d_id,
                    "domain_name": dname,
                    "token_id": local_id,
                    "token_name": tokenizer.get(local_id, f"unknown_{local_id}"),
                }
            )

    df = pd.DataFrame(rows).sort_values(["subject_id", "age", "domain_id", "token_id"]).reset_index(drop=True)
    return df


def color_by_domain(row):
    colors = {
        "hla_alleles": "background-color: #A0E5E5",
        "sex": "background-color: #E5F5FF",
        "lifestyle": "background-color: #B0B0E5",
        "diseases": "background-color: #FFF5E5",
        "death": "background-color: #F5E5FF",
        "padding": "background-color: #F0F0F0",
    }
    return [colors.get(row["domain_name"], "")] * len(row)


# ═══════════════════════════════════════════════════════════════════════════════
#  DataLoader convenience
# ═══════════════════════════════════════════════════════════════════════════════


def create_dataloader(
    dataset: DelphiDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    collate_fn: DelphiCollateFn | None = None,
    **kwargs,
) -> DataLoader:
    """
    Convenience wrapper that creates a DataLoader with the Delphi collate.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **kwargs,
    )
