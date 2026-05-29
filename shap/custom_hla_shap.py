"""
SHAP-like HLA allele effect estimation, optionally stratified by sex and age bracket.

Computes the delta-logit: the difference in disease log-odds between a subject's
original HLA genotype and counterfactually injected donor HLA blocks. Subjects who
carry the target allele AND have the disease in their record are used as cases;
non-carriers serve as HLA donors.

Supersedes custom_hla_shap2.py (no sex stratification) and custom_hla_shap3.py
(with sex stratification). Omitting --sex reproduces the custom_hla_shap2.py
behaviour exactly.

Output .pkl keys:
  - "delta":        np.ndarray (N,) — Δlogit (original − counterfactual)
  - "ages":         np.ndarray (N,) — age in days at the disease event
  - "sexes":        np.ndarray (N,) — int (0=female, 1=male, -1=unknown)
  - "age_brackets": np.ndarray (N,) — bracket label string or None

Usage:
    python shap/custom_hla_shap.py \\
        --allele_id <int> \\
        --disease "<ICD description>" \\
        --n_counterfactuals 10 \\
        [--sex female|male] \\
        --subjects data/transforms/subject_lists/genetic_white_ids.txt \\
        --output shap/delta_logits/{disease_id}__{allele_id}__{sex}.pkl

Arguments:
    --disease           Disease name (fuzzy-matched against tokenizer)
    --disease_id        Disease token ID (alternative to --disease)
    --hla_allele        HLA allele prefix (e.g. "HLA-A*02"); all matching alleles grouped
    --allele_id         Single allele token ID (alternative to --hla_allele)
    --n_counterfactuals Number of donor HLA injections to average over (default: 5)
    --sex               Restrict cases and donors to this sex: "male" or "female" (optional)
    --subjects          Path to file with subject IDs to restrict to (optional)
    --output            Output .pkl path; supports {disease_id}, {allele_id}, {sex} placeholders

Case genotype filtering:
    --case_zygosity     {any,het,hom}  Zygosity of the primary allele in cases (default: any).
                        'het' = exactly 1 copy; 'hom' = ≥2 copies of any allele in the group.
    --case_also         ALLELE_PREFIX  Cases must also carry ≥1 allele matching this prefix
                        (AND logic; repeatable for multiple additional requirements).

Donor genotype filtering:
    --donor_excludes    ALLELE_PREFIX  Alleles donors must NOT carry.  Overrides the default
                        behaviour of excluding the case alleles.  Repeatable.
    --donor_requires    ALLELE_PREFIX  Alleles donors MUST carry (AND logic, repeatable).
    --donor_zygosity    {any,het,hom}  Zygosity applied to each --donor_requires group.
"""

import argparse
import os
import pickle as pkl
import random
import sys
import warnings
from difflib import get_close_matches
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from loguru import logger
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

DELPHI_DIR = Path("/nfs/research/birney/users/bonazzola/repos/delphis/delphi-refactor")
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import AgeSampler, DelphiCollateFn, DelphiDataset  # noqa: E402
from utils import reconstruct_model  # noqa: E402
from utils.utils import read_ids  # noqa: E402

# Capture the invocation CWD before chdir so user-provided relative paths still resolve correctly.
_INVOCATION_CWD = Path.cwd()
os.chdir(DELPHI_DIR)

device = "cpu"
DAYS_PER_YEAR = 365.25

SEX_TOKENS = {"female": 0, "male": 1}
INT_TO_SEX = {v: k for k, v in SEX_TOKENS.items()}
AGE_RANGES = [(a, a + 20) for a in range(0, 70, 10)]

BATCH_SIZE = 512
CACHE_DIR = "/hps/nobackup/birney/users/bonazzola/delphi/output/cache"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

with (DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokenizer.yaml").open() as _f:
    hla_tokenizer = yaml.safe_load(_f)
with (DELPHI_DIR / "data/transforms/tokens/diseases/tokenizer.yaml").open() as _f:
    disease_tokenizer = np.array(yaml.safe_load(_f))


def assign_age_bracket(age_days):
    age_years = age_days / DAYS_PER_YEAR
    for lo, hi in AGE_RANGES:
        if lo <= age_years < hi:
            return f"{lo}-{hi}"
    return None


def _make_continuous_domains(model):
    return {dname: cfg.n_latent_tokens or 1 for dname, cfg in model.domain_cfg.items() if cfg.type == "continuous"}


def _make_collate(model):
    continuous_domains = _make_continuous_domains(model)
    age_sampler = AgeSampler(
        insertion_mode=model.config.no_event_token_insertion_mode,
        token_rate=model.config.no_event_token_rate,
        seed=model.config.seed,
    )
    return DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=model.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
    )


def get_dataloader(model, all_test_ids, subject_ids=None, block_size=None):
    if block_size is not None:
        model.set_block_size(block_size)

    if subject_ids is not None:
        if isinstance(subject_ids, str):
            assert Path(subject_ids).exists(), f"File {subject_ids} does not exist."
            subject_ids = read_ids(subject_ids)
        subject_ids = [sid for sid in all_test_ids if sid in set(subject_ids)]
    else:
        subject_ids = all_test_ids

    continuous_domains = _make_continuous_domains(model)
    dataset = DelphiDataset(
        root=DELPHI_DIR / "data" / "transforms",
        domains_cfg=model.domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=model.block_size,
        subjects=subject_ids,
        required_domains=["diseases"],
        no_event_token_rate=model.config.no_event_token_rate,
        no_event_insertion_mode=model.config.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
    )
    collate = _make_collate(model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return dataset, dataloader


def get_tokens_df_from_dataset(model, dataset):
    """Build a flat tokens DataFrame from a DelphiDataset. Vectorized: no Python loop over subjects."""
    N, T = dataset._domain_ids.shape
    subj_expanded = dataset._subject_ids.unsqueeze(1).expand(N, T)
    seq_idx = torch.arange(T).unsqueeze(0).expand(N, T)
    real_mask = seq_idx < dataset._real_counts.unsqueeze(1)

    df = pd.DataFrame(
        {
            "subject_id": subj_expanded[real_mask].numpy(),
            "domain_id": dataset._domain_ids[real_mask].numpy(),
            "token_id": dataset._local_token_ids[real_mask].numpy(),
            "age": dataset._ages[real_mask].numpy(),
        }
    )
    df["domain"] = df["domain_id"].map(model.int_to_domain)
    return df


def get_tokens_df_cached(model, test_ids, run_id, cache_dir=None):
    if cache_dir is not None:
        dataset_hash = hash(tuple(sorted(test_ids)))
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True, parents=True)
        cache_file = cache_path / f"tokens_df_{run_id}_{dataset_hash}.parquet"

        if cache_file.exists():
            print(f"Loading cached tokens_df from {cache_file}")
            return pd.read_parquet(cache_file)

    print("Computing tokens_df...")
    dataset, _ = get_dataloader(model, all_test_ids=test_ids)
    tokens_df = get_tokens_df_from_dataset(model, dataset)

    if cache_dir is not None:
        print(f"Caching tokens_df at {cache_file}")
        tokens_df.to_parquet(cache_file)

    return tokens_df


def find_most_similar_token(query, token_names, n=3, cutoff=0.6):
    match = get_close_matches(query, token_names, n=n, cutoff=cutoff)
    if not match:
        raise ValueError("No similar disease found.")
    best_name = match[0]
    tid = next(i for i, name in enumerate(token_names) if name == best_name)
    return tid, best_name


@torch.no_grad()
def disease_prev_logits_from_embeddings(model, h, batch, disease_domain, disease_token_id):
    """
    Compute per-token logits for disease_token_id and find positions where
    the *next* token is that disease.
    """
    dom_id = model.domain_to_int[disease_domain]
    global_disease_id = disease_token_id + model.domain_offsets[dom_id]

    W = model.embed._get_domain_weight(disease_domain)[disease_token_id]  # [n_embd]
    logits = (h @ W).float()  # [B, T]

    next_is_disease = (batch.global_token_ids[:, 1:] == global_disease_id) & (batch.domain_ids[:, 1:] == dom_id)
    b_idx, t_prev = torch.where(next_is_disease)
    return logits[b_idx, t_prev], torch.stack([b_idx, t_prev], dim=1)


def inject_hla_item(item_rec, item_don, hla_domain_int, padding_domain_id, padding_age=-10000.0):
    """Return a copy of item_rec with its HLA tokens replaced by those of item_don."""
    rec_real = int(item_rec["real_count"])
    don_real = int(item_don["real_count"])

    rec_doms = item_rec["domain_ids"][:rec_real]
    don_doms = item_don["domain_ids"][:don_real]

    rec_hla = rec_doms == hla_domain_int
    don_hla = don_doms == hla_domain_int

    non_hla_dom = rec_doms[~rec_hla]
    non_hla_tok = item_rec["local_token_ids"][:rec_real][~rec_hla]
    non_hla_age = item_rec["ages"][:rec_real][~rec_hla]

    don_hla_dom = don_doms[don_hla]
    don_hla_tok = item_don["local_token_ids"][:don_real][don_hla]
    don_hla_age = item_don["ages"][:don_real][don_hla]

    all_dom = torch.cat([don_hla_dom, non_hla_dom])
    all_tok = torch.cat([don_hla_tok, non_hla_tok])
    all_age = torch.cat([don_hla_age, non_hla_age])

    # Sort by age, breaking ties by domain to keep deterministic ordering
    sort_key = all_age + all_dom.float() * 0.001
    _, idx = sort_key.sort()
    all_dom = all_dom[idx]
    all_tok = all_tok[idx]
    all_age = all_age[idx]

    block_size = item_rec["domain_ids"].shape[0]
    n_real = min(len(all_dom), block_size)

    new_item = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in item_rec.items()}
    new_item["domain_ids"] = torch.full((block_size,), padding_domain_id, dtype=item_rec["domain_ids"].dtype)
    new_item["local_token_ids"] = torch.zeros(block_size, dtype=item_rec["local_token_ids"].dtype)
    new_item["ages"] = torch.full((block_size,), padding_age, dtype=item_rec["ages"].dtype)
    new_item["domain_ids"][:n_real] = all_dom[:n_real]
    new_item["local_token_ids"][:n_real] = all_tok[:n_real]
    new_item["ages"][:n_real] = all_age[:n_real]
    new_item["real_count"] = torch.tensor(n_real, dtype=item_rec["real_count"].dtype)

    return new_item


def extract_sex_map(tokens_df):
    return (
        tokens_df[tokens_df["domain"] == "sex"]
        .drop_duplicates("subject_id")
        .set_index("subject_id")["token_id"]
        .to_dict()
    )


def build_hla_allele_counts(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count HLA allele copies per subject without deduplication.
    Returns a DataFrame with columns: subject_id, token_id, n_copies.
    n_copies is 1 for heterozygous carriers and 2 for homozygous carriers.
    """
    return (
        tokens_df.query('domain == "hla_alleles"')[["subject_id", "token_id"]]
        .groupby(["subject_id", "token_id"])
        .size()
        .reset_index(name="n_copies")
    )


class GenotypeFilter:
    """
    Selects subjects based on HLA allele carrier status and zygosity.

    All selection methods return sets of integer subject_ids. OR semantics
    apply *within* each allele_ids list (any of those alleles qualifies);
    intersect (&) return values for AND semantics across multiple groups.

    Zygosity note: for a prefix group with multiple 4-field allele IDs,
    'hom' means ≥2 copies of *the same* allele_id (not 1 copy each of two
    different alleles under the same prefix).
    """

    def __init__(self, hla_counts: pd.DataFrame):
        self._c = hla_counts  # subject_id, token_id, n_copies
        self._all = set(hla_counts["subject_id"].unique())

    def carriers(self, allele_ids) -> set:
        """Subjects with ≥1 copy of ANY allele in allele_ids."""
        mask = self._c["token_id"].isin(allele_ids)
        return set(self._c.loc[mask, "subject_id"])

    def non_carriers(self, allele_ids) -> set:
        """Subjects with 0 copies of ALL alleles in allele_ids."""
        return self._all - self.carriers(allele_ids)

    def carriers_all(self, allele_id_groups) -> set:
        """
        AND across groups: subjects who carry ≥1 allele from *each* group.
        allele_id_groups: list of lists, e.g. [[A*02_ids], [B*07_ids]].
        """
        result = self._all.copy()
        for group in allele_id_groups:
            result &= self.carriers(group)
        return result

    def het(self, allele_ids) -> set:
        """Subjects with exactly 1 copy of any allele in allele_ids."""
        mask = (self._c["token_id"].isin(allele_ids)) & (self._c["n_copies"] == 1)
        return set(self._c.loc[mask, "subject_id"])

    def hom(self, allele_ids) -> set:
        """Subjects with ≥2 copies of any single allele in allele_ids."""
        mask = (self._c["token_id"].isin(allele_ids)) & (self._c["n_copies"] >= 2)
        return set(self._c.loc[mask, "subject_id"])

    def with_zygosity(self, allele_ids, zyg: str) -> set:
        """Dispatch on zyg ∈ {'any', 'het', 'hom'}."""
        if zyg == "any":
            return self.carriers(allele_ids)
        if zyg == "het":
            return self.het(allele_ids)
        if zyg == "hom":
            return self.hom(allele_ids)
        raise ValueError(f"Unknown zygosity {zyg!r}; expected 'any', 'het', or 'hom'")


def compute_delta_for_run(
    model,
    test_ids,
    tokens_df,
    disease_id,
    allele_ids,
    sex_map,
    sex_filter=None,
    n_counterfactuals=1,
    disease_name="disease",
    allele_name="allele",
    case_zygosity="any",
    case_also_allele_id_groups=None,
    donor_exclude_ids=None,
    donor_require_id_groups=None,
    donor_zygosity="any",
):
    """
    Returns (delta, ages, sexes) as 1-D tensors, one entry per disease event.

    Case selection:
      Subjects carrying allele_ids with case_zygosity AND having the disease.
      case_also_allele_id_groups: list of allele-id lists, each group must also
      be carried (≥1 copy) — AND logic across groups.

    Donor selection:
      By default, non-carriers of allele_ids.
      donor_exclude_ids: overrides the default exclusion set if provided.
      donor_require_id_groups: list of allele-id lists, each group must be
      carried with donor_zygosity — AND logic across groups.
    """
    hla_counts = build_hla_allele_counts(tokens_df)
    gf = GenotypeFilter(hla_counts)

    # --- case selection ---
    subjects_with_allele = gf.with_zygosity(allele_ids, case_zygosity)
    if case_also_allele_id_groups:
        subjects_with_allele &= gf.carriers_all(case_also_allele_id_groups)

    # --- donor selection ---
    _exclude = allele_ids if donor_exclude_ids is None else donor_exclude_ids
    donor_subjects = gf.non_carriers(_exclude) if _exclude else set(gf._all)
    if donor_require_id_groups:
        for group in donor_require_id_groups:
            donor_subjects &= gf.with_zygosity(group, donor_zygosity)

    # --- disease filter ---
    subjects_with_disease = set(
        tokens_df.loc[
            (tokens_df["token_id"] == disease_id) & (tokens_df["domain"] == "diseases"),
            "subject_id",
        ].astype(int)
    )
    subjects_with_allele_disease = subjects_with_allele & subjects_with_disease

    if sex_filter is not None:
        sex_int = SEX_TOKENS[sex_filter]
        subjects_with_allele_disease = {s for s in subjects_with_allele_disease if sex_map.get(s) == sex_int}
        donor_subjects = {s for s in donor_subjects if sex_map.get(s) == sex_int}

    if not subjects_with_allele_disease:
        logger.warning("No cases found after filtering.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)
    if not donor_subjects:
        logger.warning("No donors found after filtering.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)

    filtered_dataset, filtered_dataloader = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=subjects_with_allele_disease,
        block_size=96,
    )

    n_donors = min(len(subjects_with_allele_disease) * 2, len(donor_subjects))
    donor_dataset, _ = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=random.sample(list(donor_subjects), n_donors),
        block_size=96,
    )

    hla_domain_int = model.domain_to_int["hla_alleles"]
    padding_domain_id = model.domain_to_int["padding"]
    collate = _make_collate(model)

    donor_subjects_list = donor_dataset.subject_list
    donor_idx = 0

    delta_all, ages_all, sexes_all = [], [], []

    for batch in tqdm(
        filtered_dataloader,
        total=len(filtered_dataloader),
        desc=f"Δlogit: {disease_name} vs {allele_name}",
    ):
        batch = batch.to(device)

        with torch.no_grad():
            _, _, h = model(batch, return_embeddings=True)

        logits_orig, idx_prev = disease_prev_logits_from_embeddings(model, h, batch, "diseases", disease_id)
        n_orig = len(logits_orig)
        if n_orig == 0:
            continue

        b_idx, t_prev = idx_prev[:, 0], idx_prev[:, 1]
        ages_at_event = batch.ages[b_idx, t_prev].cpu()
        sids_at_event = batch.subject_ids[b_idx].cpu()
        sexes_at_event = torch.tensor([sex_map.get(int(s), -1) for s in sids_at_event])

        rec_sids = batch.subject_ids.tolist()
        rec_items = [filtered_dataset[filtered_dataset._sid_to_idx[sid]] for sid in rec_sids]

        logits_sw_samples = []
        for _ in tqdm(range(n_counterfactuals), desc="Counterfactuals", leave=False):
            don_items = []
            for _ in rec_sids:
                don_sid = donor_subjects_list[donor_idx % len(donor_subjects_list)]
                donor_idx += 1
                don_items.append(donor_dataset[donor_dataset._sid_to_idx[don_sid]])

            modified_items = [
                inject_hla_item(r, d, hla_domain_int, padding_domain_id)
                for r, d in zip(rec_items, don_items, strict=False)
            ]
            batch_sw = collate(modified_items).to(device)

            with torch.no_grad():
                _, _, h_sw = model(batch_sw, return_embeddings=True)

            logits_sw, _ = disease_prev_logits_from_embeddings(model, h_sw, batch_sw, "diseases", disease_id)
            if len(logits_sw) == 0:
                continue
            n = min(n_orig, len(logits_sw))
            logits_sw_samples.append(logits_sw[:n].cpu())

        if not logits_sw_samples:
            continue

        n = min(n_orig, min(len(s) for s in logits_sw_samples))
        logits_sw_avg = torch.stack([s[:n] for s in logits_sw_samples]).mean(dim=0)
        delta_all.append(logits_orig[:n].cpu() - logits_sw_avg)
        ages_all.append(ages_at_event[:n])
        sexes_all.append(sexes_at_event[:n])

    if not delta_all:
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)

    return torch.cat(delta_all), torch.cat(ages_all), torch.cat(sexes_all)


def process_fold(
    fold,
    runs_df,
    disease_id,
    allele_ids,
    sex_filter,
    n_counterfactuals,
    cache_dir,
    subjects_include=None,
    disease_name="disease",
    allele_name="allele",
    case_zygosity="any",
    case_also_allele_id_groups=None,
    donor_exclude_ids=None,
    donor_require_id_groups=None,
    donor_zygosity="any",
):
    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]

    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to("cpu")
    model.eval()

    if subjects_include is not None:
        test_ids = [int(sid) for sid in test_ids if int(sid) in subjects_include]
        logger.info(f"fold {fold}: {len(test_ids)} subjects after filtering")

    tokens_df = get_tokens_df_cached(model, test_ids, run_id=run_id, cache_dir=cache_dir)
    sex_map = extract_sex_map(tokens_df)

    delta_fold, ages_fold, sexes_fold = compute_delta_for_run(
        model,
        test_ids,
        tokens_df,
        disease_id,
        allele_ids,
        sex_map=sex_map,
        sex_filter=sex_filter,
        n_counterfactuals=n_counterfactuals,
        disease_name=disease_name,
        allele_name=allele_name,
        case_zygosity=case_zygosity,
        case_also_allele_id_groups=case_also_allele_id_groups,
        donor_exclude_ids=donor_exclude_ids,
        donor_require_id_groups=donor_require_id_groups,
        donor_zygosity=donor_zygosity,
    )

    print(f"fold {fold} n={len(delta_fold)}")
    return delta_fold, ages_fold, sexes_fold


def get_allele_pairs_for_scan(
    min_pair_freq: float = 0.005,
    subject_ids=None,
    within_loci=None,
    cross_loci=None,
    include_hom: bool = True,
    min_hom_freq: float | None = None,
) -> list:
    """
    Return list of (a_id, b_id, locus_a, locus_b, a_name, b_name, n_cocarriers, cocarrier_freq)
    tuples where co-carrier frequency >= min_pair_freq.

    within_loci   : loci to enumerate within-locus pairs. None = all loci.
                    Within hla_drb this covers DRB1/DRB3/DRB4/DRB5 combinations.
    cross_loci    : list of (locus_a, locus_b) for cross-locus haplotype pairs.
                    Default: [("hla_dpa","hla_dpb"), ("hla_dqa","hla_dqb")].
    include_hom   : also include homozygous pairs (a_id == b_id).
    min_hom_freq  : minimum homozygous carrier frequency; defaults to min_pair_freq.
    """
    tokens_path = DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokens.csv"
    meta_path = DELPHI_DIR / "data/transforms/tokens/hla_alleles/token_metadata.csv"

    tok = pd.read_csv(tokens_path)
    meta = pd.read_csv(meta_path)

    if subject_ids is not None:
        tok = tok[tok["subject_id"].isin(subject_ids)]
    n_subjects = tok["subject_id"].nunique()

    # subject sets per allele
    allele_to_subj = tok.groupby("token_id")["subject_id"].apply(set).to_dict()

    id_to_name = meta.set_index("token_id")["name"].to_dict()
    id_to_locus = meta.set_index("token_id")["locus"].to_dict()
    locus_to_ids = meta.groupby("locus")["token_id"].apply(list).to_dict()

    all_loci = list(locus_to_ids.keys())
    if within_loci is None:
        within_loci = all_loci
    if cross_loci is None:
        cross_loci = [("hla_dpa", "hla_dpb"), ("hla_dqa", "hla_dqb")]

    result = []
    seen = set()

    # --- het / compound-het pairs ---
    locus_pairs_to_check = [(loc, loc) for loc in within_loci] + list(cross_loci)
    for locus_a, locus_b in locus_pairs_to_check:
        ids_a = locus_to_ids.get(locus_a, [])
        ids_b = locus_to_ids.get(locus_b, [])
        for a_id in ids_a:
            for b_id in ids_b:
                if locus_a == locus_b and a_id >= b_id:
                    continue
                key = (min(a_id, b_id), max(a_id, b_id))
                if key in seen:
                    continue
                seen.add(key)
                s_a = allele_to_subj.get(a_id, set())
                s_b = allele_to_subj.get(b_id, set())
                n = len(s_a & s_b)
                freq = n / n_subjects
                if freq >= min_pair_freq:
                    result.append(
                        (
                            a_id,
                            b_id,
                            locus_a,
                            locus_b,
                            id_to_name.get(a_id, str(a_id)),
                            id_to_name.get(b_id, str(b_id)),
                            n,
                            freq,
                        )
                    )

    # --- homozygous pairs (a_id == b_id) ---
    if include_hom:
        min_h = min_hom_freq if min_hom_freq is not None else min_pair_freq
        copies = tok.groupby(["token_id", "subject_id"]).size().reset_index(name="n_copies")
        hom = copies[copies["n_copies"] >= 2].groupby("token_id")["subject_id"].apply(set)
        for token_id, hom_subj in hom.items():
            freq = len(hom_subj) / n_subjects
            if freq >= min_h:
                locus = id_to_locus.get(token_id, "?")
                name = id_to_name.get(token_id, str(token_id))
                result.append(
                    (
                        token_id,
                        token_id,
                        locus,
                        locus,
                        name,
                        name,
                        len(hom_subj),
                        freq,
                    )
                )

    result.sort(key=lambda r: -r[7])  # descending co-carrier / hom freq
    return result


def get_qualifying_allele_ids(min_carrier_freq: float, subject_ids=None) -> list:
    """
    Return HLA allele token IDs whose carrier frequency exceeds min_carrier_freq.

    Reads the HLA tokens file directly — no model loading required.
    Carrier frequency = fraction of subjects with ≥1 copy of the allele.
    subject_ids: if given, restrict the population to this set.
    """
    tokens_path = DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokens.csv"
    df = pd.read_csv(tokens_path)
    if subject_ids is not None:
        df = df[df["subject_id"].isin(subject_ids)]
    n_subjects = df["subject_id"].nunique()
    carrier_freq = df.groupby("token_id")["subject_id"].nunique() / n_subjects
    return sorted(carrier_freq[carrier_freq >= min_carrier_freq].index.tolist())


def process_fold_scan(
    fold,
    runs_df,
    disease_id,
    allele_ids_to_scan,
    sex_filter,
    n_counterfactuals,
    cache_dir,
    subjects_include=None,
    case_zygosity="hom",
    donor_zygosity="any",
):
    """
    Load model and tokens_df once for this fold, then compute Δlogit for every
    allele in allele_ids_to_scan.  Returns {allele_id: (delta, ages, sexes)}.

    Donors for each allele are its non-carriers (default behaviour).
    """
    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]
    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to("cpu").eval()

    if subjects_include is not None:
        test_ids = [int(sid) for sid in test_ids if int(sid) in subjects_include]
        logger.info(f"fold {fold}: {len(test_ids)} subjects after filtering")

    tokens_df = get_tokens_df_cached(model, test_ids, run_id=run_id, cache_dir=cache_dir)
    sex_map = extract_sex_map(tokens_df)

    results = {}
    for allele_id in tqdm(allele_ids_to_scan, desc=f"fold {fold} — alleles"):
        allele_name = hla_tokenizer[allele_id]
        delta, ages, sexes = compute_delta_for_run(
            model,
            test_ids,
            tokens_df,
            disease_id,
            [allele_id],
            sex_map=sex_map,
            sex_filter=sex_filter,
            n_counterfactuals=n_counterfactuals,
            disease_name="(scan)",
            allele_name=allele_name,
            case_zygosity=case_zygosity,
            donor_zygosity=donor_zygosity,
        )
        if len(delta) > 0:
            results[allele_id] = (delta, ages, sexes)

    return results


def process_fold_scan_pairs(
    fold,
    runs_df,
    disease_id,
    pairs_to_scan,
    sex_filter,
    n_counterfactuals,
    cache_dir,
    subjects_include=None,
    case_zygosity="any",
    donor_zygosity="any",
):
    """
    Load model + tokens_df once for this fold, then compute Δlogit for every
    allele pair in pairs_to_scan.

    pairs_to_scan : list of tuples as returned by get_allele_pairs_for_scan:
        (a_id, b_id, locus_a, locus_b, a_name, b_name, n_cocarriers, cocarrier_freq)

    Cases  : subjects carrying BOTH a and b.
    Donors : subjects carrying NEITHER a nor b.

    Returns {(a_id, b_id): (delta, ages, sexes)}.
    """
    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]
    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to("cpu").eval()

    if subjects_include is not None:
        test_ids = [int(sid) for sid in test_ids if int(sid) in subjects_include]
        logger.info(f"fold {fold}: {len(test_ids)} subjects after filtering")

    tokens_df = get_tokens_df_cached(model, test_ids, run_id=run_id, cache_dir=cache_dir)
    sex_map = extract_sex_map(tokens_df)

    results = {}
    for a_id, b_id, _locus_a, _locus_b, a_name, b_name, _, _ in tqdm(pairs_to_scan, desc=f"fold {fold} — pairs"):
        delta, ages, sexes = compute_delta_for_run(
            model,
            test_ids,
            tokens_df,
            disease_id,
            allele_ids=[a_id],
            sex_map=sex_map,
            sex_filter=sex_filter,
            n_counterfactuals=n_counterfactuals,
            disease_name="(scan_pairs)",
            allele_name=f"{a_name}+{b_name}",
            case_zygosity=case_zygosity,
            case_also_allele_id_groups=[[b_id]],
            donor_exclude_ids=[a_id, b_id],
            donor_zygosity=donor_zygosity,
        )
        if len(delta) > 0:
            results[(a_id, b_id)] = (delta, ages, sexes)

    return results


def _resolve_output_path(user_path, default: Path) -> Path:
    """Resolve a user-supplied output path against the invocation CWD.
    Falls back to `default` (an absolute path) if user_path is None."""
    if user_path is None:
        return default
    p = Path(user_path)
    return p if p.is_absolute() else _INVOCATION_CWD / p


def _resolve_allele_spec(spec: str) -> list:
    """Expand an allele prefix or exact name to a list of matching token IDs."""
    ids = [i for i, name in enumerate(hla_tokenizer) if name.startswith(spec)]
    if not ids:
        raise SystemExit(f"No HLA alleles found matching prefix {spec!r}.")
    return ids


def _resolve_experiment_id(prefix: str) -> str:
    """
    Return the unique experiment ID whose string starts with `prefix`.
    Aborts if zero or more than one match is found.
    """
    all_experiments = mlflow.search_experiments()
    matches = [e for e in all_experiments if e.experiment_id.startswith(prefix)]
    if not matches:
        raise SystemExit(f"No experiment found with ID prefix '{prefix}'.")
    if len(matches) > 1:
        candidates = "\n  ".join(f"{e.experiment_id}  ({e.name})" for e in matches)
        raise SystemExit(f"Ambiguous prefix '{prefix}' matches {len(matches)} experiments:\n  {candidates}")
    return matches[0].experiment_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="MLflow experiment ID or unique prefix thereof.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        metavar="REGEX",
        help="Regex applied to run name; only matching runs are kept.",
    )
    parser.add_argument(
        "--param",
        type=str,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Filter by parameter value: NAME=VALUE. Can be repeated.",
    )
    parser.add_argument("--disease", type=str)
    parser.add_argument("--disease_id", type=int)
    parser.add_argument("--hla_allele", type=str)
    parser.add_argument("--allele_id", type=int)
    parser.add_argument(
        "--allele_id_b",
        type=int,
        default=None,
        help="Second allele ID forming a pair with --allele_id. "
        "If equal to --allele_id, runs in homozygous mode (case_zygosity=hom). "
        "Donors automatically exclude both alleles.",
    )
    parser.add_argument(
        "--generate_pairs",
        action="store_true",
        help="Generate the list of qualifying allele pairs and write a TSV ready for "
        "sarray_params, then exit without running any model.",
    )
    # --- scan mode ---
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan all qualifying alleles instead of a single one. Ignores --hla_allele/--allele_id.",
    )
    parser.add_argument(
        "--min_carrier_freq",
        type=float,
        default=0.01,
        help="Minimum carrier frequency for qualifying alleles in --scan mode (default: 0.01).",
    )
    parser.add_argument(
        "--scan_pairs",
        action="store_true",
        help="Scan all qualifying allele pairs (within-locus + DPA/DPB, DQA/DQB). Ignores --hla_allele/--allele_id.",
    )
    parser.add_argument(
        "--min_pair_freq",
        type=float,
        default=0.005,
        help="Minimum co-carrier frequency for pairs in --scan_pairs mode (default: 0.005).",
    )
    parser.add_argument(
        "--cross_loci",
        type=str,
        action="append",
        default=None,
        metavar="LOCUS_A:LOCUS_B",
        help="Cross-locus pairs to include in --scan_pairs / --generate_pairs, "
        "e.g. hla_dpa:hla_dpb. Repeatable. "
        "Default: hla_dpa:hla_dpb and hla_dqa:hla_dqb.",
    )
    parser.add_argument(
        "--no_hom",
        action="store_true",
        help="Exclude homozygous pairs from --generate_pairs / --scan_pairs.",
    )
    parser.add_argument(
        "--pairs_output",
        type=str,
        default=None,
        help="Output path for the sarray_params TSV generated by --generate_pairs. "
        "Defaults to pairs_{disease_id}.tsv in the current directory.",
    )
    parser.add_argument("--n_counterfactuals", type=int, default=5)
    parser.add_argument(
        "--sex",
        type=str,
        default=None,
        choices=["male", "female", "both"],
        help="Restrict cases and donors to this sex. 'both' or omitted = no filtering.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Path to file with subject IDs to intersect with test set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pkl path. Supports {disease_id}, {allele_id}, {sex} placeholders.",
    )
    parser.add_argument(
        "--dry-run",
        "--dryrun",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Print the disease, alleles and runs that would be processed, then exit.",
    )
    # --- case genotype ---
    parser.add_argument(
        "--case_zygosity",
        choices=["any", "het", "hom"],
        default="any",
        help="Zygosity of the primary allele in cases. 'het'=exactly 1 copy, 'hom'=≥2 copies.",
    )
    parser.add_argument(
        "--case_also",
        type=str,
        action="append",
        default=[],
        metavar="ALLELE_PREFIX",
        help="Cases must also carry ≥1 allele matching this prefix (AND logic, repeatable).",
    )
    # --- donor genotype ---
    parser.add_argument(
        "--donor_excludes",
        type=str,
        action="append",
        default=None,
        metavar="ALLELE_PREFIX",
        help="Alleles donors must NOT carry. Overrides the default (case alleles). Repeatable.",
    )
    parser.add_argument(
        "--donor_requires",
        type=str,
        action="append",
        default=[],
        metavar="ALLELE_PREFIX",
        help="Alleles donors MUST carry (AND logic, repeatable).",
    )
    parser.add_argument(
        "--donor_zygosity",
        choices=["any", "het", "hom"],
        default="any",
        help="Zygosity applied to each --donor_requires group.",
    )
    args = parser.parse_args()

    experiment_id = _resolve_experiment_id(args.experiment_id)

    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
    runs_df = runs_df.loc[:, runs_df.nunique() > 1]
    runs_df = runs_df.rename(columns=lambda c: c.replace("metrics.", "").replace("params.", "").replace("tags.", ""))
    runs_df = runs_df.loc[:, ~runs_df.columns.duplicated()]

    if args.run_name is not None:
        mask = runs_df["mlflow.runName"].str.contains(args.run_name, regex=True, na=False)
        runs_df = runs_df[mask]
        if runs_df.empty:
            raise SystemExit(f"No runs matched --run_name pattern '{args.run_name}'.")

    for spec in args.param:
        if "=" not in spec:
            raise SystemExit(f"Invalid --param format: '{spec}'. Expected NAME=REGEX.")
        col, pattern = spec.split("=", 1)
        col = col.strip()
        if col not in runs_df.columns:
            available = [c for c in runs_df.columns if not c.startswith("mlflow.")]
            raise SystemExit(f"Parameter '{col}' not found. Available params: {available}")
        mask = runs_df[col].astype(str) == pattern
        runs_df = runs_df[mask]
        if runs_df.empty:
            raise SystemExit(f"No runs matched --param '{spec}'.")

    runs_df = runs_df.sort_values("test_fold").reset_index(drop=True)
    runs_df["test_fold"] = runs_df["test_fold"].astype(int)
    print(f"Selected {len(runs_df)} run(s) across fold(s): {sorted(runs_df.test_fold.unique())}")
    if args.sex == "both":
        args.sex = None

    # --- disease ---
    if args.disease_id is not None:
        disease_id = args.disease_id
        disease_name = disease_tokenizer[disease_id]
    elif args.disease is not None:
        disease_id, disease_name = find_most_similar_token(args.disease, disease_tokenizer)
    else:
        raise ValueError("You must provide either --disease or --disease_id")

    # --- subjects ---
    subjects_include = None
    if args.subjects is not None:
        assert Path(args.subjects).exists(), f"File {args.subjects} does not exist."
        subjects_include = set(read_ids(args.subjects))
        logger.info(f"Loaded {len(subjects_include)} subject IDs from {args.subjects}")

    sex_label = args.sex or "both_sexes"
    n_folds = runs_df.test_fold.nunique()

    # ------------------------------------------------------------------ #
    #  GENERATE-PAIRS MODE: write TSV for sarray_params, then exit         #
    # ------------------------------------------------------------------ #
    if args.generate_pairs:
        cross_loci = None
        if args.cross_loci:
            cross_loci = [tuple(s.split(":")) for s in args.cross_loci]

        pairs = get_allele_pairs_for_scan(
            min_pair_freq=args.min_pair_freq,
            subject_ids=subjects_include,
            cross_loci=cross_loci,
            include_hom=not args.no_hom,
        )
        print(
            f"{len(pairs)} pairs with freq ≥ {args.min_pair_freq:.1%} "
            f"({'including' if not args.no_hom else 'excluding'} hom)"
        )

        # --- sarray_params TSV: only allele_id and allele_id_b ---
        # Column names become --allele_id and --allele_id_b when submitted.
        pairs_tsv_path = _resolve_output_path(
            args.pairs_output,
            DELPHI_DIR / "shap" / f"pairs_{disease_id}.tsv",
        )
        rows_args = [{"allele_id": a_id, "allele_id_b": b_id} for a_id, b_id, *_ in pairs]
        pd.DataFrame(rows_args).to_csv(pairs_tsv_path, sep="\t", index=False)
        print(f"sarray_params TSV  → {pairs_tsv_path}")

        # --- full metadata TSV for reference ---
        meta_tsv_path = pairs_tsv_path.with_suffix(".meta.tsv")
        rows_meta = [
            {
                "allele_id": a_id,
                "allele_id_b": b_id,
                "allele_name_a": a_name,
                "allele_name_b": b_name,
                "locus_a": la,
                "locus_b": lb,
                "n_cocarriers": n,
                "cocarrier_freq": f"{freq:.4f}",
                "pair_type": "hom" if a_id == b_id else ("within" if la == lb else "cross"),
            }
            for a_id, b_id, la, lb, a_name, b_name, n, freq in pairs
        ]
        pd.DataFrame(rows_meta).to_csv(meta_tsv_path, sep="\t", index=False)
        print(f"Metadata TSV       → {meta_tsv_path}")

        print("\nExample submission:")
        print(f"  sarray_params shap/custom_hla_shap_v2.py {pairs_tsv_path} \\")
        print(f"    --experiment_id {args.experiment_id} \\")
        print(f"    --disease_id {disease_id} \\")
        print(f"    --n_counterfactuals {args.n_counterfactuals} \\")
        print("    --time=06:00:00 --mem=32G --cpus=4")
        return

    # ------------------------------------------------------------------ #
    #  SCAN-PAIRS MODE: all qualifying allele pairs                        #
    # ------------------------------------------------------------------ #
    if args.scan_pairs:
        cross_loci = None
        if args.cross_loci:
            cross_loci = [tuple(s.split(":")) for s in args.cross_loci]

        pairs_to_scan = get_allele_pairs_for_scan(
            min_pair_freq=args.min_pair_freq,
            subject_ids=subjects_include,
            cross_loci=cross_loci,
        )
        print(f"{len(pairs_to_scan)} allele pairs with co-carrier freq ≥ {args.min_pair_freq:.1%}")

        if args.dry_run:
            print("\n--- DRY RUN (scan_pairs) ---")
            print(f"Disease : {disease_name} (id={disease_id})")
            print(f"Pairs   : {len(pairs_to_scan)} (showing top 10)")
            for _a_id, _b_id, la, lb, a_name, b_name, n, freq in pairs_to_scan[:10]:
                print(f"  {a_name} + {b_name}  ({la}/{lb})  n_cocarriers={n}  freq={freq:.3f}")
            print(f"Runs ({n_folds}):")
            for _, row in runs_df.iterrows():
                print(f"  fold={row['test_fold']}  run_id={row['run_id']}")
            raise SystemExit(0)

        fold_pair_results = list(
            tqdm(
                Parallel(n_jobs=-1, backend="loky", return_as="generator")(
                    delayed(process_fold_scan_pairs)(
                        fold,
                        runs_df,
                        disease_id,
                        pairs_to_scan,
                        args.sex,
                        args.n_counterfactuals,
                        CACHE_DIR,
                        subjects_include,
                        case_zygosity=args.case_zygosity,
                        donor_zygosity=args.donor_zygosity,
                    )
                    for fold in sorted(runs_df.test_fold.unique())
                ),
                total=n_folds,
                desc="Folds",
            )
        )

        output_dir = _resolve_output_path(args.output, DELPHI_DIR / "shap/output_delta_logit/scan_pairs")
        output_dir.mkdir(parents=True, exist_ok=True)

        MIN_N_WILCOXON = 10
        {(r[0], r[1]): r for r in pairs_to_scan}
        summary_rows = []

        for a_id, b_id, locus_a, locus_b, a_name, b_name, n_cocarriers, cocarrier_freq in pairs_to_scan:
            key = (a_id, b_id)
            parts = [res[key] for res in fold_pair_results if key in res]
            if not parts:
                continue
            delta = torch.cat([p[0] for p in parts]).numpy()
            ages = torch.cat([p[1] for p in parts]).numpy()
            sexes = torch.cat([p[2] for p in parts]).numpy()
            age_brackets = np.array([assign_age_bracket(a) for a in ages])

            out_path = output_dir / f"{disease_id}__{a_id}-{b_id}__{sex_label}.pkl"
            with out_path.open("wb") as _fh:
                pkl.dump({"delta": delta, "ages": ages, "sexes": sexes, "age_brackets": age_brackets}, _fh)

            n = len(delta)
            mean_d = float(delta.mean())
            p = float(stats.wilcoxon(delta).pvalue) if n >= MIN_N_WILCOXON else float("nan")
            summary_rows.append(
                {
                    "allele_id_a": a_id,
                    "allele_name_a": a_name,
                    "allele_id_b": b_id,
                    "allele_name_b": b_name,
                    "locus_a": locus_a,
                    "locus_b": locus_b,
                    "n_cocarriers": n_cocarriers,
                    "cocarrier_freq": cocarrier_freq,
                    "n_cases": n,
                    "mean_delta": mean_d,
                    "p_wilcoxon": p,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("mean_delta", ascending=False).reset_index(drop=True)
        summary_path = output_dir / f"{disease_id}__scan_pairs__{sex_label}.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(summary_df.to_string(index=False))
        print(f"\nSummary → {summary_path}")
        return

    # ------------------------------------------------------------------ #
    #  SCAN MODE: iterate over all qualifying alleles                      #
    # ------------------------------------------------------------------ #
    if args.scan:
        qualifying_ids = get_qualifying_allele_ids(args.min_carrier_freq, subjects_include)
        print(f"{len(qualifying_ids)} alleles with carrier freq ≥ {args.min_carrier_freq:.1%}")

        if args.dry_run:
            print("\n--- DRY RUN (scan) ---")
            print(f"Disease        : {disease_name} (id={disease_id})")
            print(f"Qualifying     : {len(qualifying_ids)} alleles")
            print(f"Case zygosity  : {args.case_zygosity}")
            print(f"Donor zygosity : {args.donor_zygosity}")
            print(f"Runs ({n_folds}):")
            for _, row in runs_df.iterrows():
                print(f"  fold={row['test_fold']}  run_id={row['run_id']}")
            raise SystemExit(0)

        fold_scan_results = list(
            tqdm(
                Parallel(n_jobs=-1, backend="loky", return_as="generator")(
                    delayed(process_fold_scan)(
                        fold,
                        runs_df,
                        disease_id,
                        qualifying_ids,
                        args.sex,
                        args.n_counterfactuals,
                        CACHE_DIR,
                        subjects_include,
                        case_zygosity=args.case_zygosity,
                        donor_zygosity=args.donor_zygosity,
                    )
                    for fold in sorted(runs_df.test_fold.unique())
                ),
                total=n_folds,
                desc="Folds",
            )
        )

        output_dir = _resolve_output_path(args.output, DELPHI_DIR / "shap/output_delta_logit/scan")
        output_dir.mkdir(parents=True, exist_ok=True)

        MIN_N_WILCOXON = 10
        summary_rows = []
        for allele_id in qualifying_ids:
            allele_name = hla_tokenizer[allele_id]
            parts = [res[allele_id] for res in fold_scan_results if allele_id in res]
            if not parts:
                continue
            delta = torch.cat([p[0] for p in parts]).numpy()
            ages = torch.cat([p[1] for p in parts]).numpy()
            sexes = torch.cat([p[2] for p in parts]).numpy()
            age_brackets = np.array([assign_age_bracket(a) for a in ages])

            out_path = output_dir / f"{disease_id}__{allele_id}__{sex_label}.pkl"
            with out_path.open("wb") as _fh:
                pkl.dump({"delta": delta, "ages": ages, "sexes": sexes, "age_brackets": age_brackets}, _fh)

            n = len(delta)
            mean_d = float(delta.mean())
            if n >= MIN_N_WILCOXON:
                _, p = stats.wilcoxon(delta)
            else:
                p = float("nan")
            summary_rows.append(
                {
                    "allele_id": allele_id,
                    "allele_name": allele_name,
                    "n_hom_cases": n,
                    "mean_delta": mean_d,
                    "p_wilcoxon": p,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("mean_delta", ascending=False).reset_index(drop=True)
        summary_path = output_dir / f"{disease_id}__scan_hom__{sex_label}.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(summary_df.to_string(index=False))
        print(f"\nSummary → {summary_path}")
        return

    # ------------------------------------------------------------------ #
    #  SINGLE-ALLELE MODE (original behaviour)                             #
    # ------------------------------------------------------------------ #

    # --- allele ---
    if args.allele_id is not None:
        allele_ids = [args.allele_id]
        allele_name = hla_tokenizer[args.allele_id]
    elif args.hla_allele is not None:
        allele_ids = [i for i, hla in enumerate(hla_tokenizer) if hla.startswith(args.hla_allele)]
        allele_name = hla_tokenizer[allele_ids[0]]
    else:
        raise ValueError("Provide --hla_allele, --allele_id, or --scan")

    # --- genotype filter resolution ---
    case_also_allele_id_groups = [_resolve_allele_spec(s) for s in args.case_also]
    donor_exclude_ids = (
        None if args.donor_excludes is None else [iid for s in args.donor_excludes for iid in _resolve_allele_spec(s)]
    )
    donor_require_id_groups = [_resolve_allele_spec(s) for s in args.donor_requires]

    # --- pair mode via --allele_id_b ---
    if args.allele_id_b is not None:
        b_id = args.allele_id_b
        b_name = hla_tokenizer[b_id]
        if b_id == allele_ids[0]:
            # homozygous: same allele twice — override case_zygosity unless user set it explicitly
            if args.case_zygosity == "any":
                args.case_zygosity = "hom"
            allele_name = f"{allele_name} [hom]"
            # donors: non-carriers (donor_exclude_ids stays as allele_ids if not overridden)
        else:
            # compound het / cross-locus pair
            case_also_allele_id_groups = [*case_also_allele_id_groups, [b_id]]
            allele_name = f"{allele_name}+{b_name}"
            # donors must carry neither allele
            donor_exclude_ids = [*allele_ids, b_id] if donor_exclude_ids is None else [*donor_exclude_ids, b_id]

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Disease : {disease_name} (id={disease_id})")
        print(f"Allele  : {allele_name} (ids={allele_ids})")
        print(f"Case zygosity  : {args.case_zygosity}")
        if case_also_allele_id_groups:
            print(f"Case also      : {args.case_also}")
        _excl_label = args.donor_excludes if args.donor_excludes is not None else "(case alleles)"
        print(f"Donor excludes : {_excl_label}")
        if donor_require_id_groups:
            print(f"Donor requires : {args.donor_requires}  zygosity={args.donor_zygosity}")
        print(f"Runs    ({len(runs_df)}):")
        for _, row in runs_df.iterrows():
            print(f"  fold={row['test_fold']}  run_id={row['run_id']}  name={row['mlflow.runName']}")
        raise SystemExit(0)

    print(f"Computing Δlogit for {disease_name} and {allele_name}" + (f" (sex={args.sex})" if args.sex else ""))

    # --- output ---
    _allele_stem = f"{allele_ids[0]}-{args.allele_id_b}" if args.allele_id_b is not None else str(allele_ids[0])
    output_file = Path(
        str(
            _resolve_output_path(
                args.output,
                DELPHI_DIR / "shap" / "output_delta_logit" / f"{disease_id}__{_allele_stem}__{sex_label}.pkl",
            )
        ).format(disease_id=disease_id, allele_id=_allele_stem, sex=sex_label)
    )

    fold_results = list(
        tqdm(
            Parallel(n_jobs=-1, backend="loky", return_as="generator")(
                delayed(process_fold)(
                    fold,
                    runs_df,
                    disease_id,
                    allele_ids,
                    args.sex,
                    args.n_counterfactuals,
                    CACHE_DIR,
                    subjects_include,
                    disease_name,
                    allele_name,
                    case_zygosity=args.case_zygosity,
                    case_also_allele_id_groups=case_also_allele_id_groups,
                    donor_exclude_ids=donor_exclude_ids,
                    donor_require_id_groups=donor_require_id_groups,
                    donor_zygosity=args.donor_zygosity,
                )
                for fold in sorted(runs_df.test_fold.unique())
            ),
            total=n_folds,
            desc="Folds",
        )
    )

    all_delta, all_ages, all_sexes = zip(*fold_results, strict=False)
    delta = torch.cat(all_delta).numpy()
    ages = torch.cat(all_ages).numpy()
    sexes = torch.cat(all_sexes).numpy()
    age_brackets = np.array([assign_age_bracket(a) for a in ages])

    # --- summary ---
    MIN_N_WILCOXON = 10
    _stat, p_two_sided = stats.wilcoxon(delta)
    print(f"mean Δlogit = {delta.mean():.4f}")
    print(f"Wilcoxon two-sided p = {p_two_sided:.2e}")

    groupby_cols = ["age_bracket", "sex"] if args.sex is None else ["age_bracket"]
    summary_df = pd.DataFrame(
        {
            "delta": delta,
            "age_bracket": age_brackets,
            "sex": np.vectorize(INT_TO_SEX.get)(sexes, "unknown"),
        }
    )

    print(f"\nΔlogit summary: {disease_name}  |  {allele_name}" + (f"  |  sex={args.sex}" if args.sex else ""))
    header = f"{'Age bracket':<12}  {'Sex':<8}  {'n':>6}  {'mean Δlogit':>12}  {'p (Wilcoxon)':>14}"
    print(header)
    print("-" * len(header))

    for keys, grp in summary_df.groupby(groupby_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        bracket = keys[0]
        sex_col = keys[1] if len(keys) > 1 else (args.sex or "all")
        n = len(grp)
        mean_d = grp["delta"].mean()
        if n >= MIN_N_WILCOXON:
            _, p = stats.wilcoxon(grp["delta"])
            p_str = f"{p:.2e}"
        else:
            p_str = f"n<{MIN_N_WILCOXON}"
        print(f"{bracket!s:<12}  {sex_col!s:<8}  {n:>6}  {mean_d:>12.4f}  {p_str:>14}")

    # --- save ---
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_file).open("wb") as _fh:
        pkl.dump({"delta": delta, "ages": ages, "sexes": sexes, "age_brackets": age_brackets}, _fh)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
