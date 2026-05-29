# HLA SHAP analysis

Counterfactual attribution of HLA allele effects on disease risk using the Delphi model.

## Method

For each case subject (carrying a given HLA allele **and** having a given disease), their
HLA block is swapped with that of a randomly chosen non-carrier (donor). The change in
predicted disease logit is:

```
Δlogit = logit(original HLA) − logit(donor HLA)
```

A positive mean Δlogit means the allele increases predicted disease risk.
Statistical significance is assessed with a two-sided Wilcoxon signed-rank test.
All analyses are restricted to white-ancestry subjects by default
(`data/transforms/subject_lists/genetic_white_ids.txt`).

---

## Single-allele analysis

### Run one allele × disease

```bash
python shap/custom_hla_shap.py \
    --experiment_id 263078128312970150 \
    --disease_id <int> \
    --hla_allele "HLA-C*06" \
    --n_counterfactuals 5 \
    --subjects data/transforms/subject_lists/genetic_white_ids.txt \
    --run_name "hla_mix" --param n_head=12
```

- `--disease_id` is the 0-based index in `data/transforms/tokens/diseases/tokenizer.yaml`.
  Check `data/transforms/tokens/diseases/token_metadata.tsv` for the mapping.
- `--hla_allele` accepts a prefix; all matching alleles are grouped together.
  Use `--allele_id <int>` for an exact token ID.
- Add `--dry-run` to preview without running.

### Full scan: all alleles × all diseases × sex (cluster)

`prepare_single_allele_jobs.py` pre-computes co-occurrence counts and generates
a `sarray_params`-ready TSV covering every (disease, allele, sex) combination
with enough cases, restricted to white-ancestry subjects (`genetic_white_ids.txt`).

The TSV has columns `disease_id`, `allele_id`, `sex`, `output`. The `sex` column
takes values `both` (no filtering), `male`, or `female` — never empty, to avoid
bash IFS parsing issues with consecutive tab delimiters.

#### Step 1 — Generate the jobs TSV

```bash
python shap/prepare_single_allele_jobs.py \
    --min_count 10 \
    --output shap/single_allele_jobs.tsv
```

Options:
- `--min_count INT` — minimum co-occurrence count per (disease, allele, sex) (default: 10)
- `--sexes both male female` — which sex strata to include (default: all three)
- `--subjects PATH` — subject list (default: `genetic_white_ids.txt`)

The script prints a summary of how many jobs pass the filter per stratum.

#### Step 2 — Preview a few jobs

```bash
source ~/repos/codon_helpers/slurm_functions.sh
sarray_params shap/custom_hla_shap.py shap/single_allele_jobs.tsv \
    --experiment_id 263078128312970150 \
    --subjects data/transforms/subject_lists/genetic_white_ids.txt \
    --n_counterfactuals 5 \
    --run_name hla_mix --param n_head=12 \
    --time=06:00:00 --mem=32G --cpus=4 \
    --lines=1-5 --dry-run
```

#### Step 3 — Submit

```bash
sarray_params shap/custom_hla_shap.py shap/single_allele_jobs.tsv \
    --experiment_id 263078128312970150 \
    --subjects data/transforms/subject_lists/genetic_white_ids.txt \
    --n_counterfactuals 5 \
    --run_name hla_mix --param n_head=12 \
    --time=06:00:00 --mem=32G --cpus=4
```

> **Note:** SLURM's `MaxArraySize` on Codon is 80000. If the TSV has more rows,
> `sarray_params` will error and suggest how to split. Submit in batches using
> `--lines=1-80000`, then `--lines=80001-N`, etc.

### Output

`shap/output_delta_logit/{disease_id}__{allele_id}__{sex_label}.pkl`

Contains a dict with keys `delta`, `ages`, `sexes`, `age_brackets` (all 1-D numpy arrays,
one entry per disease-preceding position across all CV folds).

### Aggregate results

```bash
python shap/compute_pvalues_table.py \
    --pkl_dir shap/output_delta_logit \
    --output  shap/output_delta_logit/summary_single.csv \
    [--min_n 10] [--sex both_sexes]
```

Reads all `{disease_id}__{allele_id}__{sex}.pkl` files and writes `summary_single.csv`
(default: `<pkl_dir>/summary_single.csv`).

Output columns: `disease_id`, `disease_name`, `allele_id`, `allele_name`, `sex`,
`n_subjects`, `mean_delta`, `median_delta`, `wilcoxon_stat`, `p_value`, plus per-age-bracket
columns `n_{bracket}`, `mean_delta_{bracket}`, `p_{bracket}` for brackets
`0_20`, `10_30`, `20_40`, `30_50`, `40_60`, `50_70`, `60_80`.

- `--min_n INT` — minimum cases required to compute Wilcoxon (applies globally and per bracket; default: 10)
- `--sex STR` — restrict to pkl files with this sex label (e.g. `both_sexes`)

---

## Allele-pair analysis (compound-het / hom)

Estimates the joint effect of two HLA alleles carried together.

### Full pipeline (cluster)

#### Step 1 — Generate allele pairs per disease

```bash
./shap_gen_allele_pairs.sh [--dry-run]
```

Submits one SLURM job per disease. Each job discovers co-occurring allele pairs above
`--min_pair_freq` and writes:

- `shap/pairs_{ICD10}.tsv` — parameter file for SLURM array submission
- `shap/pairs_{ICD10}.meta.tsv` — metadata (allele names, loci, co-carrier frequency)

Override defaults via env vars:
```bash
MIN_SUBJECTS=500 MAX_SUBJECTS=20000 ./shap_gen_allele_pairs.sh
```

#### Step 2 — Compute Δlogit for each pair (cluster)

```bash
./shap_submit_allele_pairs.sh [--dry-run]
```

For each disease that has a `pairs_{ICD10}.tsv`, submits a SLURM array job where each
task is one allele pair. Results are written to:

```
shap/output_delta_logit/{disease_id}__{allele_id_a}-{allele_id_b}__{sex_label}.pkl
```

#### Step 3 — Aggregate

```bash
python shap/compute_pvalues_table_pairs.py \
    --pkl_dir  shap/output_delta_logit \
    --meta_dir shap \
    --output   shap/output_delta_logit/summary_pairs.tsv
```

Writes `summary_pairs.tsv` with one row per (disease × allele pair × sex), including
allele names, loci, pair type, co-carrier frequency, n_cases, mean_delta, and p_wilcoxon.

#### Step 4 — Visualise

Load `summary_pairs.tsv` in the **Epistasis** tab of the Streamlit app:

```bash
streamlit run delphi-analysis/app_auc_comparison_HLA.py
```

### Run a single pair manually

```bash
python shap/custom_hla_shap.py \
    --experiment_id 263078128312970150 \
    --disease_id <int> \
    --allele_id <int_a> \
    --allele_id_b <int_b> \
    --n_counterfactuals 5 \
    --subjects data/transforms/subject_lists/genetic_white_ids.txt \
    --run_name "hla_mix" --param n_head=12
```

If `--allele_id` == `--allele_id_b`, runs in **homozygous** mode (`case_zygosity=hom`).
Otherwise, cases must carry **both** alleles (compound-het / cross-locus haplotype).

---

## Advanced options (`custom_hla_shap.py`)

| Option | Description |
|--------|-------------|
| `--sex male\|female` | Restrict cases and donors to one sex |
| `--case_zygosity any\|het\|hom` | Zygosity of the primary allele in cases |
| `--case_also ALLELE_PREFIX` | Cases must also carry this allele (repeatable) |
| `--donor_excludes ALLELE_PREFIX` | Donors must NOT carry this (overrides default) |
| `--donor_requires ALLELE_PREFIX` | Donors MUST carry this (repeatable) |
| `--donor_zygosity any\|het\|hom` | Zygosity for `--donor_requires` |
| `--min_pair_freq FLOAT` | Min co-carrier frequency for `--generate_pairs` (default: 0.005) |
| `--n_counterfactuals INT` | Donor draws per batch to average (default: 5) |
| `--dry-run` | Print what would run without submitting |

---

## Shell script env-var overrides

Both `shap_gen_allele_pairs.sh` and `shap_submit_allele_pairs.sh` (at repo root) read:

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_SUBJECTS` | 300 (gen) / 10000 (submit) | Min cases to include a disease |
| `MAX_SUBJECTS` | 30000 | Max cases to include a disease |
| `SUBJECTS` | `genetic_white_ids.txt` | Subject list for ancestry filtering |
| `EXPERIMENT_ID` | 263078128312970150 | MLflow experiment |
