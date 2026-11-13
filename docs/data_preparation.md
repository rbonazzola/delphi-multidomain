# Data preparation

This page tells you how to create the input data for `train.py`: one directory per domain, plus the cross-validation subject lists.

> `data/README.md` describes the **legacy** format of the original Delphi (`train.bin` / `val.bin` with one global vocabulary). `train.py` does not read that format.

## 1. Directory layout

`train.py --data_root DIR` (default: `data/transforms`) expects:

```
DIR/
├── tokens/
│   ├── diseases/
│   │   ├── tokenizer.yaml
│   │   ├── tokens.csv
│   │   └── token_metadata.tsv      (optional)
│   ├── sex/
│   │   ├── tokenizer.yaml
│   │   └── tokens.csv
│   └── <one directory per domain>/
└── subject_lists/
    ├── subset1of10.csv
    ├── ...
    └── subset10of10.csv
```

The `path` field of each domain in the domain config YAML is relative to `DIR/tokens/`. An absolute `path` is also accepted.

## 2. `tokenizer.yaml`

A YAML list with one entry per token. The list index is the `token_id` (0-based).

```yaml
- bmi_low
- bmi_mid
- bmi_high
```

Here `bmi_low` is `token_id` 0. Do not reserve IDs for padding or no-event tokens: the model adds them as separate pseudo-domains.

For continuous domains, the list names the dimensions (`PC01`, `PC02`, ...). Its length must be equal to `input_size` in the domain config.

## 3. `tokens.csv`

A CSV file with a header row. The row order is not important: the loader sorts the rows.

### Categorical domains

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Subject identifier |
| `age` | int | Age at the event, **in days** |
| `token_id` | int | Index into `tokenizer.yaml` |

```csv
subject_id,age,token_id
1000015,17900,1169
1000015,19169,298
```

- One row per event. A subject can have many rows.
- For `at_birth: true` domains (sex, HLA alleles, rare variants), the `age` column is optional. The loader sets all ages to 0.
- For domains with `no_repeat: true` (diseases), write each token at most once per subject (first occurrence).

### Continuous domains

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Subject identifier |
| `token_id` | int | Dimension index (0 … `input_size`−1) |
| `value` | float | Value of that dimension |
| `age` | int | Optional; days. Omit it for `at_birth` domains |

```csv
subject_id,token_id,value
1000015,0,-15.4408
1000015,1,4.7077
```

- Write all `input_size` rows for each subject.
- A subject with no rows gets a zero vector and no placeholder tokens in the sequence.

> ⚠️ **Fixed bug (2026-11-13).** From commit `05bd8a3` (2026-01-14) to 2026-11-13, `TokenDomain._load_tokens` sorted the rows with `df.sort_values("subject_id")`, which is not a stable sort. The loader copies the values into the input vector in row order, so the dimensions were permuted differently for each subject (99.4 % of subjects for `genetic_pcs`, 100 % for NMR). The loader now sorts by `["subject_id", "token_id"]`. Models trained with a continuous domain before the fix saw permuted inputs: if you evaluate them with the current code, their results will not match the logged values.

## 4. `token_metadata` (optional)

Some features need a per-token metadata table in the domain directory:

| Feature | File read | Required columns |
|---------|-----------|------------------|
| `subdomain` (e.g. per-locus HLA) | `token_metadata.csv` | `token_id`, the `subdomain_column` (default `locus`) |
| `token_value_column` (e.g. 1-field HLA) | `token_metadata.csv` | `token_id`, the named column |
| SHAP scripts, apps (diseases) | `token_metadata.tsv` | `token_id`, `name`, `icd_code`, `icd_description`, `n_subjects` |

Scripts that create these files from `tokenizer.yaml`:

```bash
# HLA: locus, locus_group, gene, drb345_group, allele_1field, allele_2field
# Writes token_metadata.csv and a .tsv copy
python data/generate_hla_metadata.py \
    --tokenizer_path data/transforms/tokens/hla_alleles/tokenizer.yaml

# Diseases: icd_code, icd_description, n_subjects (needs tokens.csv too)
python data/generate_disease_metadata.py \
    --tokenizer_path data/transforms/tokens/diseases/tokenizer.yaml \
    --tokens_path    data/transforms/tokens/diseases/tokens.csv
```

## 5. Pretrained embeddings (optional)

For `projector: pretrained`, you need a `.pt` file with shape `[vocab_size, d_ext]`. Row *i* is the embedding of `token_id` *i* in `tokenizer.yaml`. For HLA, this script creates one file per locus from ESM2 protein embeddings:

```bash
python data/generate_hla_pretrained_embeddings.py \
    --esm_cache ~/repos/hla_genes/cache/hla_protseq_embeddings_esm2_t33_650M_UR50D.pkl \
    --hla_genes_repo ~/repos/hla_genes \
    --output_dir embeddings/hla/esm2_t33_650M_UR50D \
    --n_components 0.995
```

For the config fields, see [domain_config.md](domain_config.md#pretrained-token-embeddings-projector-pretrained).

## 6. Subject lists (cross-validation folds)

`subject_lists/` must contain 10 files, `subset1of10.csv` … `subset10of10.csv`. Each file has one subject ID per line and no header:

```
1000015
1000027
```

Each subject must be in exactly one file. For `--test_fold k`, `utils/cv_utils.py:get_data_partitions` uses 7 folds for training, 1 for validation and 2 for test, rotating with *k*.

To make random folds:

```python
import numpy as np, pandas as pd
ids = pd.read_csv("data/transforms/tokens/diseases/tokens.csv").subject_id.unique()
rng = np.random.default_rng(0)
for i, chunk in enumerate(np.array_split(rng.permutation(ids), 10), start=1):
    pd.Series(chunk).to_csv(f"data/transforms/subject_lists/subset{i}of10.csv",
                            index=False, header=False)
```

Subject filters applied at training time:
- A subject must have at least one token in the `diseases` domain. Other subjects are removed.
- `--subjects FILE` keeps only the subjects in `FILE` (one ID per line), e.g. `subject_lists/genetic_white_ids.txt`.
- A subject can be absent from any other domain. That domain then has no tokens for that subject.

## 7. Register the domain

Add an entry to a domain config YAML. For example, in a new file that extends the default config:

```yaml
# config/my_experiment.yaml
extends: domain_config_default.yaml

my_biomarker:
  projector: embed
  path: my_biomarker        # → data/transforms/tokens/my_biomarker/
  predict: false
```

Then activate it with `--domains`:

```bash
python train.py --domain_config_yaml config/my_experiment.yaml \
    --domains diseases,death,lifestyle,sex,my_biomarker ...
```

For all the config fields, see [domain_config.md](domain_config.md).

## 8. Check the result

Use `--dryrun`. It loads the data and prints the subject and token counts per split and per domain, and it does not train:

```bash
python train.py --dryrun \
    --domain_config_yaml config/my_experiment.yaml \
    --domains diseases,death,lifestyle,sex,my_biomarker \
    --attention_scheme "all:causal(mask_ties=True)" \
    --experiment_name dry --test_fold 1
```

## Example: a new categorical domain from a long table

```python
from pathlib import Path
import pandas as pd
import yaml

# raw: one row per (subject, event) with columns eid, age_days, code
raw = pd.read_csv("my_events.csv")

vocab = sorted(raw.code.unique())
out = Path("data/transforms/tokens/my_biomarker")
out.mkdir(parents=True, exist_ok=True)

yaml.safe_dump(vocab, open(out / "tokenizer.yaml", "w"))

code_to_id = {c: i for i, c in enumerate(vocab)}
pd.DataFrame({
    "subject_id": raw.eid.astype(int),
    "age":        raw.age_days.round().astype(int),
    "token_id":   raw.code.map(code_to_id),
}).to_csv(out / "tokens.csv", index=False)
```

## Origin of the existing domains

- `sex`, `lifestyle`, `diseases`, `death`: split out of the legacy Delphi `.bin` files by `data/prepare_tokens_from_previous_scheme.py`. The global token ranges are hard-coded in that script, and it writes to `test/data/transforms/<domain>/`. It writes `tokens.csv` only, not `tokenizer.yaml`.
- `hla_alleles`, `genetic_pcs`, `cv_drugs`, `ns_drugs`, `rare_variants`, `ldl`: the scripts that made them are not in this repository.
- Synthetic cohorts: `generate_synthetic_cohort.py` writes a complete `--data_root` (`tokens/`, `subject_lists/`) in this same layout.
