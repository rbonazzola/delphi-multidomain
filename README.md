> ⚠️ Stability notice
>
> The codebase is evolving and interfaces are not yet stable. Changes may affect the CLI, model architecture, configuration schema, default parameter values and output formats.

# Multi-domain Delphi

This repository extends the **Delphi** core codebase to support **multi-domain longitudinal data**, beyond standard diagnosis codes.

It focuses on:
- Adding heterogeneous domains (e.g. diseases, drugs, lifestyle, HLA alleles, rare variants).
- Defining **custom attention policies** within and across domains.
- Scaling experiments via **Slurm job arrays** while tracking models with MLflow.

## Table of contents

- [Multi-domain Delphi](#multi-domain-delphi)
  - [Table of contents](#table-of-contents)
  - [Software environment](#software-environment)
  - [Training](#training)
    - [Preparing the data for each domain](#preparing-the-data-for-each-domain)
    - [Specifying the attention scheme](#specifying-the-attention-scheme)
      - [Example 1: Fully causal attention (with tie-masking, i.e. no same-time attention)](#example-1-fully-causal-attention-with-tie-masking-ie-no-same-time-attention)
      - [Example 2 — Bidirectional HLA block + causal domains](#example-2--bidirectional-hla-block--causal-domains)
    - [Exemplar command](#exemplar-command)
    - [Batch size schedule (`--batch_size_schedule`)](#batch-size-schedule---batch_size_schedule)
    - [Dry run (`--dryrun` / `--dry`)](#dry-run---dryrun----dry)
    - [Resuming a run (`--resume_run_id` / `--interactive`)](#resuming-a-run---resume_run_id----interactive)
    - [Mixed precision (`--use_amp`)](#mixed-precision---use_amp)
    - [AUC computation (`--compute_aucs` / `--auc`)](#auc-computation---compute_aucs----auc)
    - [Model tracking with MLflow](#model-tracking-with-mlflow)
  - [Model explainability](#model-explainability)
  - [Submitting a hyperparameter search as Slurm job array](#submitting-a-hyperparameter-search-as-slurm-job-array)
  - [Tips for querying MLflow runs](#tips-for-querying-mlflow-runs)
  - [Notes for developers](#notes-for-developers)

## Software environment

The code has been tested with the following versions:
- `numpy=1.26.4`
- `pandas=2.2.3`
- `mlflow=2.22.0`
- `torch=2.3.0`

## Training

### Preparing the data for each domain

Domains are configured via a YAML file (see `config/domain_config_default.yaml` for a reference).
Each entry defines one domain and its properties:

```yaml
diseases:
  projector: embed
  path: diseases
  predict: true
  no_repeat: true   # each disease is recorded only at first diagnosis

lifestyle:
  projector: embed
  path: lifestyle
  age_jitter: true

sex:
  projector: embed
  path: sex
  at_birth: true

genetic_pcs:
  type: continuous
  projector: linear
  path: genetic_pcs
  at_birth: true
  input_size: 40
  n_latent_tokens: 5
```

**`no_repeat: true`** should be set for domains where each token appears at most once per subject by construction (e.g. diseases recorded only at first diagnosis). When enabled, the logits for already-seen tokens are set to `-inf` after each forward pass, before the loss is computed. This prevents the model from learning the spurious pattern of suppressing a disease's logit once it has appeared - an artefact of the data, not a biological signal.

Pass the config file via `--domain_config_yaml` and select which domains to activate with `--domains`:

```bash
python train.py --domain_config_yaml config/domain_config_default.yaml --domains diseases,lifestyle,sex ...
```

The `padding` domain is injected automatically — do not add it to the YAML or `--domains`.

#### Domain groups (`group`)

Each domain can declare a `group` name used for compact display in logs and run summaries. When multiple domains share a group, they appear as a single token in reports instead of listing each domain individually.

`config/domain_config_default.yaml` sets the following defaults:

| Group | Domains |
|-------|---------|
| `core` | diseases, death, lifestyle, sex |
| `drugs` | cv_drugs, ns_drugs |
| `hla` | hla_alleles |
| *(none)* | rare_variants, genetic_pcs — shown by their own name |

You can override or add groups via `--dcfg` or in a child config that extends the default.

#### Extending a config (`extends`)

A domain config YAML can inherit from another file using `extends:`:

```yaml
# config/my_experiment.yaml
extends: domain_config_default.yaml

diseases:
  predict: false        # override one field; all other fields are inherited

hla_a:
  parent: hla_alleles   # within-file inheritance still works
  subdomain: hla_a
  predict: true
  group: hla            # group is not inherited via parent; set it explicitly
```

The child file is deep-merged on top of the base: within each domain, only the fields declared in the child override those from the base; unmentioned fields are kept as-is. Chains are supported (A extends B extends C). Paths in `extends` are resolved relative to the file that declares them.

#### Overriding domain config fields from the CLI

Individual domain config fields can be overridden at runtime without editing the YAML, using `--dcfg` (alias for `--domain_config`):

```bash
python train.py --dcfg diseases.predict=True hla_alleles.dropout_rate=0.2 lifestyle.age_jitter=False ...
```

Multiple overrides can be passed in a single `--dcfg` invocation (space-separated) or across multiple invocations. The format is always `DOMAIN.FIELD=VALUE`. Values are parsed as Python literals (booleans, ints, floats, strings).

Duration fields (`age_jitter_min`, `age_jitter_max`) accept human-readable strings:

```bash
python train.py --dcfg lifestyle.age_jitter_min=-20y lifestyle.age_jitter_max=10y ...
# Accepted units: y (years), m (months), d (days)
```

For each domain, create a folder under `data/transforms/tokens/<domain_name>/` with two files:
- `tokens.csv`: columns `subject_id`, `age` (in days), `token_id` — one row per token event, all subjects together.
- `tokenizer.yaml`: list of token names in order; position determines `token_id` (zero-based).

#### Pretrained token embeddings (`projector: pretrained`)

When external token embeddings are available (e.g. ESM2 protein embeddings for HLA alleles), the domain can be configured to use them instead of learning embeddings from scratch:

```yaml
hla_a:
  projector: pretrained
  pretrained_path: /path/to/hla_a_embeddings.pt
  freeze: false          # true = keep pretrained weights fixed; false = fine-tune
  path: hla_alleles      # still needed for tokenizer.yaml and tokens.csv
  at_birth: true
  subdomain: hla_a
  subdomain_column: locus
```

**How it works:** the `.pt` file must be a `torch.Tensor` of shape `[vocab_size, d_ext]`, where `vocab_size` matches the number of entries in `tokenizer.yaml` and `d_ext` is the external embedding dimension. At runtime a learned linear layer projects each `d_ext`-dimensional vector to `n_embd`, so the external dimension does not need to match the model's embedding size.

The domain still uses the shared `tokenizer.yaml` and `tokens.csv` under `path`. The pretrained weights replace the random initialization of the embedding table; the linear projection on top is always trained from scratch.

**`freeze: true`** locks the pretrained embedding weights and only trains the projection layer. **`freeze: false`** (default) fine-tunes both. Fine-tuning is usually beneficial when the model is large enough and the downstream task differs from the pretraining objective.

**Combining with subdomain hierarchy:** a common pattern is to define an abstract parent with the shared `path` / `at_birth` / `subdomain_column` settings, then override `projector` and `pretrained_path` in each child:

```yaml
hla_alleles:
  abstract: true
  projector: embed          # default for children that don't override it
  path: hla_alleles
  at_birth: true
  subdomain_column: locus

hla_a:
  parent: hla_alleles       # inherits path, at_birth, subdomain_column
  subdomain: hla_a
  projector: pretrained     # overrides the parent's projector
  pretrained_path: /path/to/hla_a_esm2.pt
  freeze: false
```

Children that do not override `projector` fall back to the parent's value (`embed` in the example above), so you can mix pretrained and learned embeddings across loci in the same config.

#### Token collapsing (`token_value_column`)

When a metadata column maps multiple fine-grained tokens to the same coarser group, `token_value_column` collapses them into a single token without splitting into child domains. This is an alternative to the subdomain hierarchy when you want a single domain with reduced granularity.

```yaml
hla_alleles:
  projector: embed
  path: hla_alleles
  at_birth: true
  token_value_column: allele_1field   # collapses 2-field alleles to 1-field groups
```

**How it works:** at load time, `TokenDomain` reads `token_metadata.csv`, groups all `token_id`s by their value in `token_value_column`, and assigns a new contiguous ID to each unique group value. The resulting vocabulary contains one entry per unique value in that column (e.g. `HLA-A*01`, `HLA-A*02`, …) instead of one per original token. Multiple original tokens that share the same group value are treated as the same token during training.

`token_value_column` and `subdomain` are mutually exclusive — use one or the other.

### Specifying the attention scheme

The attention scheme within and across domains is specified via `--attention_scheme`. You can either pass a scheme string directly or use a named alias defined in `config/attention_schemes.yaml`:

```yaml
# config/attention_schemes.yaml
hla_bidir:
  description: HLA bidirectional with sex, rest causal
  scheme: "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"

hla_causal:
  description: fully causal, no bidirectional HLA
  scheme: "all:causal(mask_ties=True)"
```

```bash
python train.py --attention_scheme hla_bidir ...
```

Add your own aliases to that file to avoid repeating long scheme strings across runs.

#### Example 1: Fully causal attention (with tie-masking, i.e. no same-time attention)

For instance:
`"[sex,diseases,lifestyle,death,hla_alleles,rare_variants]:causal(mask_ties=True)"`

The corresponding attention matrix:

| From \ To    | HLA | sex | diseases | lifestyle | death |
|--------------|-----|-----|----------|-----------|-------|
| **HLA**      | ·   | ·   | ·        | ·         | ·     |
| **sex**      | ←   | ·   | ·        | ·         | ·     |
| **diseases** | ←   | ←   | ←        | ·         | ·     |
| **lifestyle**| ←   | ←   | ←        | ←         | ·     |
| **death**    | ←   | ←   | ←        | ←         | ←     |

In this configuration, all domains follow a strictly causal structure.
Each domain may attend to **past tokens of itself and previous domains**, but never to the future
nor to same-time tokens (`mask_ties=True`).

Two built-in aliases are available for use in any domain group:

| Alias | Expands to |
|-------|-----------|
| `all` | Every domain present in the model (equivalent to listing them all explicitly) |
| `at_birth` | Every domain whose config entry has `at_birth: true` (resolved at model construction time from `config.domains`) |

For example:
```
"all:causal(mask_ties=True)"
"[at_birth]:bidirectional,all:causal(mask_ties=True)"
```

The second scheme gives every at-birth domain (e.g. `sex`, `hla_alleles`, `genetic_pcs`) a bidirectional block and makes all remaining domains causal — without having to enumerate them by name. If you later add or remove a domain from the config, the scheme automatically reflects the change.

#### Example 2 — Bidirectional HLA block + causal domains

On the other hand:
`"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"`

| From \ To    | HLA | sex | diseases | lifestyle | death |
|--------------|-----|-----|----------|-----------|-------|
| **HLA**      | ↔   | ↔   | ·        | ·         | ·     |
| **sex**      | ↔   | ↔   | ·        | ·         | ·     |
| **diseases** | ←   | ←   | ←        | ·         | ·     |
| **lifestyle**| ←   | ←   | ←        | ←         | ·     |
| **death**    | ←   | ←   | ←        | ←         | ←     |

Here, the HLA allele and sex domains form a **bidirectional static block**, allowing mutual
contextualization of at-birth attributes. All downstream domains follow a causal structure,
ensuring temporal consistency while allowing conditioning on static information.

### Exemplar command

```bash
python train.py \
  --domains diseases,death,lifestyle,sex,rare_variants \
  --attention_scheme "[sex,diseases,lifestyle,death,rare_variants]:causal(mask_ties=True)" \
  --n_layer 12 \
  --n_embd 240 \
  --experiment_name rare_variants \
  --run_name_prefix fold1 \
  --batch_size_schedule "10:32,10:64,*:256x4" \
  --eval_batch_size 512
```

Note: `padding` is added automatically and does not need to be listed in `--domains` or `--attention_scheme`.

### Batch size schedule (`--batch_size_schedule`)

Training can use a staged batch size schedule, where different phases use different batch sizes and optional gradient accumulation. This is useful to start training with small batches for stable early learning and then scale up for efficiency.

```bash
--batch_size_schedule "10:32,10:64,*:256x4"
# epochs  0-9  → batch_size=32,  grad_accum=1  (effective batch=32)
# epochs 10-19 → batch_size=64,  grad_accum=1  (effective batch=64)
# epoch  20+   → batch_size=256, grad_accum=4  (effective batch=1024)
```

Format: comma-separated stages, each `n_epochs:batch_size` or `n_epochs:batch_sizexgrad_accum`. Use `*` as `n_epochs` for the final open-ended stage. Overrides `--batch_size`.

Starting with small batches and increasing over time serves two purposes: small batches in early training act as implicit regularization (noisy gradients help escape sharp minima), while larger batches in later stages reduce gradient variance and stabilize refinement once the model is already partially converged.

The `eval_batch_size` argument controls the batch size used for validation and AUC evaluation independently from training (default: 512). This is useful when GPU memory allows larger eval batches than training batches.

```bash
--batch_size 32 --eval_batch_size 512
```

### Dry run (`--dryrun` / `--dry`)

Validates the full configuration — domain config, attention scheme, data loading — and prints a Rich summary, but skips training entirely. Useful for checking that a command is correctly formed before submitting to a cluster:

```bash
python train.py --dry \
  --domains diseases,death,lifestyle,sex \
  --attention_scheme hla_bidir \
  --experiment_name my_experiment
```

### Resuming a run (`--resume_run_id` / `--interactive`)

Training can be resumed from the latest checkpoint of a previous MLflow run:

```bash
python train.py --resume_from_previous --resume_run_id <RUN_ID>
```

The model architecture, domain config, attention scheme and optimizer state are restored from the checkpoint. If `--lr` is also passed, the learning rate is overridden and the scheduler is reset from that value.

Instead of providing the run ID explicitly, you can use `--interactive` (`-i`) to select it from a menu:

```bash
python train.py --resume_from_previous --interactive
```

This presents a numbered list of experiments and (up to 20) recent runs to choose from, and also lets you redirect the resumed run into a different target experiment.

### Mixed precision (`--use_amp`)

Enables automatic mixed precision using **bfloat16**, which reduces memory usage and speeds up training on supported GPUs (Ampere and newer):

```bash
python train.py --use_amp ...
```

bfloat16 has the same exponent range as float32, so gradient scaling is not required. If the GPU does not support bfloat16, the flag has no effect.

### AUC computation (`--compute_aucs` / `--auc`)

If passed, AUCs are computed at the end of training and saved as a CSV file under the `aucs/` subdirectory of the run's MLflow artifact directory. A summary table of the top 50 diseases by case count (weighted mean AUC for ages 40–70) is also printed to the log.

### Model tracking with MLflow

You can specify a custom MLflow location by setting the `MLFLOW_TRACKING_URI` environment variable, otherwise it's the `mlruns` folder within this repo's root directory.

#### Run naming

```bash
python train.py --run_name_prefix fold1 --run_name_suffix _baseline
# results in run name: "fold1_baseline"
```

Both flags are optional and can be used independently. The final run name is their concatenation.

`--run_name` (alias for `--run_name_prefix`) supports `{arg}` placeholders that are replaced at runtime with the value of the corresponding CLI argument:

```bash
python train.py --run_name "fold{test_fold}__bs{batch_size}" --test_fold 3 --batch_size 64
# results in run name: "fold3__bs64"
```

This is especially useful when submitting job arrays with `sarray_params`, where each job has different parameter values.

#### Logging extra params and tags

Arbitrary key-value pairs can be attached to any run:

```bash
python train.py \
  --param setup=baseline note=first_attempt \
  --tag env=cluster status=production
```

`--param` values appear in the MLflow **Parameters** panel and are searchable across runs. `--tag` values appear in **Tags** and are useful for filtering or marking runs (e.g. environment, status, cohort).

#### Reproducibility

At the start of each run, `train.py` logs the current git commit hash and hostname as tags. If the working tree has uncommitted tracked changes, a warning is printed and the full `git diff HEAD` is saved as a `git_diff.patch` artifact, so the exact state of the code can always be recovered.

## Model explainability

_To be completed_

This section will contain details on how to perform SHAP calculation using Nextflow.

## Submitting a hyperparameter search as Slurm job array

The workflow is based on `sarray_params`, a shell function that turns a TSV/CSV parameter table into a Slurm job array. Each row in the table becomes one job; each column header becomes a `--<column>` CLI argument passed to the target script.

### Setup

`sarray_params` is a shell function defined in [`codon_helpers`](https://github.com/rbonazzola/codon_helpers). Clone the repo and source `slurm_functions.sh` before using it:

```bash
source $HOME/repos/codon_helpers/slurm_functions.sh
```

> **Tip:** add the line above to your `~/.bashrc` so it is available in every session.

### Preparing the parameter table

Each column name must match a CLI argument of `train.py` (without the `--` prefix). Each row is one training run. Boolean flags work as follows: `True`/`true` adds the flag, `False`/`false` skips it.

**Option A — Experiment Builder app (recommended)**

```bash
streamlit run apps/experiment_builder.py
```

The app lets you define a grid of hyperparameters visually, preview the expanded parameter table, edit individual cells, and download or save the TSV directly on the server. It also shows the submission command ready to copy.

**Option B — Google Sheets**

Build the table in Google Sheets (one header row, one run per subsequent row), then export via *File → Download → Tab-separated values (.tsv)*. Move the file into `train_scripts/params/`.

### Submitting

```bash
sarray_params train.py train_scripts/params/my_experiment.tsv \
    --gpus=1 --gpu-type=a100 \
    --mem=64G --cpus=8 \
    --time=09:00:00 \
    --max-parallel=8
```

Arguments are split into two groups:

- **TSV columns** — per-job arguments (vary across rows).
- **Extra args after the TSV path** — fixed arguments appended to every job (e.g. `--mem`, `--time`, or any `train.py` flag you want constant across the grid).

Key options:

| Option | Description |
|--------|-------------|
| `--gpus=N` | Number of GPUs per job |
| `--gpu-type=TYPE` | GPU model (e.g. `a100`, `l40s`, `h200`) |
| `--mem=XXG` | Memory per job (default: `64G`) |
| `--cpus=N` | CPU cores per job (default: `8`) |
| `--time=HH:MM:SS` | Wall time per job (default: `09:00:00`) |
| `--max-parallel=N` | Cap concurrent running jobs |
| `--lines=SPEC` | Run only selected rows (1-based; e.g. `3`, `1-5`, `1-3,7,10-12`) |
| `--sep=SEP` | Column separator (auto-detected: `.csv` → `,`, otherwise tab) |
| `--dry-run` | Print the commands that would be submitted, without submitting |

### Dry run

Always preview before submitting:

```bash
sarray_params train.py train_scripts/params/my_experiment.tsv \
    --gpus=1 --gpu-type=a100 --mem=64G --time=09:00:00 --dry-run
```

This prints the exact `python train.py ...` command for each row without calling `sbatch`.

## Tips for querying MLflow runs

```bash
export MLFLOW_TRACKING_URI=$HOME/...

# Examine your available experiments (try to use representative names when you create them)
mlflow experiments search

# Get runs for a given experiment
export EXP_NAME=attention_schemes # an example
export EXP_ID=$(mlflow experiments search | grep -w $EXP_NAME | awk '{print $1}')

# or directly EXP_ID=... 
mlflow runs list --exp-id $EXP_ID
```

## Notes for developers

> **Note on Jupytext usage**
>
> This repository makes extensive use of Jupyter notebooks in `.py` format via **Jupytext**.
> These files can be identified by `# %%` cell separators.
>
> This choice allows the same files to be run both as notebooks and as regular Python scripts, and improves readability and version control compared to `.ipynb` notebooks.
>
> Install:
>
> ```bash
> pip install jupytext
> ```
>
> Convert to `.ipynb`:
>
> ```bash
> jupytext --to ipynb PATH_TO_FILE.py
> ```

_To be completed_

This section will contain:
- Notes on how to extend this codebase.
- Tips on unit tests.
