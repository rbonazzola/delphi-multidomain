> ⚠️ Stability notice
>
> The codebase is evolving and interfaces are not yet stable. Changes may affect the CLI, model architecture, configuration schema, default parameter values and output formats.

# Multi-domain Delphi

This repository extends the **Delphi** core codebase to support **multi-domain longitudinal data**, beyond standard diagnosis codes.

It focuses on:
- Adding heterogeneous domains (e.g. diseases, drugs, lifestyle, HLA alleles).
- Defining **custom attention policies** within and across domains.
- Scaling experiments via **Slurm job arrays** while tracking models with MLflow.

## Table of contents

- [Multi-domain Delphi](#multi-domain-delphi)
  - [Table of contents](#table-of-contents)
  - [Software environment](#software-environment)
  - [Training](#training)
    - [Preparing the data for each domain](#preparing-the-data-for-each-domain)
    - [Specifying the attention scheme](#specifying-the-attention-scheme)
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
- `torch=2.8.0`

## Training

### Preparing the data for each domain

Each domain is a directory with a `tokenizer.yaml` and a `tokens.csv`; the CV folds are 10 subject-list files. To create these files, see [docs/data_preparation.md](docs/data_preparation.md).

Domains are configured in a YAML file (reference: `config/domain_config_default.yaml`). Give the file with `--domain_config_yaml` and select the active domains with `--domains`:

```yaml
diseases:
  projector: embed
  path: diseases      # relative to <data_root>/tokens/
  predict: true
  no_repeat: true

genetic_pcs:
  type: continuous
  projector: linear
  path: genetic_pcs
  at_birth: true
  input_size: 40
  n_latent_tokens: 5
```

```bash
python train.py --domain_config_yaml config/domain_config_default.yaml --domains diseases,lifestyle,sex ...
python train.py --dcfg diseases.predict=True lifestyle.age_jitter=False ...   # override fields from the CLI
```

`padding` and `no_event` are added automatically: do not list them in `--domains`.

Full reference (all fields, `group`, `extends`, `parent`/`abstract` hierarchy, `--dcfg`, domain dropout, pretrained embeddings, token collapsing): [docs/domain_config.md](docs/domain_config.md).

### Specifying the attention scheme

`--attention_scheme` sets which tokens each token can attend to. It takes a scheme string or a preset name from `config/attention_schemes.yaml`:

```bash
python train.py --attention_scheme "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)" ...
python train.py --attention_scheme hla_bidirectional ...   # same scheme, as a preset
```

Key points:
- Each rule `<domains>:<type>` adds allowed pairs; the final mask is the union of all rules.
- `causal` compares **token ages**, not domain order. `mask_ties=True` blocks same-age pairs (so at-birth tokens do not see each other unless a `bidirectional` rule connects them).
- Use `all` in the causal rule. An explicit domain list does not include the `no_event` domain, so no-event tokens become isolated.

Full reference (syntax, aliases, worked example with attention matrices, presets, per-layer schemes): [docs/attention_schemes.md](docs/attention_schemes.md).

### Exemplar command

```bash
python train.py \
  --domains diseases,death,lifestyle,sex \
  --attention_scheme "all:causal(mask_ties=True)" \
  --arch h6d240l12 \
  --experiment_name example \
  --run_name fold1 \
  --batch_size_schedule "10:32,10:64,*:256x4" \
  --eval_batch_size 512
```

Note: `padding` and `no_event` are added automatically and do not need to be listed in `--domains`.

`--arch hNdDlL` is a shorthand for `--n_head N --n_embd D --n_layer L` (e.g. `--arch h12d240l12`). Letter order and case do not matter (`L12H12D240` is the same); each letter must appear once. You can still pass the three flags separately; a flag that disagrees with `--arch` is an error. The resolved string is logged as the MLflow param `arch` and can be used as `{arch}` in `--run_name`. Defaults: `h6d120l12`.

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
  --attention_scheme base \
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
