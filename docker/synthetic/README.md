# Delphi · synthetic-data training image

Train Delphi — a generative transformer for health trajectories — on a
**synthetic cohort** of 485,631 subjects, without access to UK Biobank.

The image contains the training code, a pinned Python environment, and the synthetic cohort.
Run one command to train on GPU, on CPU, or (natively) on Apple Silicon.

| | |
|---|---|
| **Image** | `delphi-synthetic:latest` |
| **Native bundle** | code + data, no environment (66 MB) |
| **Domains** | `sex`, `lifestyle`, `diseases`, `death` |
| **Stack** | Python 3.11 · PyTorch 2.8.0 (CUDA 12.8) · MLflow 2.22.0 |

---

## Contents

- [Quick start](#quick-start)
- [Run options](#run-options)
- [Run on Apple Silicon (MPS)](#run-on-apple-silicon-mps)
- [Run on an HPC cluster (Singularity)](#run-on-an-hpc-cluster-singularity)
- [The synthetic cohort](#the-synthetic-cohort)
- [Privacy](#privacy)
- [Build the image](#build-the-image)
- [Limitations](#limitations)

---

## Quick start

**Requirements:** Docker. For GPU training, also an NVIDIA driver (version 570 or newer, for
CUDA 12.8) and
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

This README is also inside the image (`/opt/delphi/README.synthetic.md`).

**1. Load the image.**

```bash
sha256sum -c delphi-synthetic_2026-11-13.tar.gz.sha256
docker load -i delphi-synthetic_2026-11-13.tar.gz
```

**2. Check that the container sees the GPU.**

```bash
docker run --rm --gpus all --entrypoint python delphi-synthetic:latest \
    -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

**3. Run a smoke test** (2,000 subjects, 1 epoch, small model — about one minute).

```bash
mkdir -p mlruns
docker run --rm --gpus all --shm-size=2g \
    --user $(id -u):$(id -g) \
    -v "$PWD/mlruns:/workspace/mlruns" \
    delphi-synthetic:latest \
    --n_subjects_debug 2000 --max_epochs 1 --n_layer 2 --n_head 2 --n_embd 32 --block_size 128
```

**4. Train on the full cohort.**

The first minutes go to building the dataset cache and compiling the model; then the progress
bar starts.

```bash
docker run --rm --gpus all --shm-size=8g \
    --user $(id -u):$(id -g) \
    -v "$PWD/mlruns:/workspace/mlruns" \
    delphi-synthetic:latest --max_epochs 100 --experiment_name synthetic-delphi
```

**5. Look at the results.**

```bash
mlflow ui --backend-store-uri ./mlruns     # MLflow 2.22 recommended
```

Each run stores its metrics, parameters and checkpoints (`artifacts/checkpoints/best_model.pt`).

---

## Run options

### Docker flags

| Flag | Why |
|---|---|
| `--gpus all` | Gives the container access to the GPUs. Leave it out to train on CPU. |
| `--shm-size=2g` (or more) | With `--num_workers` > 0, the DataLoader workers share memory through `/dev/shm`. Docker's default (64 MB) is too small. |
| `--user $(id -u):$(id -g)` | Writes `mlruns/` as your user, not as root. The image sets `USER=delphi` and `HOME=/tmp`, which `torch.compile` needs when the user ID has no account in the image. |
| `-v "$PWD/mlruns:/workspace/mlruns"` | Keeps the MLflow runs on the host after the container exits. |
| `-e DEVICE=cpu` | Forces the device. Default: `cuda` if available, else `mps`, else `cpu`. |

### Training arguments

Everything after the image name goes to `train_synthetic.py`. `--tokens_dir` and `--no_rich`
are already set. Show all arguments with `docker run --rm delphi-synthetic:latest --help`.

| Argument | Default | Description |
|---|---|---|
| `--max_epochs` / `--patience` | 1000 / 20 | Training length and early stopping |
| `--n_layer` / `--n_head` / `--n_embd` | 12 / 6 / 120 | Model size |
| `--block_size` | `auto` | `auto`: each batch is trimmed to its longest subject. An integer fixes the length. |
| `--trim_multiple` | 32 | With `auto`, round each batch length up to this multiple (fewer `torch.compile` recompilations) |
| `--no_event_token_rate` | 5 | Years between "no event" tokens (same as the generating model) |
| `--batch_size` | 64 | Training batch size |
| `--num_workers` | 4 | DataLoader worker processes, started once for the whole run. 0 loads the data in the main process (about 2× slower per epoch on GPU). On macOS the workers take about 30 s to start. |
| `--lr` | 1e-4 | Peak learning rate (cosine schedule) |
| `--n_subjects_debug` | all | Use only N subjects (for quick tests) |
| `--val_frac` / `--test_frac` | 0.1 / 0.1 | Random split of the synthetic subjects |
| `--use_amp` | off | bfloat16 mixed precision (CUDA only) |
| `--no-compile` | off | Skip `torch.compile` (faster start, slower training) |
| `--experiment_name` / `--run_name` | `synthetic-delphi` | MLflow names |

### Sequence length and memory

With the defaults, most batches have 96 tokens and some have 64: the longest subject of a batch
has about 56 real tokens, plus about 16 "no event" tokens per subject. The longest subject of
the whole cohort has 106 real tokens.

With the defaults, training on the full cohort uses about 2 GB of GPU memory. On a laptop
RTX 2000 Ada (8 GB) one epoch takes about 7 minutes. A fixed `--block_size 512` uses much more
memory for no gain, and does not fit on an 8 GB GPU.

### Data split

`train_synthetic.py` makes a random train / validation / test split of the synthetic subjects
(80 / 10 / 10 %, set by `--seed`). The split is not related to the cross-validation folds of the
real cohort: the image has no fold information. This is fine for training and evaluation on
synthetic data.

### CPU

Leave out `--gpus all`. Limit the cores with `--cpus=8 -e OMP_NUM_THREADS=8`.
CPU training is slow: use it for tests with `--n_subjects_debug`, not for the full cohort.

---

## Run on Apple Silicon (MPS)

Docker on macOS cannot use the Mac GPU. Use the native bundle instead: it has the same code
and the same data as the image.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # install uv, if needed

mkdir delphi-synthetic && cd delphi-synthetic
tar -xzf ../delphi-synthetic-bundle_2026-11-13.tar.gz
uv sync --frozen --no-dev --extra log --no-install-project

uv run python train_synthetic.py --tokens_dir synthetic_data --no_rich --max_epochs 100
```

- The script selects MPS automatically and disables `torch.compile` (not tested on MPS).
- With PyTorch 2.8, `torch.isin` runs on MPS, and the DataLoader workers work on macOS: they
  take about 30 s to start, then each epoch is about 2× faster than with `--num_workers 0`.
- If an operation is not implemented on MPS, prefix the command with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` to run that operation on CPU.

> [!NOTE]
> The MPS path is not tested yet. Please report errors.

---

## Run on an HPC cluster (Singularity)

```bash
singularity build delphi-synthetic.sif docker-archive://delphi-synthetic_2026-11-13.tar.gz
singularity run --nv --bind "$PWD/mlruns:/workspace/mlruns" delphi-synthetic.sif --max_epochs 100
```

If `docker-archive` does not accept the `.gz` file, decompress it first with `gunzip -k`.

> [!NOTE]
> The Singularity path is not tested yet.

---

## The synthetic cohort

| Domain | Rows | Subjects | Source |
|---|---:|---:|---|
| `sex` | 485,631 | 485,631 | Copied from the conditioning subject |
| `lifestyle` | 1,451,273 | 485,282 | Copied, ages shifted (see [Privacy](#privacy)) |
| `diseases` | 11,193,220 | 485,543 | **Generated** by Delphi (1,256 ICD-10 codes) |
| `death` | 108,128 | 108,128 | **Generated** by Delphi |

All files use the standard Delphi layout: `tokens.csv` (`subject_id, age, token_id`, age in
days) and `tokenizer.yaml`. The vocabularies are identical to the real data, so a model trained
here can be evaluated on real data later.

**How it was made.** A Delphi model trained on UK Biobank (MLflow run `2efe8d45…`, fold 1)
generated one synthetic "twin" per participant with `generate_synthetic_cohort.py`. Each twin
starts from the participant's sex, genetics and lifestyle. The model then samples diseases,
death, and their ages, up to age 85. The real follow-up window is **not** used
(`--no_censor_at_real_cutoff`), so the trajectory length does not reveal the birth date.

| Property | Value |
|---|---|
| Age at last event (median) | 83.5 years |
| Subjects with a death event | 22.3 % |
| Age at death (median) | 79.3 years |

---

## Privacy

The image contains **no identifiers**: no UK Biobank IDs, no birth dates, no genetic data
(HLA alleles, genetic PCs), no mapping to real subjects. Subject IDs are new and randomly
assigned (`9000000`–`9485630`).

Before the build, `anonymize_synthetic.py` processes the cohort:

1. **Lifestyle ages** get one random shift per visit, drawn from N(0, 2 years). The shift is
   the same for all tokens of a visit. The seed is random and not stored.
2. **Subject IDs** are remapped by a random permutation.
3. **Generation metadata** (which lists the real conditioning IDs) is left out.

### Checks

| Check | Result |
|---|---|
| Diagnoses of a twin also present in its real participant | 8.6 % (random same-sex pair: 8.2 %) |
| Same diagnosis on the same day as the real participant | 0.001 % (random pair: 0.001 %) |
| Correlation of last event age with real age on 2020-01-01 | 0.07 |
| Lifestyle visits whose exact (age, tokens) exist in the real data | 85.8 % (real vs. other real: 89.2 %) |

The model does not reproduce the trajectories of the participants it was trained on.

### What remains

Each synthetic subject still corresponds to one real participant through the real **sex** and
the real **lifestyle categories** (with shifted ages). This is weak information, but it is
still derived from individual UK Biobank data.

> [!IMPORTANT]
> Check your UK Biobank access agreement before you share this image outside the approved
> project.

---

## Build the image

The build stages a **whitelist** of files (training code, configs, lock file, and the four
domains of the cohort). Real data never enters the build context.

```bash
SEED=<secret integer> bash docker/synthetic/build.sh <cohort>/tokens delphi-synthetic:latest
```

| Variable | Default | Description |
|---|---|---|
| `SEED` | required | Seed for the lifestyle age shift. Keep it private. |
| `SIGMA_YEARS` | `2.0` | Standard deviation of the shift, in years |
| `STAGE_DIR` | — | Stage the context in this folder and skip `docker build` |

The image gets the labels `delphi.git_commit` and `delphi.lifestyle_age_sigma_years`:

```bash
docker inspect delphi-synthetic:latest --format '{{json .Config.Labels}}'
```

Save it to a file:

```bash
docker save delphi-synthetic:latest | pigz > delphi-synthetic_$(date +%F).tar.gz
```

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Image definition (uv, pinned lock, gcc/g++ for `torch.compile`) |
| `build.sh` | Stages the whitelisted context and builds the image |
| `anonymize_synthetic.py` | Lifestyle age shift, ID remapping, metadata removal |

---

## Limitations

- **No real-data evaluation.** `--compute_real_aucs` needs real UK Biobank data, which the
  image does not contain.
- **x86-64 only.** The image does not run natively on ARM machines (Apple Silicon, NVIDIA GB10).
- **Untested paths:** MPS and Singularity.
- **Model bias.** The synthetic trajectories inherit the calibration of the generating model.
  Use the cohort for methods development, not for epidemiological estimates.
