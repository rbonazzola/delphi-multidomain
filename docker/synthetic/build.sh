#!/bin/bash
# Build the Docker image for training Delphi on synthetic data.
#
# The build context is a staging directory with a whitelist of files: the code
# needed by train_synthetic.py and the anonymized synthetic cohort. Real data
# under data/transforms/tokens/ never enters the context.
#
# Usage:
#   SEED=<secret int> bash docker/synthetic/build.sh <synthetic_dir> [image_tag]
#
#   <synthetic_dir>  output of generate_synthetic_subjects.py
#                    (sex/, lifestyle/, diseases/, death/)
#   SIGMA_YEARS      SD of the lifestyle age shift, in years (default: 2.0)
#   SEED             RNG seed for the shift; keep it private (required)
#   STAGE_DIR        if set, stage the bundle there and skip docker build
#                    (use this for native training, e.g. on a Mac with MPS)
#
# Run:
#   docker run --gpus all -v "$PWD/mlruns:/workspace/mlruns" delphi-synthetic:latest \
#       --max_epochs 100 --experiment_name synthetic-delphi
#
# Native run on a Mac (Apple Silicon, MPS). Docker on macOS has no GPU access.
#   SEED=<int> STAGE_DIR=delphi_synthetic_bundle bash docker/synthetic/build.sh <synthetic_dir>
#   # copy delphi_synthetic_bundle/ to the Mac, then, inside it:
#   uv sync --frozen --no-dev --extra log --no-install-project
#   uv run python code/train_synthetic.py --tokens_dir synthetic_data --no_rich --max_epochs 100
#   (the device is picked automatically: cuda > mps > cpu; force it with DEVICE=mps)
set -euo pipefail

SYNTH_DIR=${1:?usage: SEED=<int> build.sh <synthetic_dir> [image_tag]}
TAG=${2:-delphi-synthetic:latest}
SIGMA_YEARS=${SIGMA_YEARS:-2.0}
: "${SEED:?set SEED (keep it private)}"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -n "${STAGE_DIR:-}" ]; then
    STAGE=$STAGE_DIR; mkdir -p "$STAGE"
else
    STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
fi

# ── 1. code (whitelist) ─────────────────────────────────────────────────────
mkdir -p "$STAGE/code"/{data,auc,config}
cp "$REPO"/config/*.yaml "$STAGE/code/config/"
cp "$REPO"/{pyproject.toml,uv.lock,README.md,LICENSE} "$STAGE/"
cp "$REPO"/{train.py,train_synthetic.py} "$STAGE/code/"
cp "$REPO/docker/synthetic/README.md" "$STAGE/code/README.synthetic.md"
cp "$REPO/data/dataset.py" "$STAGE/code/data/"
cp "$REPO/auc/aucs.py" "$STAGE/code/auc/"
rsync -a --exclude='__pycache__/' --exclude='legacy/' --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs \
    "$REPO/delphi" "$REPO/utils" "$STAGE/code/"

# ── 2. synthetic data (anonymized) ──────────────────────────────────────────
python "$REPO/docker/synthetic/anonymize_synthetic.py" \
    --input_dir "$SYNTH_DIR" --output_dir "$STAGE/synthetic_data" \
    --sigma_years "$SIGMA_YEARS" --seed "$SEED"

# ── 3. build ────────────────────────────────────────────────────────────────
echo "Build context:"; (cd "$STAGE" && find . -type f | sort)
[ -n "${STAGE_DIR:-}" ] && { echo "Staged in $STAGE (no docker build)"; exit 0; }
docker build -f "$REPO/docker/synthetic/Dockerfile" -t "$TAG" \
    --label delphi.git_commit="$(git -C "$REPO" rev-parse HEAD)" \
    --label delphi.lifestyle_age_sigma_years="$SIGMA_YEARS" \
    "$STAGE"
echo "Built $TAG"
