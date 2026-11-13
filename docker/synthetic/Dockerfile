# Delphi training on synthetic data. Build with docker/synthetic/build.sh,
# which stages a whitelisted build context (code + anonymized synthetic data only).
FROM ghcr.io/astral-sh/uv:0.9.5-python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=/workspace/mlruns \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /opt/delphi

# Dependencies first (cached layer). torch comes from the cu128 index on linux.
# The uv cache lives in a BuildKit cache mount, so it does not end up in the image.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra log --no-install-project
ENV PATH="/opt/delphi/.venv/bin:$PATH"

# torch.compile compiles code at run time: a C launcher for Triton on GPU (gcc),
# and C++ kernels on CPU (g++).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY code/ ./
COPY synthetic_data/ /opt/delphi/synthetic_data/

WORKDIR /workspace
ENTRYPOINT ["python", "/opt/delphi/train_synthetic.py", "--tokens_dir", "/opt/delphi/synthetic_data", "--no_rich"]
CMD ["--max_epochs", "100"]
