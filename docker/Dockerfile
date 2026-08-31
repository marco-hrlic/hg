# Container image for the DGX Spark (GB10, arm64 + CUDA).
# Base tag is pinned per NGC release; override at build time:
#   docker build -f docker/Dockerfile --build-arg BASE=nvcr.io/nvidia/pytorch:<tag> -t hg:m2 .
ARG BASE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.hf \
    TOKENIZERS_PARALLELISM=false \
    # Determinism for the primary dependence analysis (protocol section 13).
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace

COPY requirements.txt requirements-gpu.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements-gpu.txt && pip install --no-cache-dir -e ".[dev,analysis]"

COPY . .

# Inference, fine-tuning, dataset construction and analysis all enter through
# the same CLI so a run is reproducible from a config file alone.
ENTRYPOINT ["hg"]
CMD ["--help"]
