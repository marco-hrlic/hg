#!/usr/bin/env bash
# Deploy this checkout to the DGX Spark and run a command there.
#
# The Spark is an NVIDIA GB10 (aarch64) behind the `Notch` VPN. It is reached by
# ssh + uv rather than through docker/Dockerfile: the container path in that
# Dockerfile has not been exercised on the GB10, and `pyproject.toml`'s
# `torch>=2.4` carries no aarch64/cu130 marker, so a bare `uv pip install
# -e ".[gpu]"` there resolves against default PyPI wheels. The CUDA-13 aarch64
# index is pinned explicitly below, matching what the sibling research projects
# do.
#
# Usage:
#   scripts/spark_deploy.sh                    # sync only
#   scripts/spark_deploy.sh setup              # sync, then create/refresh the venv
#   scripts/spark_deploy.sh run '<command>'    # sync, then run a command remotely
#
# Environment:
#   SPARK_HOST   default notch@10.0.4.10
#   SPARK_DIR    default ~/marco/hg
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-notch@10.0.4.10}"
SPARK_DIR="${SPARK_DIR:-\$HOME/marco/hg}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

check_vpn() {
  if ! ping -c 1 -W 3 "${SPARK_HOST#*@}" >/dev/null 2>&1; then
    echo "cannot reach ${SPARK_HOST#*@}." >&2
    echo "The VPN drops intermittently; bring it up with:" >&2
    echo "  nmcli connection up Notch" >&2
    exit 1
  fi
}

sync_repo() {
  # --delete keeps the remote tree an exact mirror, which also means scratch
  # files written directly on the Spark are removed. Keep throwaway scripts
  # outside the repo tree, or they will vanish on the next sync.
  # '/data' and '/results' are anchored to the repo root: an unanchored
  # 'data' pattern also matches src/hg_native/data (the episode-schema and
  # generator package), silently dropping it from every sync.
  rsync -az --delete \
    --exclude '.venv' --exclude '.git' \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    --exclude '/data' --exclude '/results' \
    "${REPO_ROOT}/" "${SPARK_HOST}:${SPARK_DIR}/"
  echo "synced ${REPO_ROOT} -> ${SPARK_HOST}:${SPARK_DIR}"
}

remote() {
  # shellcheck disable=SC2029
  ssh -o BatchMode=yes "${SPARK_HOST}" "cd ${SPARK_DIR} && $1"
}

setup_venv() {
  remote "export PATH=\$HOME/.local/bin:\$PATH && \
    uv venv --python 3.12 && \
    uv pip install --index-strategy unsafe-best-match \
      --extra-index-url ${TORCH_INDEX} \
      'torch>=2.9' transformers accelerate numpy pyyaml pytest matplotlib && \
    .venv/bin/python -c 'import torch, transformers; \
      print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available()); \
      print(\"transformers\", transformers.__version__)'"
}

check_vpn
sync_repo

case "${1:-sync}" in
  sync)  ;;
  setup) setup_venv ;;
  run)
    if [ $# -lt 2 ]; then echo "run needs a command" >&2; exit 2; fi
    # HF_HOME is set explicitly so a long download is not repeated under a
    # different cache root by a differently-configured shell.
    remote "HF_HOME=\$HOME/.cache/huggingface .venv/bin/python $2"
    ;;
  *) echo "unknown mode: $1" >&2; exit 2 ;;
esac
