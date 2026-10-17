#!/usr/bin/env bash
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/../.." && pwd)
UV=${UV:-uv}
BENCH_VENV=${BENCH_VENV:-"$REPO_DIR/.venv"}
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --check ) ]]; then
    echo 'Usage: bash benchmarks/video_generation/setup.sh [--check]' >&2
    exit 2
fi
if [[ ${1:-} != --check ]]; then
    if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
        "$UV" venv --python "${PYTHON_VERSION:-3.12}" "$BENCH_VENV"
    fi
    "$UV" pip sync --python "$BENCH_VENV/bin/python" "$BENCH_DIR/requirements.lock"
    # Optional compatible wheel for hosts where the upstream optional extensions cannot load.
    if [[ -n ${FASTVIDEO_KERNEL_PACKAGE:-} ]]; then
        "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --reinstall "$FASTVIDEO_KERNEL_PACKAGE"
    fi
    "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps -e "$REPO_DIR"
fi
if [[ -n ${BENCH_CUDA_COMPAT:-} ]]; then
    export LD_LIBRARY_PATH="$BENCH_CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
fi
"$UV" pip check --python "$BENCH_VENV/bin/python"
cd "$REPO_DIR"
VLLM_PLUGINS='' PYTHONPATH="$REPO_DIR" "$BENCH_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import vllm_omni
from vllm_omni.entrypoints.omni import Omni
from fastvideo_kernel.block_sparse_attn import block_sparse_attn
print('Checkout:', vllm_omni.__file__)
print('CUDA runtime:', torch.version.cuda)
print({name: metadata.version(name) for name in
       ['torch', 'vllm', 'transformers', 'diffusers', 'fastvideo-kernel', 'triton', 'av']})
PY
mkdir -p "$REPO_DIR/.cache/video-generation-benchmark"
"$UV" pip freeze --python "$BENCH_VENV/bin/python" > "$REPO_DIR/.cache/video-generation-benchmark/environment.freeze.txt"
