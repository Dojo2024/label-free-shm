#!/usr/bin/env bash
# Build .venv from scratch after a fresh clone:  ./setup_env.sh
# - Python 3.11 venv via uv, packages pinned in requirements.txt
# - torch is the CPU-only build (pulled from the PyTorch CPU index)
# - openseespy needs the system libblas.so.3; a copy goes into .venv/extra-libs so that machines
#   without it can run with LD_LIBRARY_PATH=$PWD/.venv/extra-libs
set -euo pipefail
cd "$(dirname "$0")"
VIRTUAL_ENV= uv venv --python 3.11 .venv
VIRTUAL_ENV= uv pip install --python .venv/bin/python \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -r requirements.txt
mkdir -p .venv/extra-libs
cp "$(readlink -f /lib/x86_64-linux-gnu/libblas.so.3)" .venv/extra-libs/libblas.so.3
.venv/bin/python -c "import openseespy.opensees, sbi, torch; print('env ok, torch', torch.__version__)"
