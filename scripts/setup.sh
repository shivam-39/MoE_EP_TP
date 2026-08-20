#!/usr/bin/env bash
# One-shot setup: local OpenMPI + uv .venv + Python deps (mpi4py built against local mpicc).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v curl >/dev/null; then
  echo "curl is required to download OpenMPI" >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if ! command -v make >/dev/null || ! command -v cc >/dev/null; then
  echo "A C compiler and make are required to build OpenMPI (clang or gcc)." >&2
  exit 1
fi

if [[ ! -x "$ROOT/.deps/openmpi/bin/mpicc" ]]; then
  bash "$ROOT/scripts/install_openmpi.sh"
fi

# env.sh uses `return` when sourced; `set -e` + source is fine
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"

uv venv
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

uv pip install -r "$ROOT/requirements.txt"
# Wheel builds are linked to a system libmpi we do not ship; compile against .deps/openmpi.
uv pip install --reinstall --no-binary mpi4py "mpi4py>=4.0"

echo
echo "Setup complete. In a new shell:"
echo "  source scripts/activate.sh"
echo "  mpirun -n 2 python moe/test_moe.py"
