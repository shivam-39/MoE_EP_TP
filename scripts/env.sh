#!/usr/bin/env bash
# Point PATH / library path / MPICC at the project-local OpenMPI prefix.
# Works when sourced from bash or zsh.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _this="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _this="${(%):-%x}"
else
  _this="$0"
fi
ROOT="$(cd "$(dirname "$_this")/.." && pwd)"
PREFIX="$ROOT/.deps/openmpi"
unset _this

if [[ ! -x "$PREFIX/bin/mpicc" ]]; then
  echo "Local OpenMPI not found at $PREFIX" >&2
  echo "Run: ./scripts/install_openmpi.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$PREFIX/bin:$PATH"
export MPICC="$PREFIX/bin/mpicc"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# macOS loader
export DYLD_LIBRARY_PATH="$PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
