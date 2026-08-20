#!/usr/bin/env bash
# Source this: local OpenMPI on PATH + project .venv activated.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _this="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _this="${(%):-%x}"
else
  _this="$0"
fi
ROOT="$(cd "$(dirname "$_this")/.." && pwd)"
unset _this
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
