#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f rigging.local.json ]]; then
  # Local machine paths only; parse JSON rather than executing a shell file.
  mapfile -d '' -t rigging_paths < <(python3 - <<'PY'
import json
with open('rigging.local.json') as handle:
    config=json.load(handle)
for field in ('checkout','python'):
    print(config.get(field,''),end='\0')
PY
)
  export RLSIM_SKINTOKENS_HOME="${RLSIM_SKINTOKENS_HOME:-${rigging_paths[0]:-}}"
  export RLSIM_SKINTOKENS_PYTHON="${RLSIM_SKINTOKENS_PYTHON:-${rigging_paths[1]:-}}"
fi
if [[ -x .venv/bin/rlsim ]]; then
  exec .venv/bin/rlsim serve "$@"
elif [[ -x .venv/bin/morpholab ]]; then
  exec .venv/bin/morpholab serve "$@"
else
  echo "Run ./setup.sh first."
  exit 1
fi
