#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
device="${1:-auto}"
case "$device" in
  auto|cpu|cuda) ;;
  *) printf '%s\n' "Usage: ./setup.sh [auto|cpu|cuda]" >&2; exit 2 ;;
esac

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip

# Preserve an existing installation unless a different variant is needed.
# macOS gets the standard PyPI build, which includes MPS on supported hardware.
installed_variant=$(.venv/bin/python -c 'import torch; print("cuda" if torch.version.cuda else "cpu")' 2>/dev/null || true)
install_variant="$device"
if [[ "$install_variant" == auto ]]; then
  install_variant=cpu
  nvidia_smi=$(command -v nvidia-smi || true)
  if [[ -z "$nvidia_smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    nvidia_smi=/usr/lib/wsl/lib/nvidia-smi
  fi
  if [[ "$(uname -s)" == Linux && -n "$nvidia_smi" ]] && "$nvidia_smi" -L >/dev/null 2>&1; then
    install_variant=cuda
  fi
fi
if [[ -z "$installed_variant" ||
      ( "$device" != auto && "$installed_variant" != "$device" ) ||
      ( "$install_variant" == cuda && "$installed_variant" != cuda ) ]]; then
  if [[ "$install_variant" == cuda ]]; then
    .venv/bin/python -m pip install --upgrade --force-reinstall 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
  elif [[ "$(uname -s)" == Darwin ]]; then
    .venv/bin/python -m pip install --upgrade --force-reinstall torch
  else
    .venv/bin/python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/rlsim doctor --device "$device"
