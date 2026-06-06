#!/usr/bin/env bash
# Cross-platform runner (Linux / macOS / Windows-Git-Bash).
# - Requires conda. Errors clearly if it is missing.
# - Creates the conda env if absent, reuses it if present.
# - Installs deps, then runs the ranking step.
#
# Usage:
#   ./run.sh --candidates ./candidates.jsonl --out ./submission.csv
#   ./run.sh --precompute --candidates ./candidates.jsonl   # rebuild semantic artifact first
#
# The ranking step is CPU-only and offline. To rebuild the semantic artifact
# (torch + transformers), pass --precompute.
set -euo pipefail

ENV_NAME="${CANJOB_ENV:-canjob}"
PY_VERSION="3.12"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

red()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
info() { printf '\033[36m%s\033[0m\n' "$*"; }

# ---- 1. conda must exist --------------------------------------------------- #
if ! command -v conda >/dev/null 2>&1; then
  red "ERROR: conda not found on PATH."
  red "Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
  red "Then re-run: ./run.sh"
  exit 1
fi
info "conda: $(conda --version)"

# Make 'conda activate' usable inside a non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- 2. create env if missing, else reuse ---------------------------------- #
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "Reusing existing conda env: $ENV_NAME"
else
  info "Creating conda env: $ENV_NAME (python=$PY_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

# ---- 3. dependencies ------------------------------------------------------- #
info "Installing core dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# ---- 4. optional precompute (rebuild the small semantic artifact) ---------- #
DO_PRECOMPUTE=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--precompute" ]; then DO_PRECOMPUTE=1; else ARGS+=("$arg"); fi
done
if [ "$DO_PRECOMPUTE" = "1" ]; then
  info "Installing embeddings extras (torch, transformers)..."
  python -m pip install --quiet -r requirements-embeddings.txt
  info "Pre-computing semantic artifact..."
  python precompute.py "${ARGS[@]}"
fi

# ---- 5. run the ranking step ----------------------------------------------- #
info "Running ranking step..."
python rank.py "${ARGS[@]}"
info "Done."
