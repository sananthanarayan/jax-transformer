#!/usr/bin/env bash
# Reproduce the Phase 1 length-generalization figure end-to-end.
#
# Usage:
#   bash scripts/reproduce.sh              # full sweep (~10-20 min on a 14GB GPU)
#   bash scripts/reproduce.sh quick        # fast sanity sweep (~2-5 min)
#
# Assumes Python 3.10+ and that you've already created/activated a venv.
# Pass --create-venv as the first argument to have this script create one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
MODE="${1:-full}"

if [[ "$MODE" == "--create-venv" ]]; then
    "$PYTHON" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PYTHON=python
    MODE="${2:-full}"
fi

echo "[1/3] installing dependencies"
"$PYTHON" -m pip install --quiet -r requirements.txt

echo "[2/3] running sweep ($MODE)"
case "$MODE" in
    quick)
        "$PYTHON" scripts/run_sweep.py --epochs 2 --eval-samples 100
        ;;
    full)
        "$PYTHON" scripts/run_sweep.py
        ;;
    *)
        echo "Unknown mode: $MODE (expected 'full' or 'quick')" >&2
        exit 1
        ;;
esac

echo "[3/3] plotting"
"$PYTHON" scripts/plot.py

echo
echo "Done. Figure: results/length_gen.png   Raw numbers: results/sweep.json"
