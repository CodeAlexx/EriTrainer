#!/usr/bin/env bash
# Run HiDream-O1 G0 trainer-forward parity end-to-end.
#
# Steps:
#   1. Generate the PyTorch reference dump (skip if it exists, unless --regen).
#   2. cargo build --release --bin parity_hidream_o1_g0.
#   3. Run the Rust parity binary against the dump + on-disk weights.
#
# Optional:
#   --regen           re-run the Python script even if the dump exists
#   --python <bin>    override `python3`
#   --model-path PATH override the HiDream-O1 weight dir
#
# Exit codes propagate from the Rust binary: 0=PASS, 1=FAIL, 2=BLOCKED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_TENSORS="/tmp/hidream_o1_g0_python_ref.safetensors"
REF_META="/tmp/hidream_o1_g0_python_ref_meta.json"
PY_SCRIPT="$SCRIPT_DIR/hidream_o1_g0_python_ref.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/home/alex/HiDream-O1-Image-Dev-weights}"
REGEN=0

usage() {
    cat <<EOF
Usage: $0 [--model-path PATH] [--regen] [--python BIN]

Runs HiDream-O1 G0 trainer-forward parity test.

Options:
  --model-path PATH   HiDream-O1 weight dir (default: $MODEL_PATH)
  --regen             Re-run the Python reference dump even if cached.
  --python BIN        Python interpreter (default: python3).

Exit codes:
  0 = PASS, 1 = FAIL, 2 = BLOCKED.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2;;
        --regen)      REGEN=1; shift;;
        --python)     PYTHON_BIN="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
    esac
done

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 2
fi

# --- Step 1: reference dump --------------------------------------------------
if [[ "$REGEN" -eq 1 ]] || [[ ! -f "$REF_TENSORS" ]] || [[ ! -f "$REF_META" ]]; then
    echo "==> [1/3] Generating PyTorch reference dump"
    echo "    python = $PYTHON_BIN"
    echo "    script = $PY_SCRIPT"
    if ! "$PYTHON_BIN" -c "import torch, transformers, safetensors, einops" 2>/dev/null; then
        echo "ERROR: $PYTHON_BIN missing torch / transformers / safetensors / einops." >&2
        exit 2
    fi
    "$PYTHON_BIN" "$PY_SCRIPT"
else
    echo "==> [1/3] Reference dump already present at $REF_TENSORS (use --regen to force)"
fi

# --- Step 2: build -----------------------------------------------------------
echo "==> [2/3] Building parity_hidream_o1_g0 (release)"
export LD_LIBRARY_PATH="/home/alex/libtorch-cu124/libtorch/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
cargo build --release --bin parity_hidream_o1_g0

# --- Step 3: run -------------------------------------------------------------
echo "==> [3/3] Running parity against $MODEL_PATH"
exec ./target/release/parity_hidream_o1_g0 \
    --model-path "$MODEL_PATH" \
    --ref-path "$REF_TENSORS" \
    --meta-path "$REF_META"
