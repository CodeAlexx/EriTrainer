#!/usr/bin/env bash
# Run HiDream M1.5 Llama-3.1-8B encoder parity test end-to-end.
#
# Steps:
#   1. Generate the PyTorch reference dump (skip if it exists, unless --regen).
#   2. cargo build --release --bin parity_hidream_m1_llama3.
#   3. Run the Rust parity binary against the dump + on-disk weights.
#
# Required env / args:
#   LLAMA_PATH=/path/to/unsloth-Meta-Llama-3.1-8B-Instruct/   (or pass --llama-path)
#
# Optional:
#   --regen           re-run the Python script even if the dump exists
#   --python <bin>    override `python3`
#
# Exit codes propagate from the Rust binary: 0=PASS, 1=FAIL, 2=BLOCKED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_TENSORS="/tmp/hidream_m1_llama3_ref.safetensors"
REF_META="/tmp/hidream_m1_llama3_ref_meta.json"
PY_SCRIPT="$SCRIPT_DIR/hidream_m1_llama3_python_ref.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REGEN=0
LLAMA_PATH="${LLAMA_PATH:-}"

usage() {
    cat <<EOF
Usage: $0 [--llama-path PATH] [--regen] [--python BIN]

Runs HiDream M1.5 Llama-3.1-8B parity test.

Options:
  --llama-path PATH   Path to unsloth Llama-3.1-8B-Instruct weights
                      (directory of HF shards or a single safetensors).
                      Defaults to \$LLAMA_PATH env var.
  --regen             Re-run the Python reference dump even if cached.
  --python BIN        Python interpreter to use (default: python3).

Exit codes:
  0 = PASS, 1 = FAIL, 2 = BLOCKED (missing deps / weights / dump).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-path) LLAMA_PATH="$2"; shift 2;;
        --regen)      REGEN=1; shift;;
        --python)     PYTHON_BIN="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
    esac
done

if [[ -z "$LLAMA_PATH" ]]; then
    echo "ERROR: --llama-path required (or set LLAMA_PATH env)." >&2
    echo "Download with:" >&2
    echo "  huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct \\" >&2
    echo "    --local-dir \$HOME/models/llama-3.1-8b-instruct" >&2
    exit 2
fi

if [[ ! -e "$LLAMA_PATH" ]]; then
    echo "ERROR: LLAMA_PATH does not exist: $LLAMA_PATH" >&2
    exit 2
fi

# --- Step 1: reference dump --------------------------------------------------
if [[ "$REGEN" -eq 1 ]] || [[ ! -f "$REF_TENSORS" ]] || [[ ! -f "$REF_META" ]]; then
    echo "==> [1/3] Generating PyTorch reference dump"
    echo "    python   = $PYTHON_BIN"
    echo "    script   = $PY_SCRIPT"
    if ! "$PYTHON_BIN" -c "import torch, transformers, safetensors" 2>/dev/null; then
        echo "ERROR: $PYTHON_BIN is missing torch / transformers / safetensors." >&2
        echo "       Activate an env with these installed and re-run." >&2
        exit 2
    fi
    "$PYTHON_BIN" "$PY_SCRIPT"
else
    echo "==> [1/3] Reference dump already present at $REF_TENSORS (use --regen to force)"
fi

# --- Step 2: build -----------------------------------------------------------
echo "==> [2/3] Building parity_hidream_m1_llama3 (release)"
export LD_LIBRARY_PATH="/home/alex/libtorch-cu124/libtorch/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
cargo build --release --bin parity_hidream_m1_llama3

# --- Step 3: run -------------------------------------------------------------
echo "==> [3/3] Running parity against $LLAMA_PATH"
exec ./target/release/parity_hidream_m1_llama3 \
    --llama-path "$LLAMA_PATH" \
    --ref-path "$REF_TENSORS" \
    --meta-path "$REF_META"
