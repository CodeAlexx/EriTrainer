#!/usr/bin/env bash
# Run HiDream-O1 G1 Rust self-consistency parity gate end-to-end.
#
# G1 compares the TRAINER forward path (`forward_lora` with empty LoRA) against
# the INFERENCE forward path (`forward`, no LoRA) inside the same Rust binary,
# on the same model instance, with the same inputs. With LoRA A=0 and B=0 the
# two paths are mathematically identical; any drift is a real divergence
# between trainer and inference code paths.
#
# Prereq: the G0 reference dump must already exist at
#   /tmp/hidream_o1_g0_python_ref.safetensors
# (G1 only consumes the INPUTS from that dump; it ignores the Python output.)
# If it's missing, run tests/parity/run_hidream_o1_g0.sh first.
#
# Exit codes propagate from the Rust binary: 0=PASS, 1=FAIL, 2=BLOCKED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_TENSORS="/tmp/hidream_o1_g0_python_ref.safetensors"
MODEL_PATH="${MODEL_PATH:-/home/alex/HiDream-O1-Image-Dev-weights}"

usage() {
    cat <<EOF
Usage: $0 [--model-path PATH]

Runs HiDream-O1 G1 Rust self-consistency parity gate.

Options:
  --model-path PATH   HiDream-O1 weight dir (default: $MODEL_PATH)

Exit codes:
  0 = PASS, 1 = FAIL, 2 = BLOCKED.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
    esac
done

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 2
fi

if [[ ! -f "$REF_TENSORS" ]]; then
    echo "ERROR: reference dump $REF_TENSORS not found." >&2
    echo "       Run tests/parity/run_hidream_o1_g0.sh first to generate it." >&2
    exit 2
fi

echo "==> [1/2] Building parity_hidream_o1_g1 (release)"
export LD_LIBRARY_PATH="/home/alex/libtorch-cu124/libtorch/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
cargo build --release --bin parity_hidream_o1_g1

echo "==> [2/2] Running G1 against $MODEL_PATH"
exec ./target/release/parity_hidream_o1_g1 \
    --model-path "$MODEL_PATH" \
    --ref-path "$REF_TENSORS"
