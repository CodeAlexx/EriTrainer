#!/usr/bin/env bash
# AsymFlow A0.5 — wrapper that produces (or skips) the Python ref dump
# and then runs the Rust parity harness.
#
# Exit codes propagate from the Rust binary:
#   0 PARITY pass | 1 PARITY fail | 2 BLOCKED (no ref dump)

set -u
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/home/alex/EriDiffusion}"
EDV2="${EDV2:-$REPO_ROOT/EriDiffusion-v2}"
PYTHON_REF="$EDV2/tests/parity/asymflow_a05_teacher_python_ref.py"
REF_DUMP="${REF_DUMP:-/tmp/asymflow_a05_teacher_ref.safetensors}"
REF_META="${REF_META:-/tmp/asymflow_a05_teacher_ref_meta.json}"

export LD_LIBRARY_PATH="/home/alex/libtorch-cu124/libtorch/lib:${LD_LIBRARY_PATH:-}"

echo "=== A0.5 step 1/2: Python reference dump ==="
if [[ -f "$REF_DUMP" && -f "$REF_META" && -z "${A05_FORCE_REGEN:-}" ]]; then
    echo "  reusing cached $REF_DUMP (set A05_FORCE_REGEN=1 to rebuild)"
else
    set +e
    python3 "$PYTHON_REF"
    py_status=$?
    set -e
    if [[ $py_status -ne 0 ]]; then
        echo "  Python ref dump did NOT complete (exit=$py_status)."
        echo "  Continuing — Rust harness will report BLOCKED."
    fi
fi

echo
echo "=== A0.5 step 2/2: Rust parity binary ==="
cd "$EDV2"
cargo build --release --bin parity_asymflow_a05_teacher 2>&1 | tail -5
build_status=$?
if [[ $build_status -ne 0 ]]; then
    echo "  Build failed."
    exit 2
fi

cargo run --release --quiet --bin parity_asymflow_a05_teacher -- \
    --ref-path "$REF_DUMP" \
    --meta-path "$REF_META"
exit $?
