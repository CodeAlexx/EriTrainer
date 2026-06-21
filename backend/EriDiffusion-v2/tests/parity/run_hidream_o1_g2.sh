#!/usr/bin/env bash
# HiDream-O1 G2 — Hard Launch Gate (single-sample overfit smoke).
#
# Pass criteria (ALL must hold):
#  1. Grad flow:  step-1 lora_B grads non-zero (>= 99% of B params).
#  2. Loss sanity: 50-step MSE drops >= 40% from step 1 to step 50 on a
#                  SINGLE cached sample (catches loss-target sign errors).
#  3. No crashes / no NaN / no OOM.
#
# Inputs:
#  - HiDream-O1 weights at $MODEL_PATH
#  - Source images at $SRC_IMAGES_DIR (a tiny subset is copied into /tmp)
#  - Trainer + prep binaries at $REPO_ROOT/target/release/
#
# Outputs:
#  - /tmp/g2_overfit_input/   (single-image symlinks)
#  - /tmp/g2_overfit_cache/   (cache schema v2)
#  - /tmp/g2_overfit_output/  (skipped saves during smoke)
#  - /tmp/g2_overfit.log      (trainer stdout+stderr)
#
# Exit: 0 = G2 PASS, 1 = G2 FAIL, 2 = BLOCKED.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/alex/EriDiffusion/EriDiffusion-v2}"
MODEL_PATH="${MODEL_PATH:-/home/alex/HiDream-O1-Image-Dev-weights}"
SRC_IMAGES_DIR="${SRC_IMAGES_DIR:-/home/alex/eri2}"
LIBTORCH_LIB="${LIBTORCH_LIB:-/home/alex/libtorch-cu124/libtorch/lib}"

INPUT_DIR="/tmp/g2_overfit_input"
CACHE_DIR="/tmp/g2_overfit_cache"
OUTPUT_DIR="/tmp/g2_overfit_output"
LOG_FILE="/tmp/g2_overfit.log"

STEPS=50
RANK=16
LORA_ALPHA=16
LR=3e-4

PREP_BIN="$REPO_ROOT/target/release/prepare_hidream_o1"
TRAIN_BIN="$REPO_ROOT/target/release/train_hidream_o1"

blocked() { echo "G2 BLOCKED — $*" >&2; exit 2; }
fail()    { echo "G2 FAIL — $*";   exit 1; }
pass()    { echo "G2 PASS — $*";   exit 0; }

# ── Preconditions
[[ -d "$MODEL_PATH"      ]] || blocked "MODEL_PATH missing: $MODEL_PATH"
[[ -d "$SRC_IMAGES_DIR"  ]] || blocked "SRC_IMAGES_DIR missing: $SRC_IMAGES_DIR"
[[ -d "$LIBTORCH_LIB"    ]] || blocked "LIBTORCH_LIB missing: $LIBTORCH_LIB"
[[ -x "$PREP_BIN"        ]] || blocked "prepare_hidream_o1 missing: $PREP_BIN"
[[ -x "$TRAIN_BIN"       ]] || blocked "train_hidream_o1 missing: $TRAIN_BIN"

export LD_LIBRARY_PATH="$LIBTORCH_LIB:${LD_LIBRARY_PATH:-}"
# Keep this hard gate parseable even when the parent shell exports
# RUST_LOG=warn. Override with G2_RUST_LOG=warn only for intentional quiet runs.
export RUST_LOG="${G2_RUST_LOG:-info}"
export FLAME_ASSERT_GRAD_FLOW=1

# ── Stage 1: prep a single-sample cache
echo "==> [1/3] Staging single-sample input at $INPUT_DIR"
rm -rf "$INPUT_DIR" "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_FILE"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Pick first jpg with a matching .txt caption; symlink BOTH into INPUT_DIR.
PICKED=""
for jpg in "$SRC_IMAGES_DIR"/*.jpg; do
    [[ -f "$jpg" ]] || continue
    stem="$(basename "$jpg" .jpg)"
    txt="$SRC_IMAGES_DIR/$stem.txt"
    if [[ -f "$txt" ]]; then
        ln -s "$jpg" "$INPUT_DIR/$stem.jpg"
        ln -s "$txt" "$INPUT_DIR/$stem.txt"
        PICKED="$stem"
        break
    fi
done
[[ -n "$PICKED" ]] || blocked "no jpg+txt pair found under $SRC_IMAGES_DIR"
echo "    picked sample: $PICKED"

echo "==> [2/3] Running prepare_hidream_o1 (single-sample, resolution 512)"
"$PREP_BIN" \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$CACHE_DIR" \
    --model-path "$MODEL_PATH" \
    --resolution 512 \
    --max-samples 1 \
    --skip-existing 2>&1 | tee "$LOG_FILE.prep" \
    || blocked "prepare_hidream_o1 failed (see $LOG_FILE.prep)"

META="$CACHE_DIR/_meta.json"
[[ -f "$META" ]] || blocked "_meta.json missing at $META"
if ! grep -qE '"format"[[:space:]]*:[[:space:]]*"hidream-o1-v(2|3)"' "$META"; then
    blocked "cache _meta.json format != hidream-o1-v2|v3 (got $(grep -o '"format"[[:space:]]*:[[:space:]]*"[^"]*"' "$META" | head -1))"
fi
SAMPLE_COUNT="$(ls -1 "$CACHE_DIR"/*.safetensors 2>/dev/null | wc -l)"
[[ "$SAMPLE_COUNT" -ge 1 ]] || blocked "no cache shards produced in $CACHE_DIR"
echo "    cache v2 OK, $SAMPLE_COUNT sample(s)"

# WORKAROUND: prepare_hidream_o1 names shards `{md5_hash}.safetensors`, but
# train_hidream_o1 only globs `sample_*.safetensors` (BUG-6 reference). Rename
# the produced shards to the expected pattern so the smoke can run. This is a
# test-only file move — production code is untouched. The naming mismatch
# itself is a real production bug that needs fixing separately.
i=0
for f in "$CACHE_DIR"/*.safetensors; do
    bn="$(basename "$f")"
    if [[ "$bn" != sample_* ]]; then
        new="$(printf "sample_%06d.safetensors" "$i")"
        mv "$f" "$CACHE_DIR/$new"
        i=$((i+1))
    fi
done

# ── Stage 2: run trainer 50 steps
echo "==> [3/3] Running train_hidream_o1 ($STEPS steps, rank=$RANK, lr=$LR, FLAME_ASSERT_GRAD_FLOW=1)"
T_START=$(date +%s)
"$TRAIN_BIN" \
    --cache-dir   "$CACHE_DIR" \
    --model-path  "$MODEL_PATH" \
    --output-dir  "$OUTPUT_DIR" \
    --steps       "$STEPS" \
    --rank        "$RANK" \
    --lora-alpha  "$LORA_ALPHA" \
    --lr          "$LR" \
    --save-every  0 \
    --sample-every 0 \
    > "$LOG_FILE" 2>&1
TRAIN_RC=$?
T_END=$(date +%s)
WALL=$((T_END - T_START))
echo "    train exit=$TRAIN_RC, wall=${WALL}s, log=$LOG_FILE"

# ── Stage 3: parse log

# Grad-flow check FIRST — if assert_grad_flow panicked, that's the most
# diagnostic failure. The summary precedes the "panicked at" line in the log.
GRAD_DEAD_LINE="$(grep -E "\[grad-flow\] [0-9]+ dead / [0-9]+ ok" "$LOG_FILE" | tail -1 || true)"
if [[ -n "$GRAD_DEAD_LINE" ]]; then
    DEAD_PARAMS="$(grep -E "^    - " "$LOG_FILE" | head -8 | sed 's/^    - //' | paste -sd ',' -)"
    fail "grad-flow not clean — $GRAD_DEAD_LINE; first dead params: $DEAD_PARAMS"
fi

if grep -qE "CUDA out of memory|cudaErrorMemoryAllocation" "$LOG_FILE"; then
    HEAD_ERR="$(grep -m1 -E '(CUDA out of memory|cudaErrorMemoryAllocation)' "$LOG_FILE")"
    fail "OOM: $HEAD_ERR"
fi
if grep -qE "NaN/Inf loss" "$LOG_FILE"; then
    HEAD_NAN="$(grep -m1 -E 'NaN/Inf loss' "$LOG_FILE")"
    fail "NaN/Inf loss detected: $HEAD_NAN"
fi
if grep -qE "(panicked at|^Error:|fatal runtime error)" "$LOG_FILE"; then
    HEAD_ERR="$(grep -m1 -E '(panicked at|^Error:|fatal runtime error)' "$LOG_FILE")"
    fail "trainer crashed: $HEAD_ERR"
fi

# Extract step 1 and step 50 losses from `[HiDreamO1-lora] step N/T | ... | loss X.XXXX | ...`
loss_at_step() {
    local want="$1"
    grep -oE "step ${want}/${STEPS} \| [^|]*\| loss [0-9.eE+-]+" "$LOG_FILE" \
        | tail -1 \
        | grep -oE "loss [0-9.eE+-]+" \
        | awk '{print $2}'
}
LOSS_1="$(loss_at_step 1)"
LOSS_N="$(loss_at_step "$STEPS")"

if [[ -z "$LOSS_1" || -z "$LOSS_N" ]]; then
    echo "----- log tail -----" >&2
    tail -40 "$LOG_FILE" >&2
    fail "could not extract step-1/step-$STEPS loss from log (rc=$TRAIN_RC); see $LOG_FILE"
fi

# Compute % drop (positive = loss decreased).
DROP_PCT="$(awk -v a="$LOSS_1" -v b="$LOSS_N" 'BEGIN{ if (a+0==0) print 0; else printf "%.2f", (a-b)/a*100.0; }')"
echo "    loss step 1 = $LOSS_1"
echo "    loss step $STEPS = $LOSS_N"
echo "    drop = ${DROP_PCT}%"

# Grad-flow check.
GRAD_CLEAN="$(grep -E "\[grad-flow\] step 1 clean \([0-9]+ params\)" "$LOG_FILE" | tail -1 || true)"
GRAD_DEAD_LINE="$(grep -E "\[grad-flow\].*dead.*ok" "$LOG_FILE" | tail -1 || true)"
if [[ -n "$GRAD_DEAD_LINE" ]]; then
    fail "grad-flow not clean at step 1 — $GRAD_DEAD_LINE"
fi
if [[ -z "$GRAD_CLEAN" ]]; then
    # No clean line + no dead line = assertion never ran.
    fail "grad-flow assertion produced no output (FLAME_ASSERT_GRAD_FLOW=1 wired but not triggered?)"
fi
OK_COUNT="$(echo "$GRAD_CLEAN" | grep -oE '[0-9]+' | tail -1)"
echo "    grad-flow: clean, $OK_COUNT params"

# grad-guard zeroed-N (informational, but flag if abnormally high).
ZEROED_HI="$(grep -E "\[grad-guard\] step=1 zeroed [0-9]+ non-finite-grad" "$LOG_FILE" \
             | awk '{for(i=1;i<=NF;i++) if($i=="zeroed") {print $(i+1); exit}}')"
if [[ -n "$ZEROED_HI" && "$ZEROED_HI" -gt 10 ]]; then
    fail "grad-guard zeroed $ZEROED_HI non-finite-grad params at step 1 (>10 indicates numeric instability)"
fi

# Verdict
DROP_OK="$(awk -v d="$DROP_PCT" 'BEGIN{ print (d+0>=40.0)?1:0 }')"
if [[ "$DROP_OK" != "1" ]]; then
    fail "loss dropped only ${DROP_PCT}% in $STEPS steps (need >=40%); step1=$LOSS_1 step$STEPS=$LOSS_N"
fi
if [[ "$TRAIN_RC" -ne 0 ]]; then
    fail "trainer exited rc=$TRAIN_RC despite log looking clean"
fi

pass "loss ${LOSS_1} -> ${LOSS_N} (${DROP_PCT}% drop in $STEPS steps), grad-flow clean ($OK_COUNT params), wall ${WALL}s"
