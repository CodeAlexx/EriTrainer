#!/usr/bin/env bash
# Klein 9B + --offload under compute-sanitizer.
# Triggers the step-2 CUDA_ERROR_INVALID_VALUE crash with pool ON.
#
# Pass --tool synccheck (default; fast, ~2x slowdown, catches stream-order
# bugs) or --tool memcheck (10x slowdown but catches everything). Use
# synccheck first.
#
# Output: /tmp/k9_sanitizer_5/sanitizer.log

set -euo pipefail

TOOL="${1:-synccheck}"
OUT=/tmp/k9_sanitizer_5
mkdir -p "$OUT"

cd /home/alex/EriDiffusion/EriDiffusion-v2

export FLAME_ALLOC_POOL=1
export LD_LIBRARY_PATH=/home/alex/libtorch-cu124/libtorch/lib
export RUST_LOG=info

MODEL=/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors
CACHE=/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_klein9b/
BIN=./target/release/train_klein

echo "[sanitizer] tool=$TOOL  output=$OUT/sanitizer.log"
echo "[sanitizer] FLAME_ALLOC_POOL=1 (force pool ON; trigger the bug)"
echo "[sanitizer] expected: clean step 1, crash at step 2 load_file"
echo

compute-sanitizer \
  --tool "$TOOL" \
  --print-limit 20 \
  --launch-timeout 0 \
  --target-processes all \
  "$BIN" \
    --config configs/klein9b_alina.json \
    --transformer "$MODEL" \
    --cache-dir "$CACHE" \
    --output-dir "$OUT" \
    --steps 5 \
    --rank 16 --lora-alpha 16.0 --batch-size 1 \
    --offload \
    --sample-every 0 \
    --warmup-steps 100 \
  2>&1 | tee "$OUT/sanitizer.log"

echo
echo "[sanitizer] done. Errors found:"
grep -E "ERROR|========= Invalid|========= Race|========= Saved host|========= Misaligned" "$OUT/sanitizer.log" | head -30 || echo "  (none flagged)"
