#!/bin/bash
# Klein 9B LoRA — SAME params as klein_3000_lr3e5 (lr 3e-5, rank/alpha 16,
# shift 1.8, batch 1, clip 1.0) but with COSINE LR decay -> 0 to fix the
# proven late-training collapse (clean @ step1200, black @ 1600, degraded @
# 2400 at constant LR). Cosine taper locks in the subject as lr -> 0.
# Approved by user 2026-05-26. Config: configs/klein9b_alina_cosine.json.
cd /home/alex/EriDiffusion/EriDiffusion-v2
QDIR=/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUST_LOG=info FLAME_ALLOC_POOL=0 FLAME_USE_STATIC_SLAB=0 FLAME_ASSERT_GRAD_FLOW=1 \
  LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
  ./target/release/train_klein \
    --config configs/klein9b_alina_cosine.json \
    --cache-dir cache/alina_klein9b \
    --transformer /home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors \
    --steps 3000 --rank 16 --lora-alpha 16 --lr 3e-5 \
    --batch-size 1 --warmup-steps 100 --offload \
    --sample-every 200 \
    --sample-prompt "vrtlEri2, a portrait photo of a woman, soft diffused natural lighting, detailed skin texture" \
    --sample-qwen3 "$QDIR" \
    --sample-tokenizer "$QDIR/tokenizer.json" \
    --sample-vae /home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors \
    --sample-size 512 \
    --output-dir output/klein_3000_cosine_b1
