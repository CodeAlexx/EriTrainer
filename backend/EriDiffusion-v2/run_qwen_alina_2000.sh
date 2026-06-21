#!/usr/bin/env bash
# Qwen-Image LoRA, 2000 steps, OneTrainer "#qwen LoRA 24GB" params minus quantization (BF16, no FP8).
# rank 16 / alpha 1.0 (OT default) / lr 3e-4 / constant / warmup 200 / adamw / logit-normal.
# Sample at start + every 500 (step-0 baseline, 500/1000/1500, final 2000).
set -euo pipefail
cd /home/alex/EriDiffusion/EriDiffusion-v2
mkdir -p output/qwen_alina_2000

P1="alverone , a high-resolution photograph featuring a young caucasian woman with long blonde hair, wearing a casual white sundress, standing in a sunlit garden, soft natural lighting, professional photography"
P2="alverone , a high-resolution photograph featuring a young caucasian woman with long blonde hair, wearing a black evening dress, in an elegant indoor setting, dramatic studio lighting, professional photography"

RUST_LOG=info LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib FLAME_CHECKPOINT=1 \
  target/release/train_qwenimage \
  --model /home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer \
  --cache-dir cache/AlinaAignatova_qwen_512 \
  --steps 2000 --rank 16 --lora-alpha 1.0 --lr 3e-4 \
  --resolution 512 --warmup-steps 200 \
  --lr-scheduler constant --optimizer adamw \
  --timestep-distribution logit_normal \
  --save-every 500 --save-mode full \
  --sample-every 500 \
  --sample-prompt-1 "$P1" \
  --sample-prompt-2 "$P2" \
  --sample-neg-prompt "" \
  --sample-vae /home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors \
  --sample-text-encoder /home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder \
  --sample-tokenizer /home/alex/.serenity/models/checkpoints/qwen-image-2512/tokenizer/tokenizer.json \
  --sample-size 1024 --sample-steps 50 --sample-cfg 4.0 \
  --output-dir output/qwen_alina_2000
