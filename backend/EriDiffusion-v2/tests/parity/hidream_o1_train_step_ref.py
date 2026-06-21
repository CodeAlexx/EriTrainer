#!/usr/bin/env python3
"""HiDream-O1 exact training-step reference dump.

This is stricter than the G0 inference-forward parity fixture: it loads one
real EDv2 cached training sample, pins noise + sigma, runs the ai-toolkit O1
model path, computes the velocity target/loss exactly like ai-toolkit, and
writes every tensor needed for the Rust side to replay the same step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_PATH = "/home/alex/HiDream-O1-Image-Full-weights"
DEFAULT_CACHE_DIR = (
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/"
    "gigerver3_hidream_o1_512_mropefix"
)
DEFAULT_OUT = Path("/tmp/hidream_o1_train_step_ref.safetensors")
DEFAULT_META = Path("/tmp/hidream_o1_train_step_ref_meta.json")
DEFAULT_LORA_OUT = Path("/tmp/hidream_o1_lora_step_ref.safetensors")
DTYPE_NAME = "bfloat16"
NOISE_SCALE = 8.0
T_EPS = 1.0e-3
LORA_RANK = 32
LORA_ALPHA = 32
LORA_ADAPTERS = 257  # 252 decoder + 5 resident heads (x_embedder.proj{1,2}, t_embedder1.mlp.{0,2}, final_layer2.linear)
MAX_GRAD_NORM = 1.0
ADAMW_LR = 1.0e-4
ADAMW_EPS = 1.0e-6
ADAMW_WEIGHT_DECAY = 1.0e-4  # matches current Rust trainer + V67's bnb default


def _die(msg: str, code: int = 2) -> None:
    print(f"[train-step-ref] FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def _repo_root() -> str:
    root = os.environ.get("EDV2_REFERENCE_ROOT")
    if root:
        return root
    candidates = [
        "/home/alex/edv2-reference",
        os.path.join("/home", "alex", "ai-" + "toolkit"),
    ]
    return next((p for p in candidates if os.path.isdir(p)), candidates[0])


def _first_sample(cache_dir: Path) -> Path:
    files = sorted(cache_dir.glob("sample_*.safetensors"))
    if not files:
        _die(f"no sample_*.safetensors files under {cache_dir}")
    return files[0]


class _O1BaseForLora:
    arch = "hidream-o1"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["layers"]


def _registry_key_from_lora_name(name: str) -> str:
    clean = name.replace("$$", ".")
    changed = True
    while changed:
        changed = False
        for prefix in (
            "transformer.",
            "base_model.model.",
            "model.",
        ):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                changed = True
    if clean.startswith("language_model."):
        clean = clean[len("language_model."):]
    # Accept decoder-layer adapters AND the 5 O1 resident head adapters
    # (matches Rust registry default_resident_target_keys).
    _RESIDENT_HEADS = (
        "x_embedder.proj1",
        "x_embedder.proj2",
        "t_embedder1.mlp.0",
        "t_embedder1.mlp.2",
        "final_layer2.linear",
    )
    if not (clean.startswith("layers.") or clean in _RESIDENT_HEADS):
        _die(f"unexpected O1 LoRA target name: {name} -> {clean}")
    return clean


def _parse_probe_layers(raw: str, max_layers: int) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            _die(f"invalid --probe-layers entry {part!r}")
        if idx < 0 or idx >= max_layers:
            _die(f"--probe-layers entry {idx} out of range [0,{max_layers})")
        if idx not in out:
            out.append(idx)
    return out or [0]


def _lora_named_params(network):
    out = {}
    for lora in network.unet_loras:
        key = _registry_key_from_lora_name(lora.lora_name)
        out[f"{key}.lora_A"] = lora.lora_down.weight
        out[f"{key}.lora_B"] = lora.lora_up.weight
    return dict(sorted(out.items()))


def _clone_or_zero_grad(param: "torch.Tensor") -> "torch.Tensor":
    import torch

    if param.grad is None:
        return torch.zeros_like(param, dtype=torch.float32, device="cpu")
    return param.grad.detach().float().contiguous().cpu()


def _clone_or_zero_tensor_grad(tensor: "torch.Tensor") -> "torch.Tensor":
    import torch

    if tensor.grad is None:
        return torch.zeros_like(tensor, dtype=torch.float32, device="cpu")
    return tensor.grad.detach().float().contiguous().cpu()


def _save_lora_tensor_set(dst, prefix: str, named_params) -> None:
    for name, param in named_params.items():
        dst[f"{prefix}.{name}"] = param.detach().float().contiguous().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--sample-path", default="")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--lora-out", default=str(DEFAULT_LORA_OUT))
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--lora-seed", type=int, default=4242)
    ap.add_argument("--t-scalar", type=float, default=0.5)
    ap.add_argument("--lora-step", action="store_true")
    ap.add_argument(
        "--dump-layers",
        action="store_true",
        help="also dump decoder input/layer/final-norm tensors into --out",
    )
    ap.add_argument(
        "--probe-layers",
        default=os.environ.get("HIDREAM_DUMP_PROBE_LAYERS", "0"),
        help="comma-separated decoder layers for detailed probe dumps when --dump-layers is set",
    )
    ap.add_argument("--lr", type=float, default=ADAMW_LR)
    ap.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    ap.add_argument("--weight-decay", type=float, default=ADAMW_WEIGHT_DECAY)
    ap.add_argument(
        "--use-flash-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ai-toolkit training calls HiDream-O1 with use_flash_attn=True",
    )
    ap.add_argument(
        "--deterministic-attn",
        action="store_true",
        default=os.environ.get("HIDREAM_REF_DETERMINISTIC_ATTN") == "1",
        help="enable torch deterministic algorithms for Flash Attention backward parity runs",
    )
    args = ap.parse_args()

    if not (0.0 < args.t_scalar < 1.0):
        _die(f"--t-scalar must be in (0,1), got {args.t_scalar}")

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as e:
        _die(f"missing Python dependency: {e}")

    if not torch.cuda.is_available():
        _die("CUDA is required for the HiDream-O1 reference forward")

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from extensions_built_in.diffusion_models.hidream.src.hidream_o1.qwen3_vl_transformers import (
            ALL_ATTENTION_FUNCTIONS,
            Qwen3VLForConditionalGeneration,
            apply_rotary_pos_emb,
            eager_attention_forward,
            repeat_kv,
        )
    except Exception as e:  # noqa: BLE001
        _die(f"failed importing ai-toolkit HiDream-O1 modules from {repo_root}: {e}")

    cache_dir = Path(args.cache_dir)
    sample_path = Path(args.sample_path) if args.sample_path else _first_sample(cache_dir)
    out_path = Path(args.out)
    meta_path = Path(args.meta)
    device = torch.device("cuda:0")
    dtype = getattr(torch, DTYPE_NAME)

    print(f"[train-step-ref] model          = {args.model_path}")
    print(f"[train-step-ref] sample         = {sample_path}")
    print(f"[train-step-ref] seed           = {args.seed}")
    print(f"[train-step-ref] t_scalar       = {args.t_scalar}")
    print(f"[train-step-ref] t_pixeldit     = {1.0 - args.t_scalar}")
    print(f"[train-step-ref] lora_step      = {args.lora_step}")
    print(f"[train-step-ref] use_flash_attn = {args.use_flash_attn}")
    print(f"[train-step-ref] deterministic = {args.deterministic_attn}")
    print(f"[train-step-ref] repo_root      = {repo_root}")

    sample = load_file(str(sample_path), device="cpu")
    patches = sample["patches"].to(device=device, dtype=dtype)
    input_ids = sample["input_ids"].to(device=device, dtype=torch.long)
    position_ids = sample["position_ids"].to(device=device, dtype=torch.long)
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(1)
    vinput_mask = sample["vinput_mask"].to(device=device).bool()
    token_types = sample["token_types"].to(device=device, dtype=dtype)

    gen = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn(tuple(patches.shape), generator=gen, dtype=torch.float32)
    noise = noise.to(device=device, dtype=dtype)
    scaled_noise = noise * NOISE_SCALE
    t_scalar = torch.tensor([args.t_scalar], device=device, dtype=torch.float32)
    t_pixeldit = torch.tensor([1.0 - args.t_scalar], device=device, dtype=torch.float32)
    sigma = max(args.t_scalar, T_EPS)

    noisy = patches * (1.0 - args.t_scalar) + scaled_noise * args.t_scalar
    target_velocity = (scaled_noise - patches).detach()

    print("[train-step-ref] shapes:")
    print(f"    patches      {tuple(patches.shape)} {patches.dtype}")
    print(f"    input_ids    {tuple(input_ids.shape)} {input_ids.dtype}")
    print(f"    position_ids {tuple(position_ids.shape)} {position_ids.dtype}")
    print(f"    token_types  {tuple(token_types.shape)} {token_types.dtype}")
    print(f"    vinput_mask  {tuple(vinput_mask.shape)} {vinput_mask.dtype}")
    print(f"    noisy        {tuple(noisy.shape)} {noisy.dtype}")

    t0 = time.time()
    print("[train-step-ref] loading model ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    print(f"[train-step-ref] model loaded in {time.time() - t0:.1f}s")

    lora_save = {}
    lora_meta = None
    optimizer = None
    lora_named = None
    lora_params = None
    if args.lora_step:
        try:
            import bitsandbytes as bnb
            from toolkit.lora_special import LoRASpecialNetwork
        except ImportError as e:
            _die(f"missing LoRA-step dependency: {e}")

        torch.manual_seed(args.lora_seed)
        print("[train-step-ref] attaching ai-toolkit LoRA ...")
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=model,
            multiplier=1.0,
            lora_dim=LORA_RANK,
            alpha=LORA_ALPHA,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["Qwen3VLForConditionalGeneration"],
            ignore_if_contains=["lm_head", "patch_embed", "visual"],
            transformer_only=False,  # include 5 resident O1 head adapters (matches Rust --include-resident-lora)
            is_transformer=True,
            base_model=_O1BaseForLora(),
        )
        network.force_to(device, dtype=torch.float32)
        network.apply_to([], model, False, True)
        network.prepare_grad_etc([], model)
        network.is_active = True
        network._update_torch_multiplier()
        lora_named = _lora_named_params(network)
        if len(lora_named) != LORA_ADAPTERS * 2:
            _die(
                f"expected {LORA_ADAPTERS * 2} O1 LoRA tensors, "
                f"got {len(lora_named)}"
            )
        lora_params = list(lora_named.values())
        optimizer = bnb.optim.AdamW8bit(
            lora_params,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=ADAMW_EPS,
            weight_decay=args.weight_decay,
        )
        _save_lora_tensor_set(lora_save, "init", lora_named)
        print(
            "[train-step-ref] LoRA attached: "
            f"{len(network.unet_loras)} adapters, rank={LORA_RANK}, alpha={LORA_ALPHA}"
        )

    print("[train-step-ref] running ai-toolkit training-style forward ...")
    t1 = time.time()
    layer_dump = {}
    hooks = []
    # Trap (soul.md pattern): capture intermediate tensors in layer 0's attention
    # backward path so we can walk the gradient chain and find where Q/K/V LoRA-B
    # grad cos collapses from ~1.0 to ~0.05. V path probes first (simplest — no
    # RMSNorm, no RoPE for V). Probes: o_proj input grad (== attn_out_grad,
    # gradient entering SDPA bwd from output side) and v_proj output grad
    # (gradient arriving at V projection output, ie. immediately upstream of
    # V LoRA-B). Saved as `grad_probe.layers.0.attn_out` and
    # `grad_probe.layers.0.v_proj_out` after backward().
    layer0_probes: dict[str, "torch.Tensor"] = {}
    if args.dump_layers:
        try:
            first_layer = model.model.language_model.layers[0]
            # Probe the LAST decoder layer (closest to loss): cleanest signal
            # for localizing the Q/K/V LoRA-B grad-collapse bug, since
            # upstream-of-attention grads are clean here (o_proj LoRA-B
            # cos≈0.999 at layer 35) while downstream attention LoRA-B
            # grads collapse (cos≈0.05). Matches Rust's probe location.
            probe_layer_idx = int(
                os.environ.get(
                    "HIDREAM_BWD_PROBE_LAYER",
                    str(model.config.text_config.num_hidden_layers - 1),
                )
            )
            if probe_layer_idx < 0 or probe_layer_idx >= model.config.text_config.num_hidden_layers:
                _die(f"HIDREAM_BWD_PROBE_LAYER out of range: {probe_layer_idx}")
            probe_layer = model.model.language_model.layers[probe_layer_idx]
            final_layer = model.model.final_layer2
        except Exception as e:  # noqa: BLE001
            _die(f"failed to install layer dump hooks: {e}")

        def _capture_layer0_input(_module, module_args):
            layer_dump["hidden_input_layer_00"] = (
                module_args[0].detach().float().contiguous().cpu()
            )

        def _capture_final_norm_input(_module, module_args):
            layer_dump["hidden_final_norm"] = (
                module_args[0].detach().float().contiguous().cpu()
            )

        hooks.append(first_layer.register_forward_pre_hook(_capture_layer0_input))
        hooks.append(final_layer.register_forward_pre_hook(_capture_final_norm_input))

        if args.lora_step:
            def _retain_probe(name, t):
                if isinstance(t, (tuple, list)):
                    if not t:
                        return
                    t = t[0]
                if t.requires_grad:
                    t.retain_grad()
                    layer0_probes[name] = t

            def _trap_oproj_input(_module, module_args):
                # o_proj's INPUT is attn_out after reshape — gradient here is what
                # enters SDPA's backward from the output side.
                _retain_probe("attn_out", module_args[0])

            def _trap_oproj_output(_module, _input, output):
                _retain_probe("o_proj_out", output)

            def _trap_post_attn_input(_module, module_args):
                _retain_probe("after_attn", module_args[0])

            def _trap_post_attn_output(_module, _input, output):
                _retain_probe("normed2", output)

            def _trap_layer_output(_module, _input, output):
                _retain_probe("hidden_out", output)

            def _trap_vproj_output(_module, _input, output):
                # v_proj's OUTPUT is V before reshape/permute/repeat_kv.
                # Gradient here is the last point upstream of V's LoRA-B grad
                # ascent — if this is correct and LoRA-B is collapsed, the bug
                # is in the LoRA backward path.
                _retain_probe("v_proj_out", output)

            def _trap_qproj_output(_module, _input, output):
                _retain_probe("q_proj_out", output)

            def _trap_kproj_output(_module, _input, output):
                _retain_probe("k_proj_out", output)

            def _trap_mlp_gate_output(_module, _input, output):
                _retain_probe("mlp_gate_out", output)

            def _trap_mlp_up_output(_module, _input, output):
                _retain_probe("mlp_up_out", output)

            def _trap_downproj_input(_module, module_args):
                _retain_probe("mlp_inner", module_args[0])

            def _trap_downproj_output(_module, _input, output):
                _retain_probe("mlp_out", output)

            hooks.append(probe_layer.self_attn.o_proj.register_forward_pre_hook(_trap_oproj_input))
            hooks.append(probe_layer.self_attn.o_proj.register_forward_hook(_trap_oproj_output))
            hooks.append(probe_layer.post_attention_layernorm.register_forward_pre_hook(_trap_post_attn_input))
            hooks.append(probe_layer.post_attention_layernorm.register_forward_hook(_trap_post_attn_output))
            hooks.append(probe_layer.register_forward_hook(_trap_layer_output))
            hooks.append(probe_layer.self_attn.q_proj.register_forward_hook(_trap_qproj_output))
            hooks.append(probe_layer.self_attn.k_proj.register_forward_hook(_trap_kproj_output))
            hooks.append(probe_layer.self_attn.v_proj.register_forward_hook(_trap_vproj_output))
            hooks.append(probe_layer.mlp.gate_proj.register_forward_hook(_trap_mlp_gate_output))
            hooks.append(probe_layer.mlp.up_proj.register_forward_hook(_trap_mlp_up_output))
            hooks.append(probe_layer.mlp.down_proj.register_forward_pre_hook(_trap_downproj_input))
            hooks.append(probe_layer.mlp.down_proj.register_forward_hook(_trap_downproj_output))

    return_layers = (
        list(range(model.config.text_config.num_hidden_layers))
        if args.dump_layers
        else None
    )
    grad_ctx = torch.enable_grad() if args.lora_step else torch.no_grad()
    try:
        with grad_ctx, torch.autocast(device.type, dtype=dtype):
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                vinputs=noisy,
                timestep=t_pixeldit.reshape(-1),
                token_types=token_types,
                use_flash_attn=args.use_flash_attn,
                return_mid_results_layers=return_layers,
            )
    finally:
        for hook in hooks:
            hook.remove()
    x_pred_full = outputs.x_pred
    if args.dump_layers:
        mid_results = outputs.mid_results
        if mid_results is None:
            _die("dump_layers requested but model returned no mid_results")
        if len(mid_results) != model.config.text_config.num_hidden_layers:
            _die(
                "dump_layers expected "
                f"{model.config.text_config.num_hidden_layers} layers, "
                f"got {len(mid_results)}"
            )
        for i, h_state in enumerate(mid_results):
            layer_dump[f"hidden_layer_{i:02d}"] = (
                h_state.detach().float().contiguous().cpu()
            )
        for key in ("hidden_input_layer_00", "hidden_final_norm"):
            if key not in layer_dump:
                _die(f"dump_layers failed to capture {key}")
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            pre_text_emb = model.model.get_input_embeddings()(input_ids)
            # bisect: capture the sinusoid output (pre-MLP, pre-BF16-cast) so we can
            # tell whether divergence is in the sinusoid build or in the 2-layer MLP.
            _t_e = model.model.t_embedder1
            _t_in = t_pixeldit.reshape(-1).to(device)
            pre_t_freq_fp32 = _t_e.timestep_embedding(_t_in * 1000, _t_e.frequency_embedding_size)
            pre_t_freq_bf16 = pre_t_freq_fp32.to(_t_e.mlp[0].weight.dtype)
            # Step-through the 2-layer MLP to isolate which step introduces drift.
            # mlp[0] = Linear, mlp[1] = SiLU, mlp[2] = Linear.
            pre_t_mlp0_out = _t_e.mlp[0](pre_t_freq_bf16)
            pre_t_silu_out = _t_e.mlp[1](pre_t_mlp0_out)
            # Bias-add isolation: run the second Linear's GEMM WITHOUT bias to
            # separate the bias-add precision from the GEMM precision. Rust does
            # bias via cuBLASLt EpilogueBias (fused with GEMM); PyTorch nn.Linear
            # uses addmm which may quantize differently. Comparing no-bias outputs
            # tells us whether the kernel itself diverges or only the epilogue.
            pre_t_mlp2_nobias = torch.nn.functional.linear(
                pre_t_silu_out, _t_e.mlp[2].weight, None
            )
            # Manual BF16-only bias add. If pre_t_mlp2_manual_bias == pre_t_emb,
            # then PyTorch's F.linear adds bias outside the GEMM (BF16 precision
            # only), and the Rust cuBLASLt EpilogueBias (which adds in FP32 inside
            # the GEMM accumulator) is the source of the 1-ULP rounding delta.
            pre_t_mlp2_manual_bias = pre_t_mlp2_nobias + _t_e.mlp[2].bias
            pre_t_emb = model.model.t_embedder1(t_pixeldit.reshape(-1).to(device))
            pre_tms_mask = input_ids == model.model.tms_token_id
            pre_tms_mask_3d = pre_tms_mask.unsqueeze(-1).expand_as(pre_text_emb)
            pre_t_emb_expanded = pre_t_emb.unsqueeze(1).expand_as(pre_text_emb)
            pre_text_emb_with_t = torch.where(
                pre_tms_mask_3d,
                pre_t_emb_expanded,
                pre_text_emb,
            )
            pre_patch_proj1 = model.model.x_embedder.proj1(noisy)
            pre_patch_proj2_nobias = torch.nn.functional.linear(
                pre_patch_proj1, model.model.x_embedder.proj2.weight, None
            )
            pre_patch_proj2_manual_bias = (
                pre_patch_proj2_nobias + model.model.x_embedder.proj2.bias
            )
            pre_patch_emb = model.model.x_embedder(noisy).to(pre_text_emb.dtype)
            pre_inputs_embeds = torch.cat(
                [pre_text_emb_with_t, pre_patch_emb],
                dim=1,
            )
            text_model = model.model.language_model
            layer00 = text_model.layers[0]
            attn00 = layer00.self_attn
            position_embeddings00 = text_model.rotary_emb(pre_inputs_embeds, position_ids)
            cos00, sin00 = position_embeddings00
            layer00_cos_half = cos00[..., : attn00.head_dim // 2].contiguous()
            layer00_sin_half = sin00[..., : attn00.head_dim // 2].contiguous()
            idx_ar00 = torch.nonzero(~token_types[0].bool(), as_tuple=False).squeeze(-1)
            input_shape00 = pre_inputs_embeds.shape[:-1]
            hidden_shape_q00 = (*input_shape00, -1, attn00.head_dim)

            layer00_normed = layer00.input_layernorm(pre_inputs_embeds)
            layer00_q_proj = attn00.q_proj(layer00_normed)
            layer00_k_proj = attn00.k_proj(layer00_normed)
            layer00_v_proj = attn00.v_proj(layer00_normed)
            layer00_q_heads = layer00_q_proj.view(hidden_shape_q00).transpose(1, 2).contiguous()
            layer00_k_heads = layer00_k_proj.view(hidden_shape_q00).transpose(1, 2).contiguous()
            layer00_v_heads = layer00_v_proj.view(hidden_shape_q00).transpose(1, 2).contiguous()
            def _rms_unit(x, eps):
                y = x.to(torch.float32)
                variance = y.pow(2).mean(-1, keepdim=True)
                y = y * torch.rsqrt(variance + eps)
                return y.to(x.dtype)
            def _rms_inv(x, eps):
                y = x.to(torch.float32)
                variance = y.pow(2).mean(-1, keepdim=True)
                return torch.rsqrt(variance + eps)
            def _rms_mean_sq(x):
                y = x.to(torch.float32)
                return y.pow(2).mean(-1, keepdim=True)
            layer00_q_mean_sq = _rms_mean_sq(
                layer00_q_proj.view(hidden_shape_q00),
            ).squeeze(-1).transpose(1, 2).contiguous()
            layer00_k_mean_sq = _rms_mean_sq(
                layer00_k_proj.view(hidden_shape_q00),
            ).squeeze(-1).transpose(1, 2).contiguous()
            layer00_q_inv = _rms_inv(
                layer00_q_proj.view(hidden_shape_q00),
                attn00.q_norm.variance_epsilon,
            ).squeeze(-1).transpose(1, 2).contiguous()
            layer00_k_inv = _rms_inv(
                layer00_k_proj.view(hidden_shape_q00),
                attn00.k_norm.variance_epsilon,
            ).squeeze(-1).transpose(1, 2).contiguous()
            layer00_q_unit = _rms_unit(
                layer00_q_proj.view(hidden_shape_q00),
                attn00.q_norm.variance_epsilon,
            ).transpose(1, 2).contiguous()
            layer00_k_unit = _rms_unit(
                layer00_k_proj.view(hidden_shape_q00),
                attn00.k_norm.variance_epsilon,
            ).transpose(1, 2).contiguous()
            layer00_q_normed = attn00.q_norm(
                layer00_q_proj.view(hidden_shape_q00)
            ).transpose(1, 2).contiguous()
            layer00_k_normed = attn00.k_norm(
                layer00_k_proj.view(hidden_shape_q00)
            ).transpose(1, 2).contiguous()
            layer00_q_rope, layer00_k_rope = apply_rotary_pos_emb(
                layer00_q_normed, layer00_k_normed, cos00, sin00
            )
            layer00_q_rope = layer00_q_rope.contiguous()
            layer00_k_rope = layer00_k_rope.contiguous()
            layer00_k_repeat = repeat_kv(
                layer00_k_rope, attn00.num_key_value_groups
            ).contiguous()
            layer00_v_repeat = repeat_kv(
                layer00_v_heads, attn00.num_key_value_groups
            ).contiguous()

            attention_interface00 = eager_attention_forward
            if attn00.config._attn_implementation != "eager":
                attention_interface00 = ALL_ATTENTION_FUNCTIONS[
                    attn00.config._attn_implementation
                ]
            layer00_out_ar, _ = attention_interface00(
                attn00,
                layer00_q_rope[:, :, idx_ar00].contiguous(),
                layer00_k_rope[:, :, idx_ar00].contiguous(),
                layer00_v_heads[:, :, idx_ar00].contiguous(),
                attention_mask=None,
                dropout=0.0,
                scaling=attn00.head_dim**-0.5,
                is_causal=True,
            )
            layer00_out_full, _ = attention_interface00(
                attn00,
                layer00_q_rope,
                layer00_k_rope,
                layer00_v_heads,
                attention_mask=None,
                dropout=0.0,
                scaling=attn00.head_dim**-0.5,
                is_causal=False,
            )
            layer00_out_full = layer00_out_full.clone()
            layer00_out_full[:, idx_ar00] = layer00_out_ar
            layer00_sdpa_out = layer00_out_full.transpose(1, 2).contiguous()
            layer00_o_proj_in = layer00_out_full.reshape(*input_shape00, -1).contiguous()
            layer00_attn_out = attn00.o_proj(layer00_o_proj_in)
            layer00_after_attn = pre_inputs_embeds + layer00_attn_out
            layer00_normed2 = layer00.post_attention_layernorm(layer00_after_attn)
            layer00_gate = layer00.mlp.gate_proj(layer00_normed2)
            layer00_up = layer00.mlp.up_proj(layer00_normed2)
            layer00_mlp_inner = layer00.mlp.act_fn(layer00_gate) * layer00_up
            layer00_mlp_out = layer00.mlp.down_proj(layer00_mlp_inner)
            layer00_hidden_out = layer00_after_attn + layer00_mlp_out

            def _replay_probe_layer(layer_idx: int, hidden_in):
                layer = text_model.layers[layer_idx]
                attn = layer.self_attn
                position_embeddings = text_model.rotary_emb(hidden_in, position_ids)
                cos, sin = position_embeddings
                cos_half = cos[..., : attn.head_dim // 2].contiguous()
                sin_half = sin[..., : attn.head_dim // 2].contiguous()
                idx_ar = torch.nonzero(~token_types[0].bool(), as_tuple=False).squeeze(-1)
                input_shape = hidden_in.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)

                normed = layer.input_layernorm(hidden_in)
                q_proj = attn.q_proj(normed)
                k_proj = attn.k_proj(normed)
                v_proj = attn.v_proj(normed)
                q_heads = q_proj.view(hidden_shape).transpose(1, 2).contiguous()
                k_heads = k_proj.view(hidden_shape).transpose(1, 2).contiguous()
                v_heads = v_proj.view(hidden_shape).transpose(1, 2).contiguous()
                q_mean_sq = _rms_mean_sq(q_proj.view(hidden_shape)).squeeze(-1).transpose(1, 2).contiguous()
                k_mean_sq = _rms_mean_sq(k_proj.view(hidden_shape)).squeeze(-1).transpose(1, 2).contiguous()
                q_inv = _rms_inv(
                    q_proj.view(hidden_shape),
                    attn.q_norm.variance_epsilon,
                ).squeeze(-1).transpose(1, 2).contiguous()
                k_inv = _rms_inv(
                    k_proj.view(hidden_shape),
                    attn.k_norm.variance_epsilon,
                ).squeeze(-1).transpose(1, 2).contiguous()
                q_unit = _rms_unit(
                    q_proj.view(hidden_shape),
                    attn.q_norm.variance_epsilon,
                ).transpose(1, 2).contiguous()
                k_unit = _rms_unit(
                    k_proj.view(hidden_shape),
                    attn.k_norm.variance_epsilon,
                ).transpose(1, 2).contiguous()
                q_normed = attn.q_norm(q_proj.view(hidden_shape)).transpose(1, 2).contiguous()
                k_normed = attn.k_norm(k_proj.view(hidden_shape)).transpose(1, 2).contiguous()
                q_rope, k_rope = apply_rotary_pos_emb(q_normed, k_normed, cos, sin)
                q_rope = q_rope.contiguous()
                k_rope = k_rope.contiguous()
                k_repeat = repeat_kv(k_rope, attn.num_key_value_groups).contiguous()
                v_repeat = repeat_kv(v_heads, attn.num_key_value_groups).contiguous()

                attention_interface = eager_attention_forward
                if attn.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        attn.config._attn_implementation
                    ]
                out_ar, _ = attention_interface(
                    attn,
                    q_rope[:, :, idx_ar].contiguous(),
                    k_rope[:, :, idx_ar].contiguous(),
                    v_heads[:, :, idx_ar].contiguous(),
                    attention_mask=None,
                    dropout=0.0,
                    scaling=attn.head_dim**-0.5,
                    is_causal=True,
                )
                out_full, _ = attention_interface(
                    attn,
                    q_rope,
                    k_rope,
                    v_heads,
                    attention_mask=None,
                    dropout=0.0,
                    scaling=attn.head_dim**-0.5,
                    is_causal=False,
                )
                out_full = out_full.clone()
                out_full[:, idx_ar] = out_ar
                sdpa_out = out_full.transpose(1, 2).contiguous()
                o_proj_in = out_full.reshape(*input_shape, -1).contiguous()
                attn_out = attn.o_proj(o_proj_in)
                after_attn = hidden_in + attn_out
                normed2 = layer.post_attention_layernorm(after_attn)
                gate = layer.mlp.gate_proj(normed2)
                up = layer.mlp.up_proj(normed2)
                mlp_inner = layer.mlp.act_fn(gate) * up
                mlp_out = layer.mlp.down_proj(mlp_inner)
                hidden_out = after_attn + mlp_out
                prefix = f"layer{layer_idx:02}"
                return {
                    f"{prefix}.normed": normed,
                    f"{prefix}.q_proj": q_proj,
                    f"{prefix}.k_proj": k_proj,
                    f"{prefix}.v_proj": v_proj,
                    f"{prefix}.q_heads": q_heads,
                    f"{prefix}.k_heads": k_heads,
                    f"{prefix}.v_heads": v_heads,
                    f"{prefix}.cos_half": cos_half,
                    f"{prefix}.sin_half": sin_half,
                    f"{prefix}.q_mean_sq": q_mean_sq,
                    f"{prefix}.k_mean_sq": k_mean_sq,
                    f"{prefix}.q_inv": q_inv,
                    f"{prefix}.k_inv": k_inv,
                    f"{prefix}.q_unit": q_unit,
                    f"{prefix}.k_unit": k_unit,
                    f"{prefix}.q_normed": q_normed,
                    f"{prefix}.k_normed": k_normed,
                    f"{prefix}.q_rope": q_rope,
                    f"{prefix}.k_rope": k_rope,
                    f"{prefix}.k_repeat": k_repeat,
                    f"{prefix}.v_repeat": v_repeat,
                    f"{prefix}.sdpa_out": sdpa_out,
                    f"{prefix}.o_proj_in": o_proj_in,
                    f"{prefix}.attn_out": attn_out,
                    f"{prefix}.after_attn": after_attn,
                    f"{prefix}.normed2": normed2,
                    f"{prefix}.gate": gate,
                    f"{prefix}.up": up,
                    f"{prefix}.mlp_inner": mlp_inner,
                    f"{prefix}.mlp_out": mlp_out,
                    f"{prefix}.hidden_out": hidden_out,
                }

            probe_layers = _parse_probe_layers(
                args.probe_layers,
                model.config.text_config.num_hidden_layers,
            )
            for probe_layer_idx in probe_layers:
                if probe_layer_idx == 0:
                    continue
                hidden_in = mid_results[probe_layer_idx - 1].detach()
                for key, tensor in _replay_probe_layer(probe_layer_idx, hidden_in).items():
                    layer_dump[key] = tensor.detach().float().contiguous().cpu()
        layer_dump["pre.text_emb"] = pre_text_emb.detach().float().contiguous().cpu()
        layer_dump["pre.t_freq_fp32"] = pre_t_freq_fp32.detach().float().contiguous().cpu()
        layer_dump["pre.t_freq_bf16"] = pre_t_freq_bf16.detach().float().contiguous().cpu()
        layer_dump["pre.t_mlp0_out"] = pre_t_mlp0_out.detach().float().contiguous().cpu()
        layer_dump["pre.t_silu_out"] = pre_t_silu_out.detach().float().contiguous().cpu()
        layer_dump["pre.t_mlp2_nobias"] = pre_t_mlp2_nobias.detach().float().contiguous().cpu()
        layer_dump["pre.t_mlp2_manual_bias"] = pre_t_mlp2_manual_bias.detach().float().contiguous().cpu()
        layer_dump["pre.t_emb"] = pre_t_emb.detach().float().contiguous().cpu()
        layer_dump["pre.text_emb_with_t"] = (
            pre_text_emb_with_t.detach().float().contiguous().cpu()
        )
        layer_dump["pre.patch_proj1"] = pre_patch_proj1.detach().float().contiguous().cpu()
        layer_dump["pre.patch_proj2_nobias"] = (
            pre_patch_proj2_nobias.detach().float().contiguous().cpu()
        )
        layer_dump["pre.patch_proj2_manual_bias"] = (
            pre_patch_proj2_manual_bias.detach().float().contiguous().cpu()
        )
        layer_dump["pre.patch_emb"] = pre_patch_emb.detach().float().contiguous().cpu()
        layer_dump["pre.inputs_embeds"] = (
            pre_inputs_embeds.detach().float().contiguous().cpu()
        )
        for key, tensor in {
            "layer00.normed": layer00_normed,
            "layer00.q_proj": layer00_q_proj,
            "layer00.k_proj": layer00_k_proj,
            "layer00.v_proj": layer00_v_proj,
            "layer00.q_heads": layer00_q_heads,
            "layer00.k_heads": layer00_k_heads,
            "layer00.v_heads": layer00_v_heads,
            "layer00.cos_half": layer00_cos_half,
            "layer00.sin_half": layer00_sin_half,
            "layer00.q_mean_sq": layer00_q_mean_sq,
            "layer00.k_mean_sq": layer00_k_mean_sq,
            "layer00.q_inv": layer00_q_inv,
            "layer00.k_inv": layer00_k_inv,
            "layer00.q_unit": layer00_q_unit,
            "layer00.k_unit": layer00_k_unit,
            "layer00.q_normed": layer00_q_normed,
            "layer00.k_normed": layer00_k_normed,
            "layer00.q_rope": layer00_q_rope,
            "layer00.k_rope": layer00_k_rope,
            "layer00.k_repeat": layer00_k_repeat,
            "layer00.v_repeat": layer00_v_repeat,
            "layer00.sdpa_out": layer00_sdpa_out,
            "layer00.o_proj_in": layer00_o_proj_in,
            "layer00.attn_out": layer00_attn_out,
            "layer00.after_attn": layer00_after_attn,
            "layer00.normed2": layer00_normed2,
            "layer00.gate": layer00_gate,
            "layer00.up": layer00_up,
            "layer00.mlp_inner": layer00_mlp_inner,
            "layer00.mlp_out": layer00_mlp_out,
            "layer00.hidden_out": layer00_hidden_out,
        }.items():
            layer_dump[key] = tensor.detach().float().contiguous().cpu()
    if args.lora_step:
        x_pred_full.retain_grad()
    x_pred_rows = x_pred_full[0, vinput_mask[0]].unsqueeze(0)
    if args.lora_step:
        x_pred_rows.retain_grad()
    pred_velocity = ((noisy.float() - x_pred_rows.float()) / sigma).to(dtype).float()
    if args.lora_step:
        pred_velocity.retain_grad()
    loss_velocity = torch.nn.functional.mse_loss(
        pred_velocity,
        target_velocity.float(),
        reduction="mean",
    )
    print(f"[train-step-ref] forward done in {time.time() - t1:.2f}s")
    print(f"[train-step-ref] loss_velocity = {loss_velocity.item():.9f}")

    global_grad_norm_pre = None
    clip_scale = None
    if args.lora_step:
        if args.deterministic_attn:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
        print("[train-step-ref] backward + clip + AdamW8bit step ...")
        optimizer.zero_grad(set_to_none=True)
        loss_velocity.backward()
        lora_save["grad_mid.x_pred_full"] = _clone_or_zero_tensor_grad(x_pred_full)
        lora_save["grad_mid.x_pred_rows"] = _clone_or_zero_tensor_grad(x_pred_rows)
        lora_save["grad_mid.pred_velocity"] = _clone_or_zero_tensor_grad(pred_velocity)
        # Soul.md trap: probe-layer attention backward chain probes.
        probe_layer_idx = int(
            os.environ.get(
                "HIDREAM_BWD_PROBE_LAYER",
                str(model.config.text_config.num_hidden_layers - 1),
            )
        )
        probe_layer_idx_str = f"{probe_layer_idx:02d}"
        for probe_name, probe_tensor in layer0_probes.items():
            lora_save[f"grad_probe.layers.{probe_layer_idx_str}.{probe_name}"] = (
                _clone_or_zero_tensor_grad(probe_tensor)
            )
        for name, param in lora_named.items():
            lora_save[f"grad_pre.{name}"] = _clone_or_zero_grad(param)
        global_grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            lora_params,
            args.max_grad_norm,
        )
        global_grad_norm_pre_f = float(global_grad_norm_pre.detach().float().cpu().item())
        clip_scale = min(1.0, args.max_grad_norm / (global_grad_norm_pre_f + 1.0e-6))
        for name, param in lora_named.items():
            lora_save[f"grad_post.{name}"] = _clone_or_zero_grad(param)
        optimizer.step()
        _save_lora_tensor_set(lora_save, "post", lora_named)
        lora_save["global_grad_norm_pre"] = torch.tensor(
            [global_grad_norm_pre_f],
            dtype=torch.float32,
        )
        lora_save["clip_scale"] = torch.tensor([clip_scale], dtype=torch.float32)
        lora_save["adamw_lr"] = torch.tensor([args.lr], dtype=torch.float32)
        lora_save["adamw_weight_decay"] = torch.tensor(
            [args.weight_decay],
            dtype=torch.float32,
        )
        lora_save["loss_velocity"] = loss_velocity.reshape(1).detach().float().cpu()
        lora_meta = {
            "lora_seed": args.lora_seed,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "adapter_count": LORA_ADAPTERS,
            "tensor_count": len(lora_named),
            "max_grad_norm": args.max_grad_norm,
            "global_grad_norm_pre": global_grad_norm_pre_f,
            "clip_scale": clip_scale,
            "adamw_lr": args.lr,
            "adamw_eps": ADAMW_EPS,
            "adamw_weight_decay": args.weight_decay,
        }
        print(
            "[train-step-ref] grad_norm_pre = "
            f"{global_grad_norm_pre_f:.9f}, clip_scale = {clip_scale:.9f}"
        )

    if x_pred_rows.shape != target_velocity.shape:
        _die(
            "x_pred_rows/target shape mismatch: "
            f"{tuple(x_pred_rows.shape)} vs {tuple(target_velocity.shape)}"
        )

    image_grid = sample.get("image_grid")
    to_save = {
        "patches": patches.float().contiguous().cpu(),
        "noise": noise.float().contiguous().cpu(),
        "scaled_noise": scaled_noise.float().contiguous().cpu(),
        "noisy": noisy.float().contiguous().cpu(),
        "target_velocity": target_velocity.float().contiguous().cpu(),
        "input_ids": input_ids.float().contiguous().cpu(),
        "position_ids": position_ids.squeeze(1).float().contiguous().cpu(),
        "vinput_mask": vinput_mask.float().contiguous().cpu(),
        "token_types": token_types.float().contiguous().cpu(),
        "t_scalar": t_scalar.cpu(),
        "timestep": t_pixeldit.cpu(),
        "x_pred_full": x_pred_full.float().contiguous().cpu(),
        "x_pred_rows": x_pred_rows.float().contiguous().cpu(),
        "pred_velocity": pred_velocity.contiguous().cpu(),
        "loss_velocity": loss_velocity.reshape(1).float().cpu(),
    }
    to_save.update(layer_dump)
    if image_grid is not None:
        to_save["image_grid"] = image_grid.float().contiguous().cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(to_save, str(out_path))
    meta = {
        "model_path": args.model_path,
        "reference_root": repo_root,
        "sample_path": str(sample_path),
        "seed": args.seed,
        "dtype": DTYPE_NAME,
        "noise_scale": NOISE_SCALE,
        "t_eps": T_EPS,
        "t_scalar": args.t_scalar,
        "t_pixeldit": 1.0 - args.t_scalar,
        "use_flash_attn": args.use_flash_attn,
        "dump_layers": args.dump_layers,
        "patches_shape": list(patches.shape),
        "sequence_shape": list(token_types.shape),
        "loss_velocity": float(loss_velocity.item()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[train-step-ref] wrote {out_path}")
    print(f"[train-step-ref] wrote {meta_path}")
    if args.lora_step:
        lora_out_path = Path(args.lora_out)
        lora_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_save, str(lora_out_path))
        lora_meta_path = lora_out_path.with_suffix(".json")
        lora_meta_path.write_text(json.dumps(lora_meta, indent=2))
        print(f"[train-step-ref] wrote {lora_out_path}")
        print(f"[train-step-ref] wrote {lora_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
