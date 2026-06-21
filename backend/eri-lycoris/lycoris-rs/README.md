# lycoris-rs

Rust port of [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) (advanced LoRA-family algorithms for diffusion models) by [KohakuBlueleaf](https://github.com/KohakuBlueleaf), targeting [flame-core](https://github.com/CodeAlexx/Flame) for inference-time weight merging.

**Inference-only**, **weight-merge mode** — load a LyCORIS safetensors, materialize ΔW per adapter, add into the base model's weights once, then run normal inference. No training paths.

## Algorithms shipped (v1)

| Algo | Linear | Conv2d | Conv2d Tucker |
|---|---|---|---|
| **LoCon** (LoRA-for-Conv) | ✓ | ✓ | ✓ |
| **LoHa** (Hadamard) | ✓ | ✓ | ✓ |
| **LoKr** (Kronecker) | ✓ | ✓ | ✓ |
| **Full** (`.diff` + `.diff_b` bias delta) | ✓ | ✓ | — |

Out of scope for v1: IA3, BOFT, Diag-OFT, DyLoRA, GLoRA, TLoRA, norm scaling.

## Public API

```rust
use lycoris_rs::{LycorisCollection, LycorisAdapter};

// Load + auto-detect adapter types from key suffixes
let coll = LycorisCollection::load(Path::new("my_lora.safetensors"), device)?;

// Apply ΔW into a HashMap<String, Tensor> of base weights, scaled by `strength`.
// `name_mapper` converts each adapter's Kohya prefix to a key in `weights`.
coll.apply_to(&mut weights, /*strength=*/ 1.0, |kohya_prefix| {
    // e.g. flux_kohya_to_flame from inference-flame::lycoris
    Some(format!("{}.weight", kohya_prefix.replace('_', ".")))
})?;
```

For per-model name mappers (FLUX / Z-Image / Chroma / Klein / Qwen-Image / SDXL / SD 1.5) and the `fuse_split_qkv` helper that handles fused-QKV models, see [`inference-flame/src/lycoris.rs`](https://github.com/CodeAlexx/inference-flame/blob/master/src/lycoris.rs).

## Auto-detection

Loader inspects each adapter's suffixes to pick the algorithm:

| Pattern | Algorithm |
|---|---|
| `.lora_up.weight`, `.lora_down.weight`, optional `.lora_mid.weight` | LoCon |
| `.hada_w1_a/b`, `.hada_w2_a/b`, optional `.hada_t1/t2` | LoHa |
| `.lokr_w1` (or `.lokr_w1_a + .lokr_w1_b`) and same for w2, optional `.lokr_t2` | LoKr |
| `.diff`, optional `.diff_b` | Full |
| `.dora_scale` present | **loud-skip** with warning (DoRA correction not implemented) |

## Validated against real LoRAs

Real Z-Image LoKr at `/home/alex/.serenity/models/loras/zimageLokrEri_000002250.safetensors` (240 adapters with split-QKV `_to_q/_to_k/_to_v`):
- 240 split adapters loaded
- 30 QKV triples fused via `inference-flame::lycoris::fuse_split_qkv` → 180 adapters remain (exact: 240 − 2×30)
- Sample mapping `lycoris_layers_8_attention_qkv` → `layers.8.attention.qkv.weight`

## Parity vs upstream Python

`tests/parity_dump.py` generates reference ΔW via upstream `lycoris.functional.{locon,loha,lokr}.diff_weight`; `tests/parity.rs` loads and compares element-wise. 3 algorithms validated (LoCon Linear, LoHa Linear, Full); LoKr Linear-dense path skipped (Rust `resolve_w2_full_ok_in_kh_kw` expects conv-shaped W2 — to be expanded).

## Build

```bash
LD_LIBRARY_PATH=/path/to/libtorch/lib cargo build --release
LD_LIBRARY_PATH=/path/to/libtorch/lib cargo test --release
# Parity (requires upstream Python LyCORIS at /home/alex/lycoris-upstream-lib):
python3 tests/parity_dump.py && cargo test --release --test parity
```

## Tests
- 13 lib unit tests
- 11 smoke (LoCon/LoHa/LoKr/Full Linear+Conv hand-checked)
- 4 loader_smoke (multi-adapter round-trip, Tucker LoHa detection, DoRA loud-skip, Full diff_b)
- 3 parity vs Python (LoCon, LoHa, Full Linear)

## Salvage history

This crate was a year-old prototype that drifted from current flame-core API. Salvaged 2026-04-20 via a builder → skeptic → bug-fixer agent pipeline:
- Deleted ~385 LoC of dead BF16 kernels (flame-core has native BF16 ops now)
- Fixed `Tensor::index_put` → `Tensor::stack`-based scatter for Tucker
- Skeptic flagged 7 P0s (LoCon Tucker dispatch missing, LoHa Tucker unimplemented, LoKr full-W1+W2 zero-ΔW, etc.) — all fixed

## Known limitations / v2 work

- DoRA adapters loud-skip (warning) instead of merging with DoRA correction
- LoKr Linear-dense W2 needs a small Rust API extension (resolve_w2 expects conv shape)
- Dynamic forward-time application (runtime enable/disable, strength slider) — v1 is merge-only
- BFL Chroma LoRAs (with `diffusion_model.` prefix + PEFT `lora_A/B` style) need a separate translation layer (out of scope; not Kohya format)

## Credits

Original LyCORIS: [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).
This Rust port keeps the math; ships the loader + per-model integration around it.

## License

MIT.
