# SKEPTIC review of `lycoris-rs/` (salvaged Rust port)

Reviewer: skeptic. No code changes — findings only. Citations are `file:line`.

## P0 — correctness-breaking

### P0-1. LoCon Tucker conv path is hard-failed (`src/algorithms/locon.rs:319-365`)
The Tucker reconstruction body computes the per-spatial-position `down @ mid_hw @ up` matmuls correctly (lines 350-360), but the loop body never collects/writes the result, then unconditionally returns `Err(Error::InvalidOperation("Tucker conv decomposition requires full tensor contraction implementation"))` at line 363-365.

Consequence: any LyCORIS file containing `lora_mid.weight` (Tucker LoCon, default for non-1×1 conv when `use_tucker=True` per upstream `modules/locon.py:85-90`) will be loaded by `loader.rs:317-326` and then **crash the entire `apply_collection` call** when `delta_weight()` is invoked. Not silent corruption, but loud P0 — every adapter in the file fails because of one Tucker entry.

The fix is trivial — `ops::tucker::rebuild_conv_tucker` (in `src/ops/tucker.rs:105-178`) already implements exactly this contraction with the stack-based assembly. LoCon never dispatches to it.

### P0-2. Tucker LoHa path is unimplemented (`src/ops/hadamard.rs:122-153`)
`make_hadamard_weight_tucker` validates inputs then returns `Err`. Triggered by any LoHa adapter with `hada_t1`/`hada_t2` (Python upstream `functional/loha.py:97-105` Tucker init).

Same severity rationale as P0-1: an entire `apply_collection` errors out for the file.

### P0-3. Tucker LoHa loader treats it as Linear (`src/loader.rs:362-366`)
For Tucker LoHa, upstream Python (`functional/loha.py:98-103`) saves `w1d/w1u/w2d/w2u` as **2D** `[rank, in_dim]` / `[rank, out_dim]`, with the spatial kernel split into 4D `t1`/`t2`. The Rust loader infers `is_conv` purely from `w1a.dims().len()` (line 363). For Tucker LoHa, `w1a` is 2D, so `is_conv = false`. The loader builds a Linear LoHaModule with `t1`/`t2` dropped on the floor (the `if !is_conv` branch at line 370-378 doesn't carry `t1`/`t2`). The 4D Tucker cores are silently discarded; the Linear math runs on 2D w1a/w1b ignoring the spatial dimension entirely. **Silent corruption** — no error, wrong ΔW.

Even if the loader correctly detected Tucker, P0-2 would fire next.

### P0-4. LoKr full-W1 + full-W2 produces zero ΔW (`src/loader.rs:498-507`, `src/algorithms/lokr.rs:21-23`)
Rank inference walks `w1a`, `w1b`, `w2a`, `w2b`, `t2` and returns `0` if none are present. With both `w1` and `w2` full (a legal Python configuration — see upstream `modules/lokr.py:209-211` which forces `alpha = lora_dim` for this case), `rank = 0`. Then `scale_from(alpha, 0) = 0.0` (`lokr.rs:23`).

`make_kronecker` early-exits to a zeros tensor when `scale == 0.0` (`ops/kronecker.rs:48-55`).

Consequence: any LoKr adapter that didn't factorize either side (Python `decompose_both=False && full_matrix=True`, or `rank >= dim/2`) silently merges as **zero** — adapter is fully no-op without any warning. Rare in the wild for SDXL LoRAs, but supported and saved by upstream when full ranks are used.

The fix: when neither side is factorized and `.alpha` is present, use `gamma / gamma = 1.0` (Python's behaviour — see `functional/lokr.py:139-141`).

### P0-5. LoKr Tucker conv has w2a/w2b axis mismatch (`src/algorithms/lokr.rs:81-89`)
For Tucker LoKr conv, upstream Python (`functional/lokr.py:73-74`) saves:
```
w2a = torch.empty(rank, shape[0][1])     # [R, OK]
w2b = torch.empty(rank, shape[1][1])     # [R, IN]
```
For non-Tucker conv (`functional/lokr.py:77-78`):
```
w2a = torch.empty(shape[0][1], rank)     # [OK, R]
w2b = torch.empty(rank, shape[1][1], *k) # [R, IN, KH, KW]
```
**The order of `rank` in `w2a` flips between Tucker and non-Tucker.** Rust always reads `ok = w2a.dims()[0]; r = w2a.dims()[1]` (lokr.rs:81-82), which is correct for non-Tucker but wrong for Tucker (where `w2a` is `[R, OK]`).

Consequence: silent corruption on Tucker LoKr. Output dims would still match (R ≈ OK in many cases), but the contraction reads `w2a` axes flipped. ΔW is wrong.

The reshape at `lokr.rs:88-89` then compounds: `w2a.reshape(&[1,1, r, ok])` — if w2a is actually `[R, OK]` and you label it as `[OK, R]`, the reshape to `[1, 1, R, OK]` is a no-op interpretation but the data semantics still mismatch what `rebuild_conv_tucker` expects (which is `up: [1,1,R,OC]` — and OC=OK here, so it might accidentally work if the flipped interpretation happens to be self-consistent). Same for `w2b.reshape(&[1,1, inn, r])` — `w2b` on disk is `[R, IN]`, reshape to `[1, 1, IN, R]` does NOT transpose; data ordering is corrupted.

Pure unverified-by-runtime, but the axis labelling in lokr.rs against the Python save format is demonstrably wrong for the Tucker branch.

### P0-6. DoRA adapters silently produce wrong ΔW (`src/loader.rs:135` + math missing)
`.dora_scale` is in `KNOWN_SUFFIXES` so it gets matched and dropped (the loader explicitly classifies it as "ignored"). For any LoCon/LoHa/LoKr trained with `weight_decompose=True`, upstream applies `apply_dora_scale` (`functional/general.py:95-108`) which renormalises the merged weight against the saved per-output-channel norm:
```
weight = (org + rebuild) / norm * dora_scale
diff = (weight - org) * scale
```
Rust merges `base + alpha/rank * down @ up`. **For DoRA LoRAs, this is silently wrong.** The error magnitude depends on how far `dora_scale` deviated from `||org||` — could be small, could be 10–50% off per-channel.

Public DoRA-trained LoRAs exist (released after LyCORIS 2.x). No detection, no warning.

### P0-7. Full adapter `.diff_b` (bias) silently dropped (`src/loader.rs:136`, `src/algorithms/full.rs`)
`.diff_b` is in `KNOWN_SUFFIXES` (treated as ignored). Upstream Full saves both `.diff` (weight delta) and `.diff_b` (bias delta) when the layer has a bias (`modules/full.py:128-132`). The Rust `FullAdapter` struct only carries `diff` (full.rs:9-12); `apply_collection` only mutates the weight tensor in `weights` map. Any `bias` entry corresponding to that layer is left untouched.

Consequence: bias-using layers (rare in attention QKV but common in projection-out, MLP, group-norm-replacement) lose the bias delta silently.

## P1 — likely bugs / edge cases / perf

### P1-1. `align_delta_to_base` element-count fallback is dangerous (`src/loader.rs:579-583`)
Final fallback reshapes `delta` to `base` shape iff element counts match, with no layout check. Any accidental product collision (e.g. Linear `[1024, 1024]` ↔ `[4, 262144]`, Conv `[3,3,512,512]` ↔ `[1,9,512,512]`) silently reshapes wrong.

In the current code path it's unreachable for properly-shaped LoCon/LoHa/LoKr/Full (cases 1-3 catch them), but the fallback exists, has no logging, and would be the failure mode for any future shape contract drift. Recommend deleting the fallback or at least logging when it fires.

### P1-2. 4D permute case is too permissive on symmetric dims (`src/loader.rs:564-569`)
Condition: `d[0]==b[2] && d[1]==b[3] && d[2]==b[1] && d[3]==b[0]`. For square kernels with `IC==OC` (very common in attention/projection blocks: KH=KW=1 or 3, IC=OC=512/1024), any 4D delta with those values matches regardless of what the four axes actually mean. Currently the lycoris-rs algorithms only emit `[KH,KW,IC,OC]`, so this is "safe in practice." If anyone ever feeds in a delta in PyTorch order `[OC,IC,KH,KW]` (which has the same dims when IC==OC and KH==KW), the permute would happily re-permute and produce garbage. No assertion against this.

### P1-3. Inconsistent ΔW layout convention between adapter types
- LoCon `get_diff_weight` (linear) returns `[IN, OUT]` (locon.rs:404-407).
- LoHa `get_diff_weight` (linear) returns `[IN, OUT]` (loha.rs:491-494).
- LoKr `get_diff_weight` (linear) returns `[OUT, IN]` because `make_kronecker(w1[OL,IM], w2[OK,IN])` packs to `[OL*OK, IM*IN] = [OUT, IN]` (lokr.rs:171, ops/kronecker.rs:70-72).

The doc comment at `lokr.rs:4` claims `→ [IN,OUT]` — incorrect. `align_delta_to_base` happens to absorb both via case-1 (no-op for LoKr) and case-2 (transpose for LoCon/LoHa) since flame-core base is `[OUT, IN]`. **It works by accident**, not by design. Anyone refactoring the alignment will break one or the other.

### P1-4. lib.rs and algorithm comments mislabel flame-core's weight conventions
`src/lib.rs:19`-`/` and several algorithm files claim flame-core uses `[IN, OUT]` for Linear and `[KH, KW, IC, OC]` for Conv2d. flame-core actually uses `[OUT, IN]` (`flame-core/src/linear.rs:36`) and `[OUT, IC, KH, KW]` (`flame-core/src/conv.rs:210-215`) — same as PyTorch.

The internal lycoris-rs algorithm modules use `[IN, OUT]` and `[KH, KW, IC, OC]` after on-load transpose. Net result is correct merging, but the comments cause future confusion. Cosmetic but high-cognitive-cost.

### P1-5. LoCon Tucker `mid` permute happens at load time but Tucker itself errors (`src/loader.rs:317-322`)
Loader unconditionally permutes the on-disk Tucker `mid`, but `delta_weight()` errors out before using it. Wasted work, and if the permute axis ordering itself is wrong (which I haven't independently verified against `[R_out, R_in, KH, KW]` Kohya format vs what locon.rs wants), the bug only surfaces if Tucker is later fixed.

### P1-6. LoKr 1×1 conv handled as Linear via fallback reshape (`src/algorithms/lokr.rs:486-495` is_conv detection + `src/loader.rs:579-583`)
`is_conv` for LoKr is `false` if `w2.dims()[2]==1 && w2.dims()[3]==1`. Linear path produces `[OUT, IN]` 2D delta. Then `align_delta_to_base` gets a 2D delta vs 4D base `[OC, IC, 1, 1]`, none of cases 1-3 match, falls through to case 4 (element-count match → reshape). It works because `OC*IC*1*1 == OUT*IN`, but it's the "dangerous" fallback (P1-1). At minimum should be a documented case 5.

### P1-7. BF16 tolerance `1e-2` is fine for 4×2 toy tests but too tight as inner_dim grows
At inner_dim=4, accumulated FMA error is bounded by ~1 ULP per term ≈ 4 * 7.8e-3 ≈ 3e-2 worst case for adversarial inputs, but the tests use ones (well-behaved). For real adapters with rank=8 and IN=3072 (Flux DiT), accumulated BF16 matmul drift can reach 5e-2 or more. The smoke tests don't exercise that regime; passing them does NOT validate large-shape correctness. Tolerance should scale `~ 1e-2 * sqrt(inner_dim)` or use FP32 reference for spot-check.

### P1-8. `tensor_utils::kronecker_product` does CPU round-trip (`src/tensor_utils.rs:84-127`)
`tensor.to_vec()` pulls to host, computes on CPU, uploads back. Unused (live LoKr uses `make_kronecker`), but exists. If anyone uses it on a large adapter, perf disaster.

### P1-9. `kaiming_uniform_bf16` uses normal distribution as approximation (`src/tensor_utils.rs:55-58`)
Trainer-only concern; inference-load doesn't init this way. Comment is honest. P2 if pure inference, P1 if anyone trains with this port.

### P1-10. LoKr Linear path requires both W2 and (W1 or factor) — pure-W1 LoKr errors (`src/algorithms/lokr.rs:167-170`)
Returns `InvalidOperation("missing W2 for linear LoKr")`. Unsure whether upstream has a "W1-only" mode; from the Python `weight_gen` it always produces a W2. Likely fine, but unverified. Loud failure not silent corruption.

## P2 — minor / cosmetic / possibly intentional

- `src/loader.rs:170-176` — `unknown` keys printed once with the first sample. Fine but noisy if a bunch unmatched.
- `src/algorithms/locon.rs:411-422` `merge_to` is a no-op stub. Confusing — should either implement or remove.
- `src/loader.rs:107` re-wraps `delta_weight()` errors with `prefix:` context — good. But `add()` errors (line 95-97) include only `prefix` not the dim info. Minor.
- `src/algorithms/lokr.rs:511` — `shape: ((0, 0), (0, 0))` populated with zero placeholders by the loader. No downstream use, but "unused" struct fields invite divergence.
- Unused module: `src/ops/conv2d.rs` is referenced from forward paths only; weight-merge `apply_to` doesn't need it. If pure-inference is the goal, the entire forward path could be cut.
- `src/tensor_utils.rs:46-50` — `fan_in` defaults to `dims[0]` for 1D tensors (questionable convention but unused for inference).
- LoCon test name says "linear" but no separate conv test for the simple 1×1 path exists — coverage gap.

## Test tautology assessment

**The tests are tight, not tautological.** Spot-checks pass exact values that would fail if the math were off:

- **LoCon (`tests/smoke.rs:60-91`)** — `down=[4,2]` ones, `up=[2,8]` with row 0 ones / row 1 zeros, alpha=2 rank=2 (scale=1). `(down @ up)[i,j] = 1*1+1*0 = 1`. Test asserts `1.0` at three positions. ✓ Hand-verified.
- **LoHa (`tests/smoke.rs:102-138`)** — `w1a=ones[4,2]`, `w1b=ones[2,4]` → w1=[4,4] all 2s. `w2a=ones[4,2]`, `w2b` row 0 ones / row 1 zeros → w2[i,j] = 1*1+1*0 = 1. Hadamard = 2*1 = 2. Scale = 1. Test asserts `2.0`. ✓ Hand-verified.
- **LoKr (`tests/smoke.rs:153-191`)** — `kron([[1,2],[3,4]], [[5,6],[7,8]])` spot-checks 5 elements at `[0,0]=5, [0,1]=6, [1,0]=7, [2,2]=20, [3,3]=32`. By the standard Kronecker definition `out[i*p+k, j*q+l] = A[i,j]*B[k,l]`: `[2,2]` → i=1,k=0,j=1,l=0 → 4*5 = 20 ✓. `[3,3]` → 4*8 = 32 ✓. Tight — a transposed Kronecker would fail at `[0,1]` vs `[1,0]`.
- **Full (`tests/smoke.rs:197-221`)** — trivial; 0.5 strength × identity. Tight.

What the tests **do not** exercise:
- Conv path (none of the four). 100% of Conv math (1×1 LoCon, spatial LoCon, Tucker LoCon, conv LoHa, conv LoKr, conv LoKr Tucker) is uncovered.
- The loader. No safetensors round-trip test.
- `align_delta_to_base` cases 2/3/4. Permute correctness, transpose correctness, element-count fallback safety — none exercised.
- DoRA, `.diff_b`, alpha-scalar parsing edge cases (empty tensor, BF16 .alpha file).
- Larger inner_dims where BF16 drift becomes meaningful.

So: tests prove the four narrow paths execute correctly. They prove nothing about loader, conv, Tucker, alignment, DoRA, or numerical fidelity at production sizes.

## Tucker decomposition coverage (real-world prevalence)

Sampled local LoRAs (4 representative files: SimpleTuner Flux, Wan-T2I, Qwen, Klein) — **zero** had `lora_mid`, `hada_t`, or `lokr_t` keys. All were vanilla LoCon (lora_up + lora_down + alpha). My grep across `~/`'s safetensors found no Tucker/LoHa/LoKr usage in the user's collection.

In the wider community: Tucker LoCon is the LyCORIS *default* for non-1×1 conv when `algo=loha/lokr/locon` with `use_tucker=True`, but the Kohya / SimpleTuner / civitai mainstream is overwhelmingly plain `algo=lora` (= 1×1 conv treated as linear) plus standard LoRA. SDXL/SD1.5 LoRAs are ~95%+ plain LoRA. Flux LoRAs likewise. LyCORIS variants are probably <5% of public LoRAs and Tucker even less of that.

**Practical risk ranking:**
- Vanilla LoCon (90%+ of real LoRAs): **OK** modulo P0-6 (DoRA) and P0-7 (.diff_b for full).
- LoHa Linear (small but real): **OK** for inference.
- LoKr Linear (smaller): **OK** modulo P0-4 (full-full edge).
- Tucker anything: **broken** but rare. ~1-3% of real LoRAs at most.
- DoRA-trained anything: **silently wrong**. Growing usage post-2024.

## `align_delta_to_base` risk review (`src/loader.rs:540-590`)

Cases:
1. **Exact match** (line 551-553) — safe, no-op.
2. **2D transpose** (line 557-561) — correct for LoCon/LoHa Linear ([IN,OUT] → [OUT,IN]). Also fires correctly for LoKr because LoKr emits [OUT,IN] which already matches and hits case 1 first. ✓
3. **4D permute** (line 564-569) — correct for [KH,KW,IC,OC] → [OC,IC,KH,KW]. **Caveat P1-2**: too permissive when IC==OC and KH==KW.
4. **4D reshape** (line 570-575) — element-count fallback for 4D-to-4D when permute condition didn't match. Same risk as case 5.
5. **General fallback** (line 579-583) — any element-count match. Dangerous and silent. Currently unreachable for well-formed LycorisAdapter outputs but no defence in depth.

No checks for:
- DType (the cast at line 87-93 happens AFTER alignment — fine, ordering is OK).
- Device match.
- Bias detection (a 1D adapter delta would fall through the cases — and no case handles 1D base).
- Whether the dtype is even compatible with `add` (BF16 + FP8 etc.).

**Recommendation**: replace case 5 with an explicit error including suggested permutations. The safe set is exactly 1-3.

## Unverified / couldn't check

- Whether `LoCon Tucker mid` permute `[2,3,1,0]` matches Kohya's actual on-disk convention. The Python upstream `weight_gen` saves `[rank, rank, KH, KW]` but `modules/locon.py` `lora_mid = self.module(lora_dim, lora_dim, k_size, ...)` produces `[lora_dim, lora_dim, kh, kw]` — i.e. `[R_out, R_in, KH, KW]`? The "out, in" order for nn.Conv2d is the convention. Permute `[2,3,1,0]` → `[KH, KW, R_in, R_out]`. The downstream LoCon Tucker math expects `mid_hw [R_in, R_out]` for `down [IC, R_in] @ mid_hw @ up [R_out, OC]`. **Looks consistent**, but unverified by execution.
- `tensor.reshape` for a non-contiguous tensor — does flame-core auto-contig or error? Several reshapes in lokr.rs follow permutes without explicit contiguous(). If flame-core requires contiguous, those would silently produce wrong data on a permuted tensor. I did not verify flame-core's reshape contract.
- `Tensor::stack` consistency between calls — I checked the signature (`unsqueeze + cat`, `flame-core/src/tensor.rs:3078-3101`). It adds a new axis at `axis`. Tucker reshape in `ops/tucker.rs:172,177` stacks with axis=0 twice → produces `[KH, KW, IC, OC]`. ✓ consistent.
- The exact `make_kron` behaviour of `torch.kron` for 4D × 4D inputs (with leading 1×1 on the `w1`-padded side). PyTorch docs say it computes `kron(A, B)[i_1*p_1+k_1, ..., i_n*p_n+k_n] = A[i_1,..,i_n] * B[k_1,..,k_n]`. The Rust `make_kronecker_conv_kernel` produces a permuted-then-reshaped result that I traced symbolically matches; not runtime-verified.
- LoKr `w2` 4D linear case (loader sees w2.shape=[OK,IN,1,1]): downstream `is_conv` says `false` due to the 1×1 special case (`lokr.rs:488`); then linear `get_diff_weight` reshapes `[OK,IN,1,1] → [OK,IN]` (`lokr.rs:159-161`). ✓ But the reshape is correct only because the data is already `[OK,IN]`-row-major. If a tensor came in genuinely shaped `[OK,IN,1,1]` from disk, the row-major flatten is OK. ✓
- Whether real Flux/SDXL LoRAs that include text-encoder `.alpha` keys will hit any unexpected classification (e.g. `.alpha` matches the suffix list, but with no `.lora_up.weight` in the same group → `Unknown`, skipped silently). The loader prints a one-line warning. Could miss prefixes that only have `.alpha` (e.g. corrupted LoRA).

---

## TL;DR — what to fix first

1. **Wire LoCon Tucker to `ops::tucker::rebuild_conv_tucker`** — 5 lines, eliminates P0-1.
2. **Implement Tucker LoHa via stack-based contraction** (mirror what tucker.rs does) — eliminates P0-2.
3. **Fix Tucker LoHa loader** to detect via presence of `.hada_t1`/`.hada_t2`, not w1a rank — eliminates P0-3.
4. **Fix LoKr full-W1+full-W2 scale** to default to 1.0 when both are full and rank inference returned 0 — eliminates P0-4.
5. **Detect DoRA and refuse to merge** (or implement `apply_dora_scale`) — at minimum, log a loud warning. Eliminates P0-6 silent corruption.
6. **Apply `.diff_b`** in `apply_collection` to the corresponding bias key — eliminates P0-7.
7. **Add a runtime numerical regression** that compares `delta_weight()` against a reference Python computation on a few real (small) LoRAs.
