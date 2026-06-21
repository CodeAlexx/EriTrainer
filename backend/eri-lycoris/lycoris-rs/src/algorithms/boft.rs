//! BOFT (Butterfly Orthogonal Fine-Tuning) — `m`-stage butterfly-factorized
//! rotation that extends Diag-OFT.
//!
//! Mirrors lycoris-upstream's `ButterflyOFTModule` (lycoris/modules/boft.py)
//! and the forward helper in lycoris/functional/boft.py. Built as an
//! input-rotation adapter in the OneTrainer style, sharing the wrapper plumbing
//! with [`crate::algorithms::oft::OFTModule`].
//!
//! # Algorithm
//!
//! Diag-OFT applies *one* block-diagonal rotation to the input feature axis.
//! BOFT instead applies `m` consecutive rotations, where between stages the
//! feature axis is permuted so the `block_num` blocks of stage `i` interleave
//! the blocks of stage `i+1`. This forms a butterfly network whose effective
//! group structure is exponentially richer than a single block-diagonal map,
//! at the cost of `m`× the parameter count.
//!
//! Per upstream (`functional/boft.py:55-66`), each stage `i` runs:
//!
//! ```text
//! g = 2,  k = 2^i * (b/2)
//! inp = inp.unflatten(-1, (-1, g, k))      # (..., c, g, k)
//!          .transpose(-2, -1)              # (..., c, k, g)
//!          .flatten(-3)                    # (..., c·k·g) == (..., in)
//!          .unflatten(-1, (-1, b))         # (..., num_blocks, b)
//! inp = einsum("n i j, ... n j -> ... n i", R[i], inp)   # rotate per-block
//! inp = inp.flatten(-2)                    # (..., in)
//!          .unflatten(-1, (-1, k, g))      # (..., c, k, g)
//!          .transpose(-2, -1)              # (..., c, g, k)
//!          .flatten(-3)                    # (..., in)
//! ```
//!
//! After all `m` stages the result is the rotated input. The pre/post
//! permutations are involutions (running stage `i` forward and again backward
//! yields identity) which is what gives the butterfly its group structure.
//!
//! # Storage layout
//!
//! `oft_blocks` shape `[boft_m, block_num, block_size, block_size]`, F32.
//! Initialized to zeros so each stage's `R[i] = I` and the whole network is
//! identity at step 0.
//!
//! `boft_m = sum(bits(block_num-1)) + 1` matches the Python constructor and
//! the (m, n) factor produced by `power2factorization`.
//!
//! # Factorization choice (input-side)
//!
//! Like our [`OFTModule`], BOFT factorizes the **input** feature dim (matching
//! OneTrainer's input-rotation semantics). Upstream lycoris factorizes the
//! output dim — files saved by SimpleTuner are NOT byte-compatible with this
//! crate without a transpose pass. Tracked as the same Tier-2 follow-up as
//! OFT.
//!
//! # Cast-then-record (autograd safety)
//!
//! `oft_blocks` lives in F32; the input is typically BF16. We compute
//! `R` in F32 (Cayley-Neumann), then `to_dtype(x.dtype())` in record-mode so
//! backward routes Cast → F32 leaf. This mirrors `LoConModule::forward` and
//! is mandatory after the 2026-05-09 chroma misaligned-address fix
//! (`ac5350d`): doing the cast inside `bmm`'s auto-cast goes through
//! `to_dtype_no_grad` and corrupts FlashAttention backward several layers up.
//!
//! # Public surface
//!
//! - [`BOFTModule::apply_to_input`] — forward rotation entrypoint, mirrors
//!   `OFTModule::apply_to_input`. Routed through `LycorisLinear::apply_input`
//!   when `is_input_rotation()` returns `true`.
//! - [`LycorisModule::forward`] — alias for `apply_to_input` so the trait is
//!   satisfied; callers should prefer the explicit method.
//! - `get_diff_weight` and `merge_to` error out with the same rationale as
//!   OFT (multiplicative on the base weight, needs caller-side base-weight
//!   threading).
//!
//! # Stubbed / deferred
//!
//! - **COFT constraint clamping** (`||Q||_F` cap per block): same status as
//!   OFT — wired through but `project_blocks_inplace` returns an error.
//!   flame-core lacks a per-batch Frobenius norm primitive.
//! - **Conv path**: upstream BOFT supports conv1d/2d/3d. We only ship the
//!   linear path, matching our OFTModule (conv layers in the EDv2 trainer
//!   architecture flow through Linear-equivalent reshapes already).
//! - **Rescale parameter**: upstream's optional `rescale` (per-output-feature
//!   gain) is not implemented. Trivial to add when needed; defaults to off.
//! - **Save-format transpose**: SimpleTuner BOFT checkpoints assume
//!   output-side factorization; loading them requires a `[m, n, b, b] →
//!   [m, n, b, b]^T` pass, deferred to a follow-up loader change.

use crate::tensor_utils::StorageDtype;
use crate::{Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

/// BOFT module. Butterfly-factorized orthogonal rotation with `m` stages.
pub struct BOFTModule {
    /// Trainable rotation parameters. Shape `[boft_m, num_blocks, b, b]`,
    /// F32 storage. Initialized to zeros so `R[i] = I` at every stage and
    /// the forward is identity at step 0.
    pub blocks: Parameter,

    /// Per-block edge length `b`. Power of two by construction.
    pub block_size: usize,

    /// Number of block-diagonal blocks per stage (`in_features / block_size`).
    /// Power of two by construction.
    pub num_blocks: usize,

    /// Number of butterfly stages: `sum(bits(num_blocks - 1)) + 1`. Set to
    /// `floor(log2(num_blocks)) + 1` whenever `num_blocks` is itself a power
    /// of two (which it is, by construction).
    pub boft_m: usize,

    /// Linear layer's input feature count. Drives the factorization
    /// (`num_blocks * block_size == in_features`). Like OFT, the rotation
    /// acts on the input axis.
    pub in_features: usize,

    /// Linear layer's output feature count. Recorded for shape checks /
    /// save metadata only.
    pub out_features: usize,

    /// BOFT scaling multiplier. Typically `1.0`.
    pub alpha: f32,

    /// Optional COFT clamp (`||Q||_F` cap per block). Not auto-applied.
    pub constraint: Option<f32>,

    /// Stability epsilon for COFT division. 1e-6.
    pub epsilon: f32,

    /// Number of Neumann series terms (default 5).
    pub neumann_terms: usize,

    /// Compute-path dtype. `blocks` always F32; `R` is cast at the boundary.
    pub storage_dtype: StorageDtype,

    /// Device handle (kept for future allocations / merge ops).
    pub device: Arc<CudaDevice>,
}

#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

/// Compute `(block_size, block_num)` for BOFT given target `dim` and a max
/// block-size hint `factor` (set to `lora_dim` upstream, or `-1`/`dim` for "no
/// cap"). Mirrors `lycoris/functional/general.py:power2factorization`.
///
/// Returns `Err` when `dim` cannot be expressed as `m * 2^p` under
/// `m <= factor`. For our trainer this typically only fires on tiny or
/// non-power-of-two layers; the caller should fall back to plain OFT.
fn power2factorization(dim: usize, factor: usize) -> Result<(usize, usize)> {
    if dim == 0 {
        return Err(Error::InvalidOperation("BOFT dim must be > 0".into()));
    }
    let factor = if factor == 0 || factor == usize::MAX {
        dim
    } else {
        factor
    };

    let mut m: usize = 0;
    let mut n: usize = 0;
    // Mirror upstream's loop structure exactly.
    loop {
        m += 2;
        // Walk to next m that divides dim, capped at factor.
        while dim % m != 0 && m < dim {
            m += 2;
        }
        if m > factor {
            break;
        }
        let q = dim / m;
        // q must be a power of two for BOFT (single bit set).
        if q.is_power_of_two() {
            n = q;
        }
        if m >= factor {
            break;
        }
    }

    if n == 0 {
        return Err(Error::InvalidOperation(format!(
            "BOFT cannot factor dim={} with factor<={} (need m*2^p form)",
            dim, factor
        )));
    }
    // (block_size, block_num) — block_size = dim / n, block_num = n.
    Ok((dim / n, n))
}

/// Build a `[boft_m, num_blocks, b, b]` identity tensor in F32. Used to
/// kick off the Cayley-Neumann series.
fn batched_identity_2stage(
    boft_m: usize,
    num_blocks: usize,
    block_size: usize,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let total = boft_m * num_blocks * block_size * block_size;
    let mut data = vec![0.0f32; total];
    for s in 0..boft_m {
        for r in 0..num_blocks {
            let base = ((s * num_blocks) + r) * block_size * block_size;
            for i in 0..block_size {
                data[base + i * block_size + i] = 1.0;
            }
        }
    }
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[boft_m, num_blocks, block_size, block_size]),
        device,
        DType::F32,
    )
    .map_err(Error::Flame)
}

impl BOFTModule {
    /// Construct a BOFT module for a Linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features`   — input dim of the underlying Linear; must factor as
    ///                     `m * 2^p` for some `m ≤ max_block_size`.
    /// * `out_features`  — output dim (recorded only).
    /// * `max_block_size`— hint cap on `b` (the upstream `lora_dim`).
    ///                     Common: 4 or 8. Use `0` for "no cap" (= `dim`).
    /// * `alpha`         — typically `1.0`.
    /// * `constraint`    — optional COFT clamp.
    /// * `storage_dtype` — compute-path dtype for `R` application.
    /// * `device`        — CUDA device.
    ///
    /// # Init behavior
    ///
    /// `blocks` starts as zeros → every stage's `R = I` → forward is
    /// identity. Verify with `is_identity_init`.
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        max_block_size: usize,
        alpha: f32,
        constraint: Option<f32>,
        storage_dtype: StorageDtype,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Like OFT, we factorize the input dim. Upstream factorizes output;
        // the file-format compatibility caveat is documented in the module
        // docstring.
        let (block_size, num_blocks) = power2factorization(in_features, max_block_size)?;

        // Upstream: boft_m = sum(bits(block_num - 1)) + 1.  For power-of-two
        // block_num this equals log2(block_num) + 1 (block_num-1 is all-ones
        // up to that bit width).  We compute the literal popcount form to
        // stay bit-faithful to the Python.
        let boft_m = if num_blocks == 0 {
            1
        } else {
            ((num_blocks - 1) as u32).count_ones() as usize + 1
        };

        // Sanity: BOFT requires block_size to be even (the per-stage permute
        // splits the feature axis into groups of 2).  power2factorization's
        // `m` is incremented by 2 so this is guaranteed, but assert anyway.
        if block_size % 2 != 0 {
            return Err(Error::InvalidOperation(format!(
                "BOFT block_size ({}) must be even",
                block_size
            )));
        }

        let blocks = Tensor::zeros_dtype(
            Shape::from_dims(&[boft_m, num_blocks, block_size, block_size]),
            DType::F32,
            device.clone(),
        )
        .map_err(Error::Flame)?
        .requires_grad_(true);

        Ok(Self {
            blocks: Parameter::new(blocks),
            block_size,
            num_blocks,
            boft_m,
            in_features,
            out_features,
            alpha,
            constraint,
            epsilon: 1e-6,
            neumann_terms: 5,
            storage_dtype,
            device,
        })
    }

    /// Override the default Neumann term count (default 5). Must be ≥ 1.
    pub fn with_neumann_terms(mut self, n: usize) -> Self {
        self.neumann_terms = n.max(1);
        self
    }

    /// Compute the per-stage skew-symmetric tensor
    /// `Q_skew = blocks − blocks^T` (transpose over the inner [b, b] of each
    /// `[boft_m, num_blocks, b, b]` slice). F32, shape unchanged.
    fn skew(&self) -> Result<Tensor> {
        let blocks = param_tensor(&self.blocks)?;
        // `transpose_batch` swaps the trailing two dims regardless of the
        // batch-dim count, so [m, n, b, b] → [m, n, b, b] with inner
        // transpose applied per (m, n) slice.
        let bt = blocks.transpose_batch().map_err(Error::Flame)?;
        blocks.sub(&bt).map_err(Error::Flame)
    }

    /// Build the full per-stage rotation tensor `R` (F32) via Cayley-Neumann.
    /// Shape `[boft_m, num_blocks, b, b]`. Each `R[i]` is `num_blocks`
    /// independent rotations applied in stage `i`. Mirrors
    /// `oft_utils.py:_cayley_batch` lifted to a 4D layout.
    fn cayley_neumann(&self) -> Result<Tensor> {
        let q = self.skew()?;
        let n = self.neumann_terms.max(1);

        let id = batched_identity_2stage(
            self.boft_m,
            self.num_blocks,
            self.block_size,
            self.device.clone(),
        )?;
        if n <= 1 {
            return Ok(id);
        }

        // R = I + 2·Q
        let two_q = q.mul_scalar(2.0).map_err(Error::Flame)?;
        let mut r = id.add(&two_q).map_err(Error::Flame)?;
        if n <= 2 {
            return Ok(r);
        }

        // R += 2·Q²  (bmm operates on trailing two dims, so [m,n,b,b] @
        // [m,n,b,b] works as a fully-batched per-stage per-block matmul.)
        let q_squared = q.bmm(&q).map_err(Error::Flame)?;
        let two_q_sq = q_squared.mul_scalar(2.0).map_err(Error::Flame)?;
        r = r.add(&two_q_sq).map_err(Error::Flame)?;

        // Loop body identical to OFT (same Neumann coefficient table):
        //   for _ in range(3, n - 1):
        //       Q_power = Q_power @ Q
        //       R += 2·Q_power
        //   Q_power = Q_power @ Q
        //   R += Q_power           # final term coefficient 1
        let mut q_power = q_squared;
        let loop_end = if n >= 4 { n - 1 } else { 3 };
        for _ in 3..loop_end {
            q_power = q_power.bmm(&q).map_err(Error::Flame)?;
            let scaled = q_power.mul_scalar(2.0).map_err(Error::Flame)?;
            r = r.add(&scaled).map_err(Error::Flame)?;
        }
        q_power = q_power.bmm(&q).map_err(Error::Flame)?;
        r = r.add(&q_power).map_err(Error::Flame)?;

        Ok(r)
    }

    /// Public accessor: current per-stage rotation tensor (F32).
    pub fn rotation(&self) -> Result<Tensor> {
        self.cayley_neumann()
    }

    /// Apply the BOFT butterfly rotation to an input tensor.
    ///
    /// Input shape: `[*..., in_features]`. The leading dims are flattened
    /// into a single batch axis so the per-stage permutes act on `[B, in]`
    /// uniformly. After all `m` stages the tensor is reshaped back to the
    /// caller's input shape.
    ///
    /// Per-stage math (for stage `i`):
    /// 1. `g = 2`, `k = 2^i * (b / 2)`
    /// 2. Reshape feature axis `[B, in]` → `[B, c, g, k]` where
    ///    `c = in / (g·k)`; transpose last two → `[B, c, k, g]`; flatten →
    ///    `[B, in]`; unflatten last → `[B, num_blocks, b]`.
    /// 3. Per-block bmm with `R[i, :, :, :]` of shape `[num_blocks, b, b]`:
    ///    permute `[B, num_blocks, b]` → `[num_blocks, B, b]`, bmm with R[i],
    ///    permute back to `[B, num_blocks, b]`. (Same einsum trick as OFT.)
    /// 4. Inverse of step 2: flatten `[B, num_blocks, b]` → `[B, in]`,
    ///    unflatten `[B, c, k, g]`, transpose → `[B, c, g, k]`, flatten →
    ///    `[B, in]`.
    ///
    /// At init (`blocks = 0`), every `R[i] = I`, so each stage's bmm is the
    /// identity and the permute pairs cancel — output bit-equal to input.
    pub fn apply_to_input(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims().to_vec();
        let last = *in_dims.last().ok_or_else(|| {
            Error::InvalidOperation("BOFT input must have at least 1 dim".into())
        })?;
        if last != self.in_features {
            return Err(Error::InvalidOperation(format!(
                "BOFT input last dim {} != in_features {}",
                last, self.in_features
            )));
        }

        // Cast R to x.dtype() in record-mode so backward routes Cast → F32
        // leaf (see module docstring; mandatory after 2026-05-09 fix).
        let r_f32 = self.cayley_neumann()?;
        let x_dtype = x.dtype();
        let r = if x_dtype != DType::F32 {
            r_f32.to_dtype(x_dtype).map_err(Error::Flame)?
        } else {
            r_f32
        };

        // Flatten leading dims to a single batch.
        let numel: usize = in_dims.iter().product();
        let batch = numel / self.in_features;
        let mut h = x
            .reshape(&[batch, self.in_features])
            .map_err(Error::Flame)?;

        let b = self.block_size;
        let r_b = b / 2;
        let in_f = self.in_features;
        let num_blocks = self.num_blocks;

        for i in 0..self.boft_m {
            let g = 2usize;
            // k = 2^i * (b/2).  By construction k * g = 2^(i+1) * (b/2),
            // and `c = in / (g * k)` is a positive integer (in is a
            // multiple of `b`, and `b = g * (b/2)` so `c` divides
            // `num_blocks / 2^i`).
            let k = (1usize << i) * r_b;
            let gk = g * k;
            if gk == 0 || in_f % gk != 0 {
                return Err(Error::InvalidOperation(format!(
                    "BOFT stage {} cannot reshape: in_features {} not divisible by g*k = {}",
                    i, in_f, gk
                )));
            }
            let c = in_f / gk;

            // Step 1 (forward permute): [B, in] → [B, c, g, k] → [B, c, k, g]
            //                                   → [B, in]      → [B, n, b].
            let h_cgk = h.reshape(&[batch, c, g, k]).map_err(Error::Flame)?;
            let h_ckg = h_cgk.permute(&[0, 1, 3, 2]).map_err(Error::Flame)?;
            let h_flat = h_ckg.reshape(&[batch, in_f]).map_err(Error::Flame)?;
            let h_nb = h_flat
                .reshape(&[batch, num_blocks, b])
                .map_err(Error::Flame)?;

            // Step 2 (per-block bmm with R[i]): slice the i-th stage of R.
            //   R is [boft_m, num_blocks, b, b].  Use `narrow` over dim 0
            //   to grab `[1, num_blocks, b, b]`, then squeeze stage axis.
            //   Falls back to a reshape after narrow if there's no squeeze.
            let r_stage = r
                .narrow(0, i, 1)
                .map_err(Error::Flame)?
                .reshape(&[num_blocks, b, b])
                .map_err(Error::Flame)?;
            // Permute [B, n, b] → [n, B, b] so bmm batch axis = num_blocks.
            let h_nb_b = h_nb.permute(&[1, 0, 2]).map_err(Error::Flame)?;
            // bmm( [n, B, b], [n, b, b] ) → [n, B, b].
            let rotated = h_nb_b.bmm(&r_stage).map_err(Error::Flame)?;
            // Permute back [n, B, b] → [B, n, b].
            let rotated_bn = rotated.permute(&[1, 0, 2]).map_err(Error::Flame)?;

            // Step 3 (inverse permute): [B, n, b] → [B, in] → [B, c, k, g]
            //                              → [B, c, g, k]  → [B, in].
            let r_flat = rotated_bn.reshape(&[batch, in_f]).map_err(Error::Flame)?;
            let r_ckg = r_flat.reshape(&[batch, c, k, g]).map_err(Error::Flame)?;
            let r_cgk = r_ckg.permute(&[0, 1, 3, 2]).map_err(Error::Flame)?;
            h = r_cgk.reshape(&[batch, in_f]).map_err(Error::Flame)?;
        }

        // Reshape back to caller's leading dims.
        h.reshape(&in_dims).map_err(Error::Flame)
    }

    /// COFT projection. Caller-driven, currently stubbed.
    pub fn project_blocks_inplace(&mut self) -> Result<()> {
        if self.constraint.is_none() {
            return Ok(());
        }
        Err(Error::InvalidOperation(
            "BOFT COFT projection not yet implemented (needs batched Frobenius \
             norm primitive in flame-core); ship constraint=None for now"
                .into(),
        ))
    }

    /// True if `blocks` is exactly zero (forward is identity).
    pub fn is_identity_init(&self) -> Result<bool> {
        let blocks = param_tensor(&self.blocks)?;
        let v = blocks.to_vec().map_err(Error::Flame)?;
        Ok(v.iter().all(|x| *x == 0.0))
    }
}

impl LycorisModule for BOFTModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mirrors OFTModule: "forward" on a rotation adapter means "rotate
        // the input"; the base linear runs separately on the rotated value.
        self.apply_to_input(x)
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        // Same rationale as OFT: BOFT is multiplicative
        // (`W' = ... R[m-1] · ... · R[0] · W` after the corresponding axis
        // permutes), so an additive ΔW would need access to the base weight.
        // Caller must use `apply_to_input` for forward-time application or
        // synthesize the merged weight externally.
        Err(Error::InvalidOperation(
            "BOFT delta is multiplicative (butterfly rotation chain on input); \
             cannot synthesize an additive ΔW without access to the base \
             weight. Use BOFTModule::apply_to_input."
                .into(),
        ))
    }

    fn merge_to(&mut self, _multiplier: f32) -> Result<()> {
        Err(Error::InvalidOperation(
            "BOFT merge requires base weight; perform the butterfly rotation \
             chain on W externally."
                .into(),
        ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![param_tensor(&self.blocks).expect("BOFT.blocks mutex poisoned")]
    }

    fn parameters_handles(&self) -> Vec<Parameter> {
        vec![self.blocks.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power2factorization_basic() {
        // dim=64, factor=4: m must be ≤ 4 and even. m=4 works iff 64/4=16 is
        // a power of two. → (block_size=4, block_num=16).
        let (b, n) = power2factorization(64, 4).unwrap();
        assert_eq!(b, 4);
        assert_eq!(n, 16);
    }

    #[test]
    fn test_power2factorization_no_cap() {
        // factor=0 means "no cap", upstream uses `factor=-1` → same effect.
        // For dim=32 (= 2·16 = 4·8 = 8·4 = 16·2), the loop walks m up to
        // dim and the last valid m wins.
        let (b, n) = power2factorization(32, 0).unwrap();
        assert!(b * n == 32, "block_size * block_num must equal dim");
        assert!(n.is_power_of_two(), "block_num must be a power of two");
        assert!(b % 2 == 0, "block_size must be even");
    }

    #[test]
    fn test_power2factorization_unfactorable() {
        // 50 = 2·25; 25 isn't a power of two, no other even m divides 50
        // with a power-of-two cofactor → must error.
        assert!(power2factorization(50, 4).is_err());
    }

    #[test]
    fn test_boft_m_popcount() {
        // num_blocks=16 → 16-1=15=0b1111 → popcount=4 → boft_m=5.
        let n = 16usize;
        let m = ((n - 1) as u32).count_ones() as usize + 1;
        assert_eq!(m, 5);

        // num_blocks=8 → 8-1=7=0b111 → popcount=3 → boft_m=4.
        let n = 8usize;
        let m = ((n - 1) as u32).count_ones() as usize + 1;
        assert_eq!(m, 4);
    }

    #[test]
    fn test_neumann_loop_bounds_n5() {
        // Same coefficient table as OFT: n=5 → loop runs once (Q³ added with
        // coeff 2), final tail Q⁴ with coeff 1.
        let n = 5usize;
        let loop_iters = if n >= 4 { (n - 1) - 3 } else { 0 };
        assert_eq!(loop_iters, 1);
    }
}
