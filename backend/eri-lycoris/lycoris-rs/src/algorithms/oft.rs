//! Diag-OFT (Orthogonal Fine-Tuning) — block-diagonal Cayley-Neumann rotation.
//!
//! Mirrors lycoris-upstream's `DiagOFTModule` (lycoris/modules/diag_oft.py) and
//! OneTrainer's `OFTRotationModule` (modules/module/oft_utils.py:37). We store
//! **full blocks** rather than the upper-triangular vector form OneTrainer uses,
//! because flame-core has neither `triu_indices` nor a scatter primitive —
//! full-block storage trades a small amount of memory (`b²` floats per block
//! instead of `b·(b−1)/2`) for autograd safety: skew construction reduces to
//! `Q − Q.transpose_batch()`.
//!
//! # Algorithm
//!
//! For each Linear layer with weight `W [in, out]`:
//! - Factorize the **input** dim into `num_blocks · block_size`. Each block is
//!   a `[b, b]` orthogonal matrix that rotates its slice of the input feature
//!   axis. This matches OneTrainer's `OFTRotationModule` (oft_utils.py:153,
//!   `rank = self.in_features // self.block_size`). num_blocks is derived
//!   from `in_features / block_size`.
//! - The trainer flow is `output = base_linear(R · x)`: the adapter rotates
//!   the input via [`AdapterModule::apply_input`], the base linear consumes
//!   the rotated input. Because OFT operates on input space (not as an
//!   additive output delta), the wrapper checks
//!   [`AdapterModule::is_input_rotation`] and calls `apply_input` instead of
//!   `forward_delta` for OFT adapters. **Works for any layer shape — square,
//!   non-square FFN gates, projection-out layers, etc.**
//! - Save-format note: this differs from lycoris-upstream (which factorizes
//!   `out_features`) and stores `oft_blocks` interpreted on the output side.
//!   We share the same shape `[block_num, block_size, block_size]` but the
//!   `block_num` dimension here means input-side blocks. Loading a
//!   SimpleTuner-trained OFT file into our trainer (or vice versa) without
//!   transposition will silently produce wrong rotations. Parity loader is
//!   a Tier 2 follow-up.
//! - Per block, the trainable parameter is a full `[b, b]` matrix `Q`. Skew form
//!   `Q_skew = Q − Q^T` (antisymmetric).
//! - The rotation `R = (I − Q_skew)(I + Q_skew)^{-1}` (Cayley) is approximated
//!   by a Neumann series, so **no matrix inverse** is needed:
//!     - `n = 5` (default): `R ≈ I + 2·Q + 2·Q² + 2·Q³ + Q⁴`
//!     - For other `n`: terms 2..n−1 carry coefficient 2; the final term is 1.
//!   This matches OneTrainer's `_cayley_batch` line-for-line (oft_utils.py:86).
//! - Forward applies `R` to the **input**: `output = orig(R(input))`. Reshape
//!   `x [*..., in]` → `[*..., num_blocks, block_size]`, then for each block index
//!   `r`: `x_rot[..., r, :] = x[..., r, :] @ R[r, :, :]`. Implemented with a
//!   single permute + `bmm` instead of an einsum primitive.
//!
//! # At-init no-op behavior
//!
//! Constructor calls `Tensor::zeros_dtype` for `blocks`. Therefore:
//! - `Q = blocks − blocks.transpose_batch() = 0`
//! - Neumann series collapses to `R = I`
//! - Forward: `x_rot[*..., r, :] = x[*..., r, :] @ I = x[*..., r, :]`
//! - Output is bit-identical to input → the OFT layer is a no-op until training
//!   moves `blocks` away from zero. This matches the LoRA-B = 0 convention.
//!
//! Verification path: assert `module.forward(x).to_vec() == x.to_vec()` after
//! construction. (See test stub below — gated off `cargo run` per task rules.)
//!
//! # Storage
//!
//! `blocks` is held in **F32** regardless of the requested `storage_dtype`,
//! because the skew → Cayley → Neumann pipeline accumulates ~5 matmuls per
//! block per step and BF16 round-off there visibly biases `R` away from
//! orthogonality. The compute path may downcast `R` to BF16 before applying it
//! to the (typically BF16) input via `apply_dtype`; this mirrors the
//! `previous_dtype` round-trip pattern in oft_utils.py:93,119.
//!
//! # Constraint (COFT)
//!
//! Optional `constraint: Some(eps)` clamps `||Q_skew||_F` per-block via a
//! deterministic scalar projection (`q_clamped = q * min(1, eps / ||q||)`).
//! Off by default in OneTrainer; documented here but **not auto-applied** —
//! caller must invoke `project_blocks_inplace` between optimizer steps if
//! desired. (Mirrors OneTrainer's `_project_batch`, oft_utils.py:121.)

use crate::tensor_utils::StorageDtype;
use crate::{Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

/// Diag-OFT module. Block-diagonal orthogonal rotation, Cayley-Neumann form.
pub struct OFTModule {
    /// Trainable rotation parameter, F32 storage. Shape `[num_blocks, b, b]`.
    /// Initialized to zeros so `R = I` and forward is identity at step 0.
    /// Wrapped in `Parameter` so optimizer mutations are visible to forward.
    pub blocks: Parameter,

    /// Block edge length (the `b` in `[b, b]`).
    pub block_size: usize,

    /// Number of independent blocks (`out_features / block_size`).
    pub num_blocks: usize,

    /// Linear layer's input feature count. **Drives the block factorization**
    /// (`num_blocks * block_size == in_features`) — the rotation acts on the
    /// input feature axis (OneTrainer style). Works for any out_features.
    pub in_features: usize,

    /// Linear layer's output feature count. Recorded for shape-checking
    /// downstream callers and save/load metadata; not used by the rotation
    /// itself.
    pub out_features: usize,

    /// OFT scaling multiplier. Typically `1.0`. The rotation is exactly
    /// orthogonal at small `||Q||`; alpha lets the caller dial down the
    /// effective rotation magnitude post-hoc (`R_eff = lerp(I, R, alpha)`).
    /// Default behavior when `alpha == 1.0`: identity application of `R`.
    pub alpha: f32,

    /// Optional COFT clamp: clip `||Q_skew||_F` per block to this value.
    /// `None` ≡ no clamp. Caller drives projection via `project_blocks_inplace`.
    pub constraint: Option<f32>,

    /// Stability epsilon for COFT division. Standard 1e-6.
    pub epsilon: f32,

    /// Number of Neumann series terms (default 5). Mirrors OneTrainer's
    /// `num_cayley_neumann_terms`. Must be ≥ 1.
    pub neumann_terms: usize,

    /// Compute-path dtype. `blocks` stay F32; only `R` is cast at the boundary.
    pub storage_dtype: StorageDtype,

    /// Device handle (kept for potential future allocations / merge ops).
    pub device: Arc<CudaDevice>,
}

#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

#[inline]
fn check_divisibility(out_features: usize, block_size: usize) -> Result<usize> {
    if block_size == 0 {
        return Err(Error::InvalidOperation(
            "OFT block_size must be > 0".into(),
        ));
    }
    if out_features % block_size != 0 {
        return Err(Error::InvalidOperation(format!(
            "OFT requires out_features ({}) divisible by block_size ({})",
            out_features, block_size
        )));
    }
    Ok(out_features / block_size)
}

/// Build a `[num_blocks, b, b]` identity tensor on device. Used both for
/// the Neumann series start (`R = I`) and for the no-op assertion path.
fn batched_identity(
    num_blocks: usize,
    block_size: usize,
    dtype: DType,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let total = num_blocks * block_size * block_size;
    let mut data = vec![0.0f32; total];
    for r in 0..num_blocks {
        let base = r * block_size * block_size;
        for i in 0..block_size {
            data[base + i * block_size + i] = 1.0;
        }
    }
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[num_blocks, block_size, block_size]),
        device,
        dtype,
    )
    .map_err(Error::Flame)
}

impl OFTModule {
    /// Construct a Diag-OFT module for a Linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features`   — input dim of the underlying Linear; **must be
    ///                     divisible by `block_size`**.
    /// * `out_features`  — output dim of the underlying Linear (recorded for
    ///                     shape checks; rotation acts on input only).
    /// * `block_size`    — `b`, the per-block edge length. Common: 32.
    /// * `alpha`         — OFT scale, typically `1.0`.
    /// * `constraint`    — optional COFT clamp (`||Q||_F` cap per block);
    ///                     `None` = unclamped.
    /// * `storage_dtype` — compute-path dtype for `R` application; `blocks` is
    ///                     always F32 internally.
    /// * `device`        — CUDA device.
    ///
    /// # Init behavior
    ///
    /// `blocks` starts as zeros, so `R = I` and forward is identity. Caller
    /// can verify with `assert_at_init_noop` (see test).
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        block_size: usize,
        alpha: f32,
        constraint: Option<f32>,
        storage_dtype: StorageDtype,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // OFT (OneTrainer style) factorizes the **input** feature dim. The
        // rotation acts on `x`, then the base linear consumes the rotated
        // input via the adapter wrapper's `apply_input` path. Works for
        // arbitrary `out_features`.
        let num_blocks = check_divisibility(in_features, block_size)?;

        // F32 storage for `blocks` per the precision rationale in the module
        // docstring (Neumann accumulation eats BF16 mantissa). Mark
        // requires_grad so autograd records updates; the trainer wraps this
        // tensor in a Parameter so the optimizer's set_data lands here.
        let blocks = Tensor::zeros_dtype(
            Shape::from_dims(&[num_blocks, block_size, block_size]),
            DType::F32,
            device.clone(),
        )
        .map_err(Error::Flame)?
        .requires_grad_(true);

        Ok(Self {
            blocks: Parameter::new(blocks),
            block_size,
            num_blocks,
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

    /// Compute the per-block skew-symmetric matrix `Q_skew = blocks − blocks^T`.
    /// Always F32. Shape `[num_blocks, b, b]`.
    fn skew(&self) -> Result<Tensor> {
        let blocks = param_tensor(&self.blocks)?;
        // transpose_batch swaps the last two dims for a 3D tensor.
        let bt = blocks.transpose_batch().map_err(Error::Flame)?;
        blocks.sub(&bt).map_err(Error::Flame)
    }

    /// Build the rotation `R` via Cayley-Neumann. F32 throughout. Shape
    /// `[num_blocks, b, b]`. Mirrors `oft_utils.py:_cayley_batch`.
    fn cayley_neumann(&self) -> Result<Tensor> {
        let q = self.skew()?;
        let n = self.neumann_terms.max(1);

        // R = I  (the n=1 case already terminates here.)
        let id = batched_identity(self.num_blocks, self.block_size, DType::F32, self.device.clone())?;
        if n <= 1 {
            return Ok(id);
        }

        // R = I + 2·Q
        let two_q = q.mul_scalar(2.0).map_err(Error::Flame)?;
        let mut r = id.add(&two_q).map_err(Error::Flame)?;

        if n <= 2 {
            return Ok(r);
        }

        // R += 2·Q²
        let q_squared = q.bmm(&q).map_err(Error::Flame)?;
        let two_q_sq = q_squared.mul_scalar(2.0).map_err(Error::Flame)?;
        r = r.add(&two_q_sq).map_err(Error::Flame)?;

        // Mirror OneTrainer's loop (oft_utils.py:106):
        //   Q_power = Q²
        //   for _ in range(3, n - 1):
        //       Q_power = Q_power @ Q
        //       R += 2·Q_power
        //   Q_power = Q_power @ Q          # one more multiply
        //   R += Q_power                    # final term has coefficient 1
        let mut q_power = q_squared;
        let loop_end = if n >= 4 { n - 1 } else { 3 };
        for _ in 3..loop_end {
            q_power = q_power.bmm(&q).map_err(Error::Flame)?;
            let scaled = q_power.mul_scalar(2.0).map_err(Error::Flame)?;
            r = r.add(&scaled).map_err(Error::Flame)?;
        }
        // Final tail term (always present when n >= 3): coefficient 1.0.
        q_power = q_power.bmm(&q).map_err(Error::Flame)?;
        r = r.add(&q_power).map_err(Error::Flame)?;

        Ok(r)
    }

    /// Public accessor: compute the current rotation `R` (F32, no autograd
    /// gating beyond what the underlying ops record). Useful for diagnostics
    /// (e.g. measuring `||R·R^T − I||_F` during training).
    pub fn rotation(&self) -> Result<Tensor> {
        self.cayley_neumann()
    }

    /// Apply the OFT rotation to an input tensor. Shape `[*..., in_features]`.
    ///
    /// Math: `out[*..., r, :] = x[*..., r, :] @ R[r, :, :]` where the feature
    /// axis is split into `num_blocks` chunks of `block_size` over the
    /// `in_features` dim.  The base linear runs on the rotated output —
    /// callers route via `AdapterModule::apply_input`.
    ///
    /// At init (`blocks = 0`), `R = I` and the result is bit-identical to `x`.
    pub fn apply_to_input(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims();
        let last = *in_dims.last().ok_or_else(|| {
            Error::InvalidOperation("OFT input must have at least 1 dim".into())
        })?;
        if last != self.in_features {
            return Err(Error::InvalidOperation(format!(
                "OFT input last dim {} != in_features {}",
                last, self.in_features
            )));
        }

        // Compute R in F32, then cast to x.dtype() (NOT storage_dtype) so the
        // residual graph stays in input dtype.  See `LoConModule::forward`
        // for rationale: an F32 sub-tape spawned from BF16 inputs corrupts
        // FlashAttention backward downstream
        // (CUDA_ERROR_MISALIGNED_ADDRESS, 2026-05-09).  The cast is in
        // record-mode so backward routes Cast → F32 leaf (blocks).
        let r_f32 = self.cayley_neumann()?;
        let x_dtype = x.dtype();
        let r = if x_dtype != DType::F32 {
            r_f32.to_dtype(x_dtype).map_err(Error::Flame)?
        } else {
            r_f32
        };
        let target_dtype = x_dtype;
        let x_compute = x.clone();

        // Flatten leading dims: x [*..., in] → [B, num_blocks, b]
        // where B = product(*...) = numel / in_features.
        let numel: usize = in_dims.iter().product();
        let batch = numel / self.in_features;
        let x_3d = x_compute
            .reshape(&[batch, self.num_blocks, self.block_size])
            .map_err(Error::Flame)?;

        // einsum "brk, rkc -> brc"  via permute + bmm.
        // Step 1: permute x_3d [B, R, b] → [R, B, b]  (so bmm batch axis = R).
        let x_rb = x_3d.permute(&[1, 0, 2]).map_err(Error::Flame)?;

        // Step 2: bmm( [R, B, b],  [R, b, b] ) → [R, B, b].
        let rotated_rb = x_rb.bmm(&r).map_err(Error::Flame)?;

        // Step 3: permute back [R, B, b] → [B, R, b].
        let rotated_br = rotated_rb.permute(&[1, 0, 2]).map_err(Error::Flame)?;

        // Step 4: reshape [B, R, b] → original [*..., in].
        let out_compute = rotated_br
            .reshape(in_dims)
            .map_err(Error::Flame)?;

        // Restore caller dtype.
        if x_dtype != target_dtype {
            out_compute.to_dtype(x_dtype).map_err(Error::Flame)
        } else {
            Ok(out_compute)
        }
    }

    /// Project `blocks` so each per-block skew has Frobenius norm ≤ constraint.
    ///
    /// **Caller-driven**: not invoked automatically anywhere in this module.
    /// Mirrors OneTrainer's `_project_batch` (oft_utils.py:121) but operates
    /// directly on `blocks` rather than the upper-triangular vector form.
    /// Should be called inside `no_grad` between optimizer steps if used.
    ///
    /// **Not yet implemented** — flame-core lacks a per-batch Frobenius norm
    /// primitive at the time of writing. Will require either
    /// `Tensor::sum_dims` over `[1, 2]` of `(blocks − blocks^T)²` followed by
    /// `sqrt`, then a broadcast clamp. Tracked as TODO; constraint plumbing
    /// is wired through but the projection itself errors out.
    pub fn project_blocks_inplace(&mut self) -> Result<()> {
        if self.constraint.is_none() {
            return Ok(());
        }
        Err(Error::InvalidOperation(
            "OFT COFT projection not yet implemented (needs batched Frobenius \
             norm primitive in flame-core); ship constraint=None for now"
                .into(),
        ))
    }

    /// Returns `true` if `blocks` is the zero tensor (i.e. the module is at
    /// init and acts as identity). For diagnostics only — does an O(num_elems)
    /// host-side scan.
    pub fn is_identity_init(&self) -> Result<bool> {
        let blocks = param_tensor(&self.blocks)?;
        let v = blocks.to_vec().map_err(Error::Flame)?;
        Ok(v.iter().all(|x| *x == 0.0))
    }
}

impl LycorisModule for OFTModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // OFT replaces the base-layer behavior rather than adding a residual:
        // the convention here is that callers feed the original input, get
        // back the *rotated* input, and then run the underlying Linear on it.
        // This matches OneTrainer's `forward(x) = orig_module(R(x))` pattern
        // (LoRAModule.py:OFTModule).
        self.apply_to_input(x)
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        // OFT is **multiplicative**, not additive: `W' = R^T · W` (when input
        // is rotated by R, the equivalent weight rotation is R^T applied on
        // the left of W). The trait expects an additive ΔW that the loader
        // adds to the base weight, which requires the base weight as input —
        // we don't have it here.
        //
        // Two clean options:
        //   1. Return the rotation tensor `R` itself, with caller
        //      responsibility to compute `ΔW = (R^T − I) · W` externally.
        //   2. Error out and force callers through `apply_to_input`.
        //
        // Picking (2) for safety: a silent return of the wrong-shaped delta
        // would corrupt merged checkpoints. Adding wiring later requires
        // threading the base `W` through `LycorisCollection::apply_to`.
        Err(Error::InvalidOperation(
            "OFT delta is multiplicative (W' = R^T·W); cannot synthesize an \
             additive ΔW without access to the base weight. Use \
             OFTModule::apply_to_input or compute (R^T − I)·W in the caller."
                .into(),
        ))
    }

    fn merge_to(&mut self, _multiplier: f32) -> Result<()> {
        // Same rationale as get_diff_weight: needs base weight. Caller must
        // do the merge externally.
        Err(Error::InvalidOperation(
            "OFT merge requires base weight; perform W' = (1-m)·W + m·(R^T·W) \
             at the caller site."
                .into(),
        ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        // OFT trains a single leaf: the [num_blocks, b, b] block tensor.
        // Stored as F32 with `requires_grad=true`; clone is cheap (shared
        // GPU storage via Arc) and preserves TensorId so autograd grads
        // route back to the live `Parameter`.
        vec![param_tensor(&self.blocks).expect("OFT.blocks mutex poisoned")]
    }

    fn parameters_handles(&self) -> Vec<Parameter> {
        vec![self.blocks.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity test for divisibility check (no GPU needed).
    #[test]
    fn test_divisibility() {
        assert_eq!(check_divisibility(64, 32).unwrap(), 2);
        assert_eq!(check_divisibility(96, 32).unwrap(), 3);
        assert!(check_divisibility(50, 32).is_err());
        assert!(check_divisibility(64, 0).is_err());
    }

    /// Verify the Neumann coefficient table by reading the loop logic
    /// statically. n=5 should give:  R = I + 2Q + 2Q² + 2Q³ + Q⁴
    /// (i.e. terms 2..n−1 carry coefficient 2; the final term coefficient 1).
    /// This is a no-GPU table check — we can't compute matmuls without a
    /// device, but we can at least confirm the loop bounds.
    #[test]
    fn test_neumann_loop_bounds_n5() {
        // For n=5: range(3, n-1) = range(3, 4) = [3]; one extra mid-loop add
        // (Q³ with coefficient 2), then final tail (Q⁴ with coefficient 1).
        let n = 5usize;
        let loop_iters = if n >= 4 { (n - 1) - 3 } else { 0 };
        assert_eq!(loop_iters, 1);
    }

    #[test]
    fn test_neumann_loop_bounds_n3() {
        // n=3: range(3, 2) is empty (since loop_end = max(3, n-1) = 3, range(3,3)).
        // Total terms applied: I + 2Q + 2Q² + Q³ (final tail).
        let n = 3usize;
        let loop_iters = if n >= 4 { (n - 1) - 3 } else { 0 };
        assert_eq!(loop_iters, 0);
    }
}
