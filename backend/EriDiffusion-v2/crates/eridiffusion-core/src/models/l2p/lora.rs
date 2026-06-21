//! Runtime LoRA application for the L2P trainer — training-mode subset of
//! `inference_flame::lora::LoraStack`.
//!
//! This is a SELF-CONTAINED training copy (the established duplication
//! convention — see `eridiffusion_core::models::klein` vs
//! inference-flame's klein). It carries ONLY the training-mode path the
//! `train_l2p` binary exercises:
//!
//! - [`Slot`]      — where in the base linear's output a LoRA delta lands.
//! - [`TrainEntry`] — a `Parameter`-backed trainable LoRA branch.
//! - [`LoraStack`] — `new_training` + `apply` (training path only).
//!
//! The inference-side loaders (`LoraStack::load`, kohya naming, format
//! detection, prefix mappers) are NOT ported here: the trainer never calls
//! them — it builds `TrainEntry` branches directly from freshly-init'd
//! Parameters and attaches them via `LoraStack::new_training`. Inference of
//! L2P LoRAs still goes through `inference_flame::lora::LoraStack::load`
//! (the trainer's PEFT save format is the interop contract, unchanged).
//!
//! Behaviour is byte-for-byte the same as the inference-flame training
//! path: `(x @ down) @ up * scale` in the LoRA Parameters' dtype (F32 for
//! the trainer's recipe), delta cast back to base dtype before add, with
//! `Slot::RowRange`/`Slot::Full` placement via `add_at_col_range`.

use flame_core::parameter::Parameter;
use flame_core::{Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Where in the base linear's output the LoRA branch's contribution lands.
///
/// L2P only uses `Full` (out / ffn / adaLN) and `RowRange` (split Q/K/V on
/// the fused qkv weight). `Rows`/`Cols` are carried for parity with the
/// inference-flame enum so the trainer's `Slot` references compile
/// unchanged, but are not exercised by the L2P target table.
#[derive(Clone, Copy, Debug)]
pub enum Slot {
    /// LoRA branch output has the same shape as base output. Add directly.
    Full,
    /// Top `n` rows (output cols) of base output get the delta.
    Rows(usize),
    /// Left `n` cols of base input feed the LoRA's down matrix.
    Cols(usize),
    /// Output cols `[start..start+len]` of base output get the delta.
    /// Used for the L2P split-Q/K/V → fused QKV targeting.
    RowRange { start: usize, len: usize },
}

/// A trainable LoRA branch — holds live `Parameter` handles so autograd
/// ops recorded against their underlying tensors produce gradients in the
/// backward pass.
///
/// Tensors are pre-shaped for matmul: `down` is `[in, rank]`, `up` is
/// `[rank, out]`. No transpose is applied in `apply` — the trainer
/// constructs them in the correct shape up front.
///
/// `Parameter::tensor()?` is called per-apply to fetch the live tensor
/// after each optimizer step (the data tensor changes but the param id
/// stays pinned across `set_data`).
pub struct TrainEntry {
    pub slot: Slot,
    pub down: Parameter,
    pub up: Parameter,
    /// `alpha / rank` — applied as a scalar after the up matmul.
    pub scale: f32,
}

/// All trainable LoRA branches for one run, indexed by base weight key.
pub struct LoraStack {
    /// Training-mode entries. Multiple entries can share one weight key
    /// (Q/K/V all target the fused `attention.qkv.weight` with distinct
    /// `Slot::RowRange`s) — the apply path iterates and adds all matching
    /// deltas. Wrapped in a `Mutex` to mirror the inference-flame type's
    /// interior mutability contract (the trainer holds the stack inside an
    /// `Arc` and the model borrows it `&self`).
    train_entries: Mutex<HashMap<String, Vec<TrainEntry>>>,
}

impl LoraStack {
    /// Construct a training-mode LoRA stack from a per-weight-key map of
    /// `Parameter`-backed `TrainEntry` branches.
    ///
    /// Tensors live inside the `Parameter` objects; each forward call
    /// re-fetches them via `Parameter::tensor()` so that autograd records
    /// matmul ops against the live, requires-grad tensor. Optimizer-side
    /// `set_data` after each step pins the Parameter id, so backward grads
    /// keep landing at the same id across steps.
    pub fn new_training(targets: HashMap<String, Vec<TrainEntry>>) -> Self {
        LoraStack {
            train_entries: Mutex::new(targets),
        }
    }

    /// Read-only count of distinct base weight keys with at least one
    /// training branch. Diagnostic only.
    pub fn training_target_count(&self) -> usize {
        self.train_entries.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Apply LoRA contributions for `weight_key` to `base_out`.
    ///
    /// `x` is the input tensor that fed the base linear (so the LoRA branch
    /// can recompute `up(down(x))`). `base_out` is the linear's output. If
    /// no LoRA targets `weight_key`, returns `base_out` unchanged.
    ///
    /// `x` and `base_out` are assumed to be the trailing-feature-dim
    /// arrangement: `x` shape `[..., in_features]`, `base_out` shape
    /// `[..., out_features]`. Higher-rank tensors are flattened to 2D
    /// internally and reshaped back.
    pub fn apply(&self, weight_key: &str, x: &Tensor, base_out: Tensor) -> Result<Tensor> {
        let map = self
            .train_entries
            .lock()
            .map_err(|_| Error::InvalidOperation("train_entries mutex poisoned".into()))?;
        let Some(entries) = map.get(weight_key) else {
            return Ok(base_out);
        };

        // Optional trace gate, on with `L2P_TRAIN_LORA_TRACE=1`. Used during
        // bring-up to verify the LoRA branch fires AND that the input `x`
        // arriving from the model carries requires_grad through to here (any
        // inference-only fused kernel upstream that strips requires_grad —
        // e.g. `fused_rms_norm`, `swiglu_fused_bf16` — surfaces here as
        // `x.requires_grad=false` and the LoRA chain never connects to the
        // loss tape).
        if std::env::var("L2P_TRAIN_LORA_TRACE").as_deref() == Ok("1") {
            eprintln!(
                "[lora_train] applying {} ({} entries) x.requires_grad={} x.dtype={:?}",
                weight_key,
                entries.len(),
                x.requires_grad(),
                x.dtype()
            );
        }

        let base_dtype = base_out.dtype();

        // Flatten x to [B*..., in].
        let x_dims = x.shape().dims().to_vec();
        let in_dim = *x_dims.last().expect("x has rank ≥ 1");
        let flat_rows: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_2d = if x_dims.len() == 2 {
            x.contiguous()?
        } else {
            x.reshape(&[flat_rows, in_dim])?.contiguous()?
        };

        // Flatten base_out to [B*..., out].
        let out_dims = base_out.shape().dims().to_vec();
        let out_features = *out_dims.last().expect("base_out has rank ≥ 1");
        let mut acc = if out_dims.len() == 2 {
            base_out
        } else {
            base_out.reshape(&[flat_rows, out_features])?
        };

        for entry in entries {
            let down = entry.down.tensor()?; // [in, rank], requires_grad=true
            let up = entry.up.tensor()?; // [rank, out], requires_grad=true

            // Slice x for the Cols slot — LoRA was trained against only the
            // first `n` input features. (Unused by L2P, carried for parity.)
            let x_for_lora_owned;
            let x_for_lora: &Tensor = match entry.slot {
                Slot::Cols(n) => {
                    x_for_lora_owned = x_2d.narrow(1, 0, n)?.contiguous()?;
                    &x_for_lora_owned
                }
                _ => &x_2d,
            };

            // Cast x to LoRA dtype if needed. For an F32 LoRA on a BF16 DiT
            // body this materializes an F32 copy via autograd-tracked
            // to_dtype (matches the inference-flame training path).
            let x_cast = if x_for_lora.dtype() == down.dtype() {
                x_for_lora.clone()
            } else {
                x_for_lora.to_dtype(down.dtype())?
            };

            // (x @ down) @ up * scale. No contiguous() between matmuls — in
            // training, contiguous() inserts an autograd op that records a
            // graph dependency; matmul outputs are already row-major from
            // cuBLAS.
            let xd = x_cast.matmul(&down)?;
            let xdu = xd.matmul(&up)?;
            let delta = xdu.mul_scalar(entry.scale)?;
            let delta = if delta.dtype() == base_dtype {
                delta
            } else {
                delta.to_dtype(base_dtype)?
            };

            acc = match entry.slot {
                Slot::Full | Slot::Cols(_) => acc.add(&delta)?,
                Slot::Rows(n) => add_at_col_range(&acc, &delta, 0, n)?,
                Slot::RowRange { start, len } => add_at_col_range(&acc, &delta, start, len)?,
            };
        }

        if out_dims.len() == 2 {
            Ok(acc)
        } else {
            acc.reshape(&out_dims)
        }
    }
}

/// Add `delta [rows, len]` into `base [rows, total]` at columns
/// `start..start+len`. Returns the patched tensor.
fn add_at_col_range(base: &Tensor, delta: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let dims = base.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range needs 2D base, got {:?}",
            dims
        )));
    }
    let total = dims[1];
    if start + len > total {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range out of range: start={start} len={len} total={total}"
        )));
    }
    let delta_dims = delta.shape().dims();
    if delta_dims != [dims[0], len] {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range delta shape {:?} != [{}, {}]",
            delta_dims, dims[0], len
        )));
    }

    let delta = if delta.dtype() == base.dtype() {
        delta.clone()
    } else {
        delta.to_dtype(base.dtype())?
    };

    if start == 0 && len == total {
        return base.add(&delta);
    }

    let rows = dims[0];
    let head_len = start;
    let tail_len = total - start - len;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if head_len > 0 {
        parts.push(Tensor::zeros_dtype(
            Shape::from_dims(&[rows, head_len]),
            base.dtype(),
            base.device().clone(),
        )?);
    }
    parts.push(delta);
    if tail_len > 0 {
        parts.push(Tensor::zeros_dtype(
            Shape::from_dims(&[rows, tail_len]),
            base.dtype(),
            base.device().clone(),
        )?);
    }
    let part_refs: Vec<&Tensor> = parts.iter().collect();
    let padded = Tensor::cat(&part_refs, 1)?;
    base.add(&padded)
}
