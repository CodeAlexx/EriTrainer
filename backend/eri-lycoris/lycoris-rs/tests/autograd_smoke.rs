//! Autograd smoke test for the four LyCORIS adapter variants.
//!
//! P0 milestone for LyCORIS in EDv2: prove that the LoKr / LoCon / LoHa /
//! Full math primitives can record autograd correctly. This is the gate —
//! every downstream LyCORIS trainer work depends on this.
//!
//! Strategy: construct each adapter with leaf weights flipped to
//! `requires_grad_(true)`, run forward (or `get_diff_weight` for Full),
//! sum to a scalar loss, run backward, and assert that the "up"-side
//! tensor's gradient is non-zero with a finite max-abs > 1e-9.
//!
//! Rationale (init choices per algo):
//!  - LoCon: down=randn, up=zeros. d_up = scale * (x @ down)^T @ d_out;
//!           non-zero because x and down are non-zero.
//!  - LoHa : branch-1 fully random, branch-2 has w2a=randn / w2b=zeros.
//!           d_w2b = w2a^T @ (d_diff ⊙ w1) is non-zero because w1 ≠ 0.
//!  - LoKr : same idea — w1a, w1b, w2a are randn; w2b is zeros.
//!           d_w2 = sum over A axes of (d_kron · w1); w1 ≠ 0, so d_w2 ≠ 0,
//!           which propagates back to d_w2b.
//!  - Full : trivial — out = strength * diff, d_diff = strength * d_out.
//!
//! Run after qwen finishes:
//!   LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:/home/alex/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
//!     cargo test --release -p lycoris-rs --test autograd_smoke -- --nocapture

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};

use lycoris_rs::algorithms::{
    full::FullAdapter,
    locon::LoConModule,
    loha::LoHaModule,
    lokr::LoKrModule,
};
use lycoris_rs::LycorisModule;
use lycoris_rs::tensor_utils;

const PASS_THRESHOLD: f32 = 1.0e-9;

fn try_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

/// Cast an existing Tensor to BF16 storage and flip `requires_grad=true`.
fn bf16_grad(t: Tensor) -> Tensor {
    let bf = t.to_dtype(DType::BF16).expect("to_dtype BF16");
    bf.requires_grad_(true)
}

/// Make a randn BF16 leaf with `requires_grad=true`.
fn leaf_randn_bf16(
    shape: &[usize],
    std: f32,
    device: Arc<CudaDevice>,
) -> Tensor {
    let f32 = Tensor::randn(Shape::from_dims(shape), 0.0, std, device).expect("randn");
    bf16_grad(f32)
}

/// Make a zeros BF16 leaf with `requires_grad=true`.
fn leaf_zeros_bf16(shape: &[usize], device: Arc<CudaDevice>) -> Tensor {
    let z = tensor_utils::zeros_bf16(Shape::from_dims(shape), device).expect("zeros_bf16");
    z.requires_grad_(true)
}

/// Compute max(|grad|) on a 1-D / N-D tensor by reading it back to host F32.
fn max_abs_host(t: &Tensor) -> f32 {
    let host = t.to_dtype(DType::F32).expect("to F32 for host read")
        .to_vec().expect("to_vec");
    host.into_iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

#[test]
fn locon_linear_autograd_records_lora_b_grad() {
    let Some(dev) = try_device() else { eprintln!("[locon_linear] no CUDA — skipped"); return; };

    const IN: usize = 64;
    const OUT: usize = 64;
    const RANK: usize = 4;
    const ALPHA: f32 = 4.0;

    // Build the module with the standard constructor (BF16 inference leaves
    // wrapped in Parameter), then overwrite the leaves with grad-enabled
    // Parameter handles. The LycorisModule forward path reads via
    // `param.tensor()?` so the swapped-in leaf is what gets used.
    let mut m = LoConModule::new_linear(IN, OUT, RANK, Some(ALPHA), dev.clone())
        .expect("LoConModule::new_linear");
    m.down = Parameter::new(leaf_randn_bf16(&[IN, RANK], 0.1, dev.clone()));
    m.up   = Parameter::new(leaf_zeros_bf16(&[RANK, OUT], dev.clone()));

    // Input
    let x = leaf_randn_bf16(&[2, IN], 1.0, dev.clone());

    // Forward + scalar loss
    let out = m.forward(&x).expect("LoCon forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_up   = grads.get(m.up.id())  .expect("missing grad for up");
    let g_down = grads.get(m.down.id()).expect("missing grad for down");
    let mu = max_abs_host(g_up);
    let md = max_abs_host(g_down);
    println!("[LoCon-linear] grad_up max_abs={:e}, grad_down max_abs={:e}", mu, md);
    assert!(mu.is_finite() && mu > PASS_THRESHOLD,
        "LoCon-linear: lora_B (up) grad is zero / non-finite — autograd not recording");
    // down grad with up=0 propagates only through h ⇒ d_h = scale * d_out @ up.T
    // which is zero. So we DO NOT assert on d_down for LoCon — it is structurally
    // zero at the canonical LoRA init, and that's correct PyTorch behavior too.
    println!("[LoCon-linear] PASS (down grad expected zero with up=zeros init)");
}

/// LoHa step-0 grad pattern with one zero-init leaf.
///
/// Setup mirrors upstream LyCORIS Python init (`use_scalar=False`): three
/// non-zero leaves and one zero leaf in the second Hadamard branch (here,
/// `w2a = 0`). Math at step 0:
///   diff_w = (w1a @ w1b) ⊙ (w2a @ w2b) = nonzero ⊙ (0 @ w2b) = 0
///   ∂L/∂w1 = ∂L/∂diff_w · w2 = ∂L/∂diff_w · 0 = 0
///   ∂L/∂w2 = ∂L/∂diff_w · w1 ≠ 0  (w1 = w1a@w1b nonzero)
/// Backward through the two MatMuls then gives:
///   ∂L/∂w1a = 0 @ w1b^T = 0  (DEAD)
///   ∂L/∂w1b = w1a^T @ 0 = 0  (DEAD)
///   ∂L/∂w2a = ∂L/∂w2 @ w2b^T ≠ 0  (ALIVE)
///   ∂L/∂w2b = w2a^T @ ∂L/∂w2 = 0 @ ∂L/∂w2 = 0  (DEAD)
///
/// Only the zero-init leaf itself (`w2a`) gets a non-zero gradient at
/// step 0. This is identical to upstream PyTorch LyCORIS step-0 behavior
/// and is **not an autograd bug** — it's the structural saddle point the
/// algorithm starts from. After the first optimizer update drives `w2a`
/// off zero, all four matrices receive non-zero gradients.
///
/// This test exists to:
///   (a) gate that the autograd tape DOES route a gradient back to `w2a`
///       through the MatMul → Mul → MatMul chain (the cast-then-record
///       fix from `lycoris-rs ac5350d` is what makes this work for F32
///       trainable storage paired with BF16 inputs);
///   (b) document, in code, that the other three leaves are expected to
///       be exactly zero at step 0 — so a future change that "fixes"
///       this test by claiming all four should be alive is recognized
///       as a regression of either the math or the test's intent.
#[test]
fn loha_linear_autograd_records_w2a_grad() {
    let Some(dev) = try_device() else { eprintln!("[loha_linear] no CUDA — skipped"); return; };

    const IN: usize = 64;
    const OUT: usize = 64;
    const RANK: usize = 4;
    const ALPHA: f32 = 4.0;

    let mut m = LoHaModule::new_linear(IN, OUT, RANK, Some(ALPHA), dev.clone())
        .expect("LoHaModule::new_linear");
    // Upstream init (`use_scalar=False`): only `w2a` is zero.
    m.w1a = Parameter::new(leaf_randn_bf16(&[IN, RANK],  0.1, dev.clone()));
    m.w1b = Parameter::new(leaf_randn_bf16(&[RANK, OUT], 1.0, dev.clone()));
    m.w2a = Parameter::new(leaf_zeros_bf16(&[IN, RANK],       dev.clone()));
    m.w2b = Parameter::new(leaf_randn_bf16(&[RANK, OUT], 1.0, dev.clone()));

    let x = leaf_randn_bf16(&[2, IN], 1.0, dev.clone());

    let out = m.forward(&x).expect("LoHa forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_w1a = grads.get(m.w1a.id()).expect("missing grad for w1a");
    let g_w1b = grads.get(m.w1b.id()).expect("missing grad for w1b");
    let g_w2a = grads.get(m.w2a.id()).expect("missing grad for w2a");
    let g_w2b = grads.get(m.w2b.id()).expect("missing grad for w2b");
    let mw1a = max_abs_host(g_w1a);
    let mw1b = max_abs_host(g_w1b);
    let mw2a = max_abs_host(g_w2a);
    let mw2b = max_abs_host(g_w2b);
    println!(
        "[LoHa-linear] grad max_abs: w1a={:e} w1b={:e} w2a={:e} w2b={:e}",
        mw1a, mw1b, mw2a, mw2b,
    );
    // Only `w2a` (the zero-init leaf) is alive — see header comment.
    assert!(mw2a.is_finite() && mw2a > PASS_THRESHOLD,
        "LoHa-linear: w2a grad is zero / non-finite — autograd not recording \
         the (w2a @ w2b) MatMul branch back through the Hadamard product");
    // Other three are mathematically zero at step 0; assert exactly that
    // so a regression that introduces spurious non-zero grads (e.g. an
    // accidentally non-recording op leaking a constant) gets caught.
    assert!(mw1a == 0.0,
        "LoHa-linear: w1a grad must be exactly zero at step 0 (w1 chain killed by w2=0); got {:e}", mw1a);
    assert!(mw1b == 0.0,
        "LoHa-linear: w1b grad must be exactly zero at step 0 (w1 chain killed by w2=0); got {:e}", mw1b);
    assert!(mw2b == 0.0,
        "LoHa-linear: w2b grad must be exactly zero at step 0 (w2a=0 multiplies upstream); got {:e}", mw2b);
    println!("[LoHa-linear] PASS");
}

#[test]
fn lokr_linear_autograd_records_w2b_grad() {
    let Some(dev) = try_device() else { eprintln!("[lokr_linear] no CUDA — skipped"); return; };

    // Linear shape (IN, OUT) = (IM*INN, OL*OK) = (16, 8).
    // The mathematical ΔW for LoKr (PyTorch convention) is kron(W1, W2) with
    // shape [OL*OK, IM*INN] = [OUT, IN]. The Rust forward, however, applies
    // the factored multiply as `y = x @ ΔW^T` (equivalently F.linear), so the
    // user-visible mapping is: input has IN=IM*INN features, output has
    // OUT=OL*OK features. (See `forward_linear_factored` for the math.)
    const OL: usize = 4;
    const IM: usize = 4;
    const OK: usize = 2;
    const INN: usize = 4;
    const RANK: usize = 2;
    const ALPHA: f32 = 2.0;

    let in_total  = IM * INN;       // 16  (input features)
    let out_total = OL * OK;        //  8  (output features)

    // Factorized W1: w1a [OL, R], w1b [R, IM]  → W1 [OL, IM]
    let w1a = leaf_randn_bf16(&[OL, RANK], 1.0, dev.clone());
    let w1b = leaf_randn_bf16(&[RANK, IM], 0.5, dev.clone());
    // Factorized W2: w2a [OK, R], w2b [R, INN] → W2 [OK, INN]
    let w2a = leaf_randn_bf16(&[OK, RANK], 1.0, dev.clone());
    let w2b = leaf_zeros_bf16(&[RANK, INN], dev.clone());

    let m = LoKrModule {
        w1: None,
        w1a: Some(Parameter::new(w1a)),
        w1b: Some(Parameter::new(w1b)),
        w2: None,
        w2a: Some(Parameter::new(w2a)),
        w2b: Some(Parameter::new(w2b)),
        t2: None,
        rank: RANK,
        alpha: ALPHA,
        device: dev.clone(),
        shape: ((OL, OK), (IM, INN)),
        is_conv: false,
    };

    // x: [batch, IM*INN] → forward yields [batch, OL*OK].
    let x = leaf_randn_bf16(&[2, in_total], 1.0, dev.clone());
    let _ = out_total; // silence unused if assertions below don't reference

    let out = m.forward(&x).expect("LoKr forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    // We want the grad on the W2 "up" side (w2b — all-zero leaf).
    let w2b_id = m.w2b.as_ref().unwrap().id();
    let w1b_id = m.w1b.as_ref().unwrap().id();
    let g_w2b = grads.get(w2b_id).expect("missing grad for w2b");
    let g_w1b = grads.get(w1b_id).expect("missing grad for w1b");
    let mw2b = max_abs_host(g_w2b);
    let mw1b = max_abs_host(g_w1b);
    println!("[LoKr-linear] grad_w2b max_abs={:e}, grad_w1b max_abs={:e}", mw2b, mw1b);
    assert!(mw2b.is_finite() && mw2b > PASS_THRESHOLD,
        "LoKr-linear: w2b grad is zero / non-finite — autograd not recording");
    assert!(mw1b.is_finite() && mw1b > PASS_THRESHOLD,
        "LoKr-linear: w1b grad is zero / non-finite — autograd not recording");
    println!("[LoKr-linear] PASS");
}

#[test]
fn full_autograd_records_diff_grad() {
    let Some(dev) = try_device() else { eprintln!("[full] no CUDA — skipped"); return; };

    // Full adapter is the trivial case: out = base + strength * diff.
    // Construct directly with a grad-enabled diff leaf.
    const IN: usize = 64;
    const OUT: usize = 64;
    const STRENGTH: f32 = 1.5;

    let diff = leaf_randn_bf16(&[IN, OUT], 0.1, dev.clone());
    let diff_id = diff.id();
    let f = FullAdapter {
        diff: Parameter::new(diff),
        diff_b: None,
    };

    // delta_weight returns strength * diff; sum it as a scalar loss.
    let delta = f.delta_weight(STRENGTH).expect("delta_weight");
    let loss = delta.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_diff = grads.get(diff_id).expect("missing grad for diff");
    let mg = max_abs_host(g_diff);
    println!("[Full] grad_diff max_abs={:e}", mg);
    assert!(mg.is_finite() && mg > PASS_THRESHOLD,
        "Full: diff grad is zero / non-finite — autograd not recording");
    println!("[Full] PASS");
}

/// Phase 2b gradient-isolation regression test.
///
/// Verifies that calling `set_data` on a `Parameter` handle returned by
/// `LycorisModule::parameters_handles()` actually mutates the algorithm's
/// internal leaf storage (i.e. subsequent `forward` reads see the new data).
/// Prior to the migration, leaves were stored as bare `Tensor` and
/// `LycorisLinear::to_parameters` wrapped clones in fresh `Parameter`s — so
/// optimizer `set_data` calls landed on a throwaway wrapper, never reaching
/// the adapter, and `forward` kept reading the init values. This test fails
/// loudly if the bug is reintroduced.
#[test]
fn locon_set_data_through_handle_propagates_to_forward() {
    let Some(dev) = try_device() else { eprintln!("[locon_setdata] no CUDA — skipped"); return; };

    const IN: usize = 8;
    const OUT: usize = 8;
    const RANK: usize = 2;

    // Build the module the way EDv2 trainers do — F32 leaves via
    // `new_linear_for_training`, then collect `Parameter` handles.
    let m = LoConModule::new_linear_for_training(
        IN, OUT, RANK, Some(RANK as f32), dev.clone(), lycoris_rs::tensor_utils::StorageDtype::F32,
    ).expect("LoConModule::new_linear_for_training");

    let handles = m.parameters_handles();
    assert_eq!(handles.len(), 2, "LoCon: expected [down, up]");
    let up_handle = handles[1].clone();

    // Initial up tensor is zeros (canonical LoRA init).
    let up_before = m.up.tensor().expect("up before");
    let up_before_v = up_before.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let nonzero_before = up_before_v.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(nonzero_before, 0, "up should start zero");

    // Mutate via the handle (mimics AdamW8bit's update path).
    let one_tensor = Tensor::randn(
        Shape::from_dims(&[RANK, OUT]), 0.0, 1.0, dev.clone(),
    ).expect("randn for set_data").to_dtype(flame_core::DType::F32).expect("F32");
    up_handle.set_data(one_tensor).expect("set_data");

    // Forward path reads via `param.tensor()?` — should see the new data.
    let up_after = m.up.tensor().expect("up after");
    let up_after_v = up_after.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let nonzero_after = up_after_v.iter().filter(|&&x| x != 0.0).count();
    assert!(
        nonzero_after > 0,
        "Phase 2b regression: set_data on handle did not propagate to adapter — \
         optimizer updates would be silently dropped. nonzero_after={}", nonzero_after
    );
    println!("[locon_setdata] PASS: set_data via handle reached adapter (nonzero_after={})", nonzero_after);
}
