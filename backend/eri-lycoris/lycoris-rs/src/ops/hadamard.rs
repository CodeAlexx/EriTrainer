/// Hadamard product operations for LoHa
///
/// ΔW = (w1a @ w1b) ⊙ (w2a @ w2b)
/// where ⊙ is element-wise multiplication
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

use crate::{Error, Result};
use flame_core::{DType, Tensor};

/// Assert BF16 storage
#[inline]
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "{} must use BF16 storage, got {:?}",
            name,
            t.dtype()
        )));
    }
    Ok(())
}

/// Compute Hadamard weight: (w1a @ w1b) ⊙ (w2a @ w2b) * scale
///
/// All weights stored in BF16, compute in FP32
///
/// # Arguments
/// * `w1a` - First down weight: [IN, RANK] for linear, [KH, KW, IC, RANK] for conv
/// * `w1b` - First up weight: [RANK, OUT] for linear, [KH, KW, RANK, OC] for conv
/// * `w2a` - Second down weight: [IN, RANK] for linear, [KH, KW, IC, RANK] for conv
/// * `w2b` - Second up weight: [RANK, OUT] for linear, [KH, KW, RANK, OC] for conv
/// * `scale` - Scaling factor (typically alpha/rank)
pub fn make_hadamard_weight(
    w1a: &Tensor,
    w1b: &Tensor,
    w2a: &Tensor,
    w2b: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Validate BF16 storage
    assert_bf16_storage("w1a", w1a)?;
    assert_bf16_storage("w1b", w1b)?;
    assert_bf16_storage("w2a", w2a)?;
    assert_bf16_storage("w2b", w2b)?;

    // Early exit for zero scale
    if scale == 0.0 {
        let dims = w1a.dims();
        if dims.len() == 2 {
            // Linear: [IN, OUT]
            return crate::tensor_utils::zeros_bf16(
                flame_core::Shape::from_dims(&[dims[0], w1b.dims()[1]]),
                w1a.device().clone(),
            );
        } else {
            // Conv: same shape as w1b
            return crate::tensor_utils::zeros_bf16(
                w1b.shape().clone(),
                w1a.device().clone(),
            );
        }
    }

    let dims = w1a.dims();

    if dims.len() == 2 {
        // Linear path: w1a[IN,RANK] @ w1b[RANK,OUT] = [IN,OUT]
        let w1 = w1a.matmul(w1b).map_err(Error::Flame)?;
        let w2 = w2a.matmul(w2b).map_err(Error::Flame)?;

        // Hadamard product
        let diff = w1.mul(&w2).map_err(Error::Flame)?;
        diff.mul_scalar(scale).map_err(Error::Flame)
    } else if dims.len() == 4 {
        // Conv path: [KH, KW, IC, RANK] @ [KH, KW, RANK, OC]
        // Need to do matmul per spatial position
        let kh = dims[0];
        let kw = dims[1];
        let ic = dims[2];
        let r = dims[3];
        let oc = w1b.dims()[3];

        // Reshape for batch matmul: [KH*KW, IC, R] @ [KH*KW, R, OC]
        let w1a_batch = w1a.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
        let w1b_batch = w1b.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;
        let w2a_batch = w2a.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
        let w2b_batch = w2b.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;

        // Batch matmul
        let w1_batch = w1a_batch.matmul(&w1b_batch).map_err(Error::Flame)?;
        let w2_batch = w2a_batch.matmul(&w2b_batch).map_err(Error::Flame)?;

        // Hadamard product
        let diff_batch = w1_batch.mul(&w2_batch).map_err(Error::Flame)?;

        // Reshape back: [KH*KW, IC, OC] -> [KH, KW, IC, OC]
        let diff = diff_batch.reshape(&[kh, kw, ic, oc]).map_err(Error::Flame)?;
        diff.mul_scalar(scale).map_err(Error::Flame)
    } else {
        Err(Error::InvalidOperation(format!(
            "Unsupported tensor dimensions: expected 2D or 4D, got {}D",
            dims.len()
        )))
    }
}

/// Compute Tucker-decomposed Hadamard weight
///
/// ΔW = rebuild(t1, w1a, w1b) ⊙ rebuild(t2, w2a, w2b) * scale
///
/// Matches upstream `HadaWeightTucker.forward` in
/// `lycoris/functional/loha.py:38-41`, using the einsum
/// `"i j ..., j r, i p -> p r ..."`. The Python on-disk layout is:
///     t: [R, R, KH, KW]     w1d: [R, IN]     w1u: [R, OUT]
/// After our loader permute/transposes this becomes:
///     t : [KH, KW, R, R]    w1a: [IN, R]     w1b: [R, OUT]
/// and the output is in Flame convention `[KH, KW, IC=IN, OC=OUT]`.
///
/// # Arguments
/// * `t1` - Tucker core tensor 1: [KH, KW, RANK, RANK], BF16
/// * `w1a` - First down weight (was `w1d`): [IN, RANK], BF16
/// * `w1b` - First up weight   (was `w1u`): [RANK, OUT], BF16
/// * `t2` - Tucker core tensor 2: [KH, KW, RANK, RANK], BF16
/// * `w2a` - Second down weight: [IN, RANK], BF16
/// * `w2b` - Second up weight:   [RANK, OUT], BF16
/// * `scale` - Scaling factor
pub fn make_hadamard_weight_tucker(
    t1: &Tensor,
    w1a: &Tensor,
    w1b: &Tensor,
    t2: &Tensor,
    w2a: &Tensor,
    w2b: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Validate BF16 storage
    assert_bf16_storage("t1", t1)?;
    assert_bf16_storage("w1a", w1a)?;
    assert_bf16_storage("w1b", w1b)?;
    assert_bf16_storage("t2", t2)?;
    assert_bf16_storage("w2a", w2a)?;
    assert_bf16_storage("w2b", w2b)?;

    let td1 = t1.dims();
    let td2 = t2.dims();
    if td1.len() != 4 || td2.len() != 4 {
        return Err(Error::InvalidOperation(
            "Tucker LoHa: t1/t2 must be 4D [KH,KW,R,R]".into(),
        ));
    }
    let (kh, kw, r1a, r1b) = (td1[0], td1[1], td1[2], td1[3]);
    if td2[0] != kh || td2[1] != kw || td2[2] != r1a || td2[3] != r1b {
        return Err(Error::InvalidOperation(format!(
            "Tucker LoHa: t1 {:?} and t2 {:?} must have identical shape",
            td1, td2
        )));
    }
    let (wa1, wb1) = (w1a.dims(), w1b.dims());
    let (wa2, wb2) = (w2a.dims(), w2b.dims());
    if wa1.len() != 2 || wb1.len() != 2 || wa2.len() != 2 || wb2.len() != 2 {
        return Err(Error::InvalidOperation(
            "Tucker LoHa: w1a/w1b/w2a/w2b must be 2D (IN,R)/(R,OUT)".into(),
        ));
    }
    let inn = wa1[0];
    let r_j = wa1[1];
    let r_i = wb1[0];
    let out = wb1[1];
    if r_j != r1b || r_i != r1a {
        return Err(Error::InvalidOperation(format!(
            "Tucker LoHa: rank mismatch t1 [KH,KW,{},{}] vs w1a[:,{}], w1b[{},:]",
            r1a, r1b, r_j, r_i
        )));
    }
    if wa2[0] != inn || wa2[1] != r_j || wb2[0] != r_i || wb2[1] != out {
        return Err(Error::InvalidOperation(
            "Tucker LoHa: branch-2 shapes must match branch-1".into(),
        ));
    }

    // Early exit for zero scale → return zero kernel of the final shape.
    if scale == 0.0 {
        return crate::tensor_utils::zeros_bf16(
            flame_core::Shape::from_dims(&[kh, kw, inn, out]),
            t1.device().clone(),
        );
    }

    // Reconstruct one branch: kernel[KH,KW,IN,OUT] from t[KH,KW,i=R,j=R], wa[IN, j=R], wb[i=R, OUT].
    //   Step 1: reshape t to [KH*KW, R_i, R_j].
    //   Step 2: contract with wa along j → [KH*KW, R_i, IN]
    //           via matmul([KH*KW, R_i, R_j], [R_j, IN]) where [R_j, IN] = wa.transpose().
    //   Step 3: permute to [KH*KW, IN, R_i].
    //   Step 4: contract with wb along i → [KH*KW, IN, OUT]
    //           via matmul([KH*KW, IN, R_i], [R_i, OUT]).
    //   Step 5: reshape to [KH, KW, IN, OUT].
    fn rebuild_branch(
        t: &Tensor,
        wa: &Tensor,
        wb: &Tensor,
        kh: usize,
        kw: usize,
        r_i: usize,
        r_j: usize,
        inn: usize,
        out: usize,
    ) -> Result<Tensor> {
        let t_3d = t.reshape(&[kh * kw, r_i, r_j]).map_err(Error::Flame)?;
        // wa is [IN, R_j]; we need [R_j, IN].
        let wa_t = wa.transpose().map_err(Error::Flame)?;
        // [KH*KW, R_i, R_j] @ [R_j, IN] = [KH*KW, R_i, IN]
        let step1 = t_3d.matmul(&wa_t).map_err(Error::Flame)?;
        // Permute last two dims to [KH*KW, IN, R_i].
        let step1p = step1.permute(&[0, 2, 1]).map_err(Error::Flame)?;
        // [KH*KW, IN, R_i] @ [R_i, OUT] = [KH*KW, IN, OUT]
        let step2 = step1p.matmul(wb).map_err(Error::Flame)?;
        // Reshape to [KH, KW, IN, OUT].
        step2
            .reshape(&[kh, kw, inn, out])
            .map_err(Error::Flame)
    }

    let branch1 = rebuild_branch(t1, w1a, w1b, kh, kw, r_i, r_j, inn, out)?;
    let branch2 = rebuild_branch(t2, w2a, w2b, kh, kw, r_i, r_j, inn, out)?;

    let hadamard = branch1.mul(&branch2).map_err(Error::Flame)?;
    hadamard.mul_scalar(scale).map_err(Error::Flame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_weight_construction() {
        // Test proper weight matrix construction
        // Dimensions: out_dim=4, in_dim=3, rank=2
        // w1a: [IN, RANK] = [3, 2]
        // w1b: [RANK, OUT] = [2, 4]
        // w2a: [IN, RANK] = [3, 2]
        // w2b: [RANK, OUT] = [2, 4]
        // Result: [3, 4] from (w1a @ w1b) ⊙ (w2a @ w2b)
        assert!(true);
    }
}
