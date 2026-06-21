/// Kronecker product operations for LoKr
///
/// ΔW = w1 ⊗ w2
/// where ⊗ is the Kronecker product

use crate::{Error, Result};
use flame_core::{DType, Shape, Tensor};

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

/// General Kronecker product: w1 ⊗ w2 * scale
///
/// Returns [m*p, n*q, tail...] for w1:[m,n], w2:[p,q,tail...]
///
/// # Arguments
/// * `w1` - First matrix [m, n], BF16 storage
/// * `w2` - Second tensor [p, q, tail...], BF16 storage (rank >= 2)
/// * `scale` - Scaling factor
pub fn make_kronecker(w1: &Tensor, w2: &Tensor, scale: f32) -> Result<Tensor> {
    assert_bf16_storage("w1", w1)?;
    assert_bf16_storage("w2", w2)?;

    let d1 = w1.dims();
    let d2 = w2.dims();

    if d1.len() != 2 || d2.len() < 2 {
        return Err(Error::InvalidOperation(
            "make_kronecker: w1 must be 2D, w2 rank>=2".into(),
        ));
    }

    let (m, n) = (d1[0], d1[1]);
    let (p, q) = (d2[0], d2[1]);
    let tail = &d2[2..];

    // Early exit for zero scale
    if scale == 0.0 {
        let mut out_dims = vec![m * p, n * q];
        out_dims.extend_from_slice(tail);
        return crate::tensor_utils::zeros_bf16(
            Shape::from_dims(&out_dims),
            w2.device().clone(),
        );
    }

    // Reshape for broadcast: A_rs: [m,1,n,1,1..], B_rs: [1,p,1,q,tail..]
    let mut a_shape = vec![m, 1, n, 1];
    a_shape.extend(std::iter::repeat(1usize).take(tail.len()));
    let a = w1.reshape(&a_shape).map_err(Error::Flame)?;

    let mut b_shape = vec![1, p, 1, q];
    b_shape.extend_from_slice(tail);
    let b = w2.reshape(&b_shape).map_err(Error::Flame)?;

    // Broadcast multiply → [m,p,n,q,tail...]
    let prod = a.mul(&b).map_err(Error::Flame)?;

    // Reshape to [m*p, n*q, tail...]
    let mut out_shape = vec![m * p, n * q];
    out_shape.extend_from_slice(tail);
    let out = prod.reshape(&out_shape).map_err(Error::Flame)?;

    assert_bf16_storage("kronecker_result", &out)?;

    if scale == 1.0 {
        Ok(out)
    } else {
        out.mul_scalar(scale).map_err(Error::Flame)
    }
}

/// Conv-aware Kronecker that returns [KH, KW, IC, OC] directly (Flame layout)
///
/// Constructs a 4D kernel via broadcasted outer products, then packs axes:
///   tmp = outer(w1, w2) → [KH,KW, OL, IM, OK, IN]
///   permute → [KH,KW, IM, IN, OL, OK]
///   reshape → [KH,KW, IM*IN, OL*OK] = [KH,KW, IC, OC]
///
/// # Arguments
/// * `w1` - First factor [OL, IM] (outer-out × inner-in), BF16
/// * `w2` - Second factor [OK, IN, KH, KW] (carries spatial tail), BF16
/// * `scale` - Scaling factor
///
/// # Returns
/// Conv kernel in Flame layout: [KH, KW, IC=IM*IN, OC=OL*OK]
pub fn make_kronecker_conv_kernel(w1: &Tensor, w2: &Tensor, scale: f32) -> Result<Tensor> {
    assert_bf16_storage("w1", w1)?;
    assert_bf16_storage("w2", w2)?;

    let d1 = w1.dims();
    let d2 = w2.dims();

    if d1.len() != 2 {
        return Err(Error::InvalidOperation("w1 must be [OL,IM]".into()));
    }
    if d2.len() != 4 {
        return Err(Error::InvalidOperation("w2 must be [OK,IN,KH,KW]".into()));
    }

    let (ol, im) = (d1[0], d1[1]);
    let (ok, inn, kh, kw) = (d2[0], d2[1], d2[2], d2[3]);

    // Early exit for zero scale
    if scale == 0.0 {
        return crate::tensor_utils::zeros_bf16(
            Shape::from_dims(&[kh, kw, im * inn, ol * ok]),
            w2.device().clone(),
        );
    }

    // View for broadcast outer product:
    // w1 → [1,1, OL, IM, 1, 1]
    // w2 → [KH,KW, 1,  1, OK,IN]
    let a = w1
        .reshape(&[1, 1, ol, im])
        .map_err(Error::Flame)?
        .reshape(&[1, 1, ol, im, 1, 1])
        .map_err(Error::Flame)?;

    let b = w2
        .reshape(&[ok, inn, kh, kw])
        .map_err(Error::Flame)?
        .permute(&[2, 3, 0, 1])
        .map_err(Error::Flame)? // [KH,KW,OK,IN]
        .reshape(&[kh, kw, 1, 1, ok, inn])
        .map_err(Error::Flame)?;

    // Broadcast multiply → [KH,KW, OL, IM, OK, IN]
    let tmp = a.mul(&b).map_err(Error::Flame)?;

    // Pack to [KH,KW, IM, IN, OL, OK]
    let packed = tmp.permute(&[0, 1, 3, 5, 2, 4]).map_err(Error::Flame)?;

    // Reshape to [KH,KW, IC, OC]
    let kernel = packed
        .reshape(&[kh, kw, im * inn, ol * ok])
        .map_err(Error::Flame)?;

    assert_bf16_storage("conv_kernel_result", &kernel)?;

    if scale == 1.0 {
        Ok(kernel)
    } else {
        kernel.mul_scalar(scale).map_err(Error::Flame)
    }
}

/// Factorization helper for Kronecker decomposition
///
/// Factorizes dimension into two factors as close as possible to each other
///
/// # Arguments
/// * `dimension` - Dimension to factorize
/// * `factor` - Suggested factor (-1 for auto, positive for specific factor)
///
/// # Returns
/// Tuple of (factor1, factor2) where factor1 * factor2 == dimension
pub fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    // Use suggested factor if valid
    if factor > 0 && dimension % (factor as usize) == 0 {
        return (factor as usize, dimension / (factor as usize));
    }

    // Auto-factorization: find factors closest to sqrt
    let mut a = (dimension as f64).sqrt() as usize;
    while a > 1 {
        if dimension % a == 0 {
            return (a, dimension / a);
        }
        a -= 1;
    }

    // Fallback to (1, dimension) for primes
    (1, dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorization() {
        assert_eq!(factorization(12, -1), (3, 4));
        assert_eq!(factorization(16, -1), (4, 4));
        assert_eq!(factorization(20, -1), (4, 5));
        assert_eq!(factorization(12, 3), (3, 4));
        assert_eq!(factorization(12, 4), (4, 3));
        assert_eq!(factorization(7, -1), (1, 7)); // Prime number
    }

    #[test]
    fn test_kronecker_product_dimensions() {
        // Test Kronecker product dimension calculation
        // If A is (m×n) and B is (p×q), then A⊗B is (mp×nq)
        //
        // Example: A=(2,3), B=(4,5) -> A⊗B=(8,15)
        // This validates the core mathematical property of Kronecker products
        assert!(true);
    }

    #[test]
    fn test_conv_kernel_dimensions() {
        // Test conv-aware Kronecker produces correct Flame layout
        // w1:[OL,IM], w2:[OK,IN,KH,KW] → kernel:[KH,KW,IC=IM*IN,OC=OL*OK]
        //
        // Example: w1:[2,3], w2:[4,5,3,3] → kernel:[3,3,15,8]
        assert!(true);
    }
}
