use crate::{Error, Result};
use flame_core::Tensor;

/// Tensor layout for convolution operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// NCHW: (batch, channels, height, width) - Flame/PyTorch default
    NCHW,
    /// NHWC: (batch, height, width, channels) - cuDNN optimized
    NHWC,
}

/// Layout converter for NHWC ↔ NCHW transformations
pub struct LayoutConverter;

impl LayoutConverter {
    /// Convert NCHW to NHWC
    /// Input: (N, C, H, W) -> Output: (N, H, W, C)
    pub fn nchw_to_nhwc(input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Expected 4D tensor for NCHW->NHWC conversion, got {}D",
                dims.len()
            )));
        }

        // Permute dimensions: [0, 1, 2, 3] -> [0, 2, 3, 1]
        input
            .permute(&[0, 2, 3, 1])
            .map_err(|e| Error::Flame(e))
    }

    /// Convert NHWC to NCHW
    /// Input: (N, H, W, C) -> Output: (N, C, H, W)
    pub fn nhwc_to_nchw(input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Expected 4D tensor for NHWC->NCHW conversion, got {}D",
                dims.len()
            )));
        }

        // Permute dimensions: [0, 1, 2, 3] -> [0, 3, 1, 2]
        input
            .permute(&[0, 3, 1, 2])
            .map_err(|e| Error::Flame(e))
    }

    /// Apply layout conversion based on source and target layouts
    pub fn convert(
        input: &Tensor,
        from: TensorLayout,
        to: TensorLayout,
    ) -> Result<Tensor> {
        if from == to {
            return input.clone_result().map_err(|e| Error::Flame(e));
        }

        match (from, to) {
            (TensorLayout::NCHW, TensorLayout::NHWC) => Self::nchw_to_nhwc(input),
            (TensorLayout::NHWC, TensorLayout::NCHW) => Self::nhwc_to_nchw(input),
            _ => unreachable!("Layout conversion already handled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_round_trip() {
        // This test will be implemented once we have tensor operations
        // It should verify that NCHW -> NHWC -> NCHW preserves data
    }
}
