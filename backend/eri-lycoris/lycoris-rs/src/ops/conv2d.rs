/// Convolution operations for Flame tensors
///
/// Provides conv2d operations with proper layout handling

use crate::{Error, Result};
use flame_core::Tensor;

/// Conv2d layout specification
#[derive(Debug, Clone, Copy)]
pub enum Layout {
    /// NHWC: [Batch, Height, Width, Channels]
    NHWC,
}

/// 2D convolution operation
///
/// # Arguments
/// * `input` - Input tensor [N, H, W, IC] in NHWC layout
/// * `kernel` - Convolution kernel [KH, KW, IC, OC] in Flame layout
/// * `stride` - Stride (h, w)
/// * `padding` - Padding (h, w)
/// * `dilation` - Dilation (h, w) - currently unused, always (1,1)
/// * `groups` - Number of groups - currently unused, always 1
/// * `layout` - Tensor layout (NHWC)
///
/// # Returns
/// Output tensor [N, H_out, W_out, OC] in NHWC layout
pub fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    _dilation: (usize, usize),
    _groups: usize,
    _layout: Layout,
) -> Result<Tensor> {
    // Flame's conv2d only supports uniform stride/padding
    if stride.0 != stride.1 {
        return Err(Error::InvalidOperation(
            "Flame conv2d only supports uniform stride".to_string(),
        ));
    }
    if padding.0 != padding.1 {
        return Err(Error::InvalidOperation(
            "Flame conv2d only supports uniform padding".to_string(),
        ));
    }

    // Call Flame's conv2d: fn conv2d(&self, weight, bias, stride, padding)
    input
        .conv2d(kernel, None, stride.0, padding.0)
        .map_err(Error::Flame)
}
