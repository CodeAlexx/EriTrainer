/// Re-export Flame DType
pub use flame_core::DType;

/// Helper trait for dtype operations
pub trait DTypeExt {
    /// Check if dtype is BF16
    fn is_bf16(&self) -> bool;

    /// Check if dtype is FP32
    fn is_f32(&self) -> bool;

    /// Check if dtype is FP16
    fn is_f16(&self) -> bool;
}

impl DTypeExt for DType {
    fn is_bf16(&self) -> bool {
        matches!(self, DType::BF16)
    }

    fn is_f32(&self) -> bool {
        matches!(self, DType::F32)
    }

    fn is_f16(&self) -> bool {
        matches!(self, DType::F16)
    }
}
