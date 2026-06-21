use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Flame error: {0}")]
    Flame(#[from] flame_core::Error),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: flame_core::Shape,
        got: flame_core::Shape,
    },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: String, got: String },

    #[error("Layout mismatch: expected {expected:?}, got {got:?}")]
    LayoutMismatch { expected: String, got: String },

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Kernel compilation error: {0}")]
    KernelCompilation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
