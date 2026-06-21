use super::{CudaDevice, DriverError};

use crate::driver::{result, sys};
use std::cell::Cell;

// 2026-05-12 (EriDiffusion launch-storm Phase 4): cache the per-thread
// bound context. cudarc 0.11.9 calls `bind_to_thread()` from every safe
// launch and alloc operation (~80k times per zimage train step). Each
// call is ~106ns of `cuCtxSetCurrent`, totalling ~9ms/step of pure
// driver-call overhead. PyTorch's CUDA bindings set the context once
// per device per thread; this patch matches that pattern by short-
// circuiting bind_to_thread when the requested context is already the
// current one for this thread.
//
// Safety: cuCtxSetCurrent's semantics are "make this context current
// on the calling thread, no-op if already current at the driver level".
// The cached value is a thread-local — single-device single-thread
// training (the common case) benefits the most. Multi-device usage
// still works: the cache flips between contexts at each bind boundary.
thread_local! {
    static BOUND_CTX: Cell<sys::CUcontext> = const { Cell::new(std::ptr::null_mut()) };
}

impl CudaDevice {
    /// Binds the device to the calling thread. You must call this before
    /// using the device on a separate thread!
    ///
    /// Fast-pathed via a thread-local: when this thread already has the
    /// same primary context current, the underlying `cuCtxSetCurrent`
    /// is skipped. See the file header for the rationale.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        let target = self.cu_primary_ctx;
        let cached = BOUND_CTX.with(|c| c.get());
        if cached == target {
            return Ok(());
        }
        unsafe { result::ctx::set_current(target) }?;
        BOUND_CTX.with(|c| c.set(target));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::thread;

    #[test]
    fn test_threading() {
        let dev1 = CudaDevice::new(0).unwrap();
        let dev2 = dev1.clone();

        let thread1 = thread::spawn(move || {
            dev1.bind_to_thread()?;
            dev1.alloc_zeros::<f32>(10)
        });
        let thread2 = thread::spawn(move || {
            dev2.bind_to_thread()?;
            dev2.alloc_zeros::<f32>(10)
        });

        let _: crate::driver::CudaSlice<f32> = thread1.join().unwrap().unwrap();
        let _: crate::driver::CudaSlice<f32> = thread2.join().unwrap().unwrap();
    }
}
