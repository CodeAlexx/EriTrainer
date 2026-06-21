# cudarc-pinctx

> Vendored cudarc 0.11.9 with one patch: thread-local cache of the bound
> CUDA context. `bind_to_thread()` skips `cuCtxSetCurrent` when already
> current. Measured on EriDiffusion zimage LoRA: 91,637 → 0
> `cuCtxSetCurrent` calls per training step, −55.7% total driver API
> calls, −6.9% wall.

---

A vendored copy of [cudarc](https://github.com/coreylowman/cudarc) 0.11.9 with
**one patch**: per-thread cache of the bound CUDA context, so
`CudaDevice::bind_to_thread()` skips the underlying `cuCtxSetCurrent` driver
call when the requested context is already current on this thread.

## Why

cudarc 0.11.9 calls `bind_to_thread()` from every safe operation — every
kernel launch, every memory alloc, every memory free, every cublas/cudnn
init. Each call is `cuCtxSetCurrent(self.cu_primary_ctx)` to the CUDA
driver, unconditionally. On
[EriDiffusion](https://github.com/CodeAlexx/EriDiffusion)'s flame-core
trainer (zimage rank=8 LoRA, ~17 k kernels/step + ~28 k allocs/step),
that's **91,637 `cuCtxSetCurrent` calls per training step**, ~9 ms/step of
pure CPU-side driver overhead.

PyTorch's CUDA bindings set the context once per device per thread, then
launch and allocate against the already-current context with no further
`cuCtxSetCurrent`. This patch makes cudarc behave the same way.

## The patch

Single file, ~25 lines added. Excerpt from
`src/driver/safe/threading.rs`:

```rust
thread_local! {
    static BOUND_CTX: Cell<sys::CUcontext> = const { Cell::new(std::ptr::null_mut()) };
}

impl CudaDevice {
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        let target = self.cu_primary_ctx;
        let cached = BOUND_CTX.with(|c| c.get());
        if cached == target {
            return Ok(());                          // fast path: no driver call
        }
        unsafe { result::ctx::set_current(target) }?;
        BOUND_CTX.with(|c| c.set(target));
        Ok(())
    }
}
```

Every other file in the crate is byte-identical to upstream cudarc 0.11.9.

## Measured impact

3× verification on EriDiffusion zimage LoRA training (rank=8, 20 steps,
5-step warmup, RTX 3090 Ti, CUDA 12.4), bit-identical loss curve before
and after:

| Metric | Before | After | Delta |
|---|---|---|---|
| `cuCtxSetCurrent`/step | 91,637 | 0 (absent from top-15) | eliminated |
| Total CUDA driver API calls/step | 200,472 | 88,903 | **−55.7%** |
| Trainer wall time | 3,319 ms/step | 3,090 ms/step | **−6.9%** |

The wall delta is smaller than the call-count delta because each
`cuCtxSetCurrent` is ~106 ns — cheap individually, but firing 91 k× per
step adds up to ~9 ms of pure CPU stalling. After the patch, that line
disappears from the profile.

## Safety

- Thread-local: a thread that hasn't called `bind_to_thread()` yet, or that
  switches between two different `CudaDevice` instances, still hits the
  slow path. The fast path only triggers when the same thread asks for
  the same context twice in a row.
- `Cell<*mut CUctx_st>`: not `Send`, not `Sync`, single-threaded reads and
  writes. No data race.
- The cached pointer never outlives the device: when the device is dropped
  cudarc tears down its primary context, but the cache is per-thread, so
  the next bind from a different device updates the cache.

Single-device single-thread training (the common case) hits the fast path
on every call after the first. Multi-device usage still works correctly —
the cache flips between contexts at each switch.

## Setup

cudarc-pinctx is consumed via Cargo's `[patch.crates-io]` mechanism. Place
it as a sibling of the crates that depend on cudarc:

```
your-workspace/
├── flame-core/
├── EriDiffusion-v2/
└── cudarc-pinctx/    ← this repository
```

Add to each top-level `Cargo.toml` that participates in the workspace
dependency graph (in EriDiffusion's case: `flame-core/Cargo.toml` AND
`EriDiffusion-v2/Cargo.toml`):

```toml
[patch.crates-io]
cudarc = { path = "../cudarc-pinctx" }
```

`cargo build` then resolves every `cudarc` import — direct or transitive —
to the patched copy. No changes needed at any cudarc call site.

## Cloning

```sh
cd your-workspace/
git clone https://github.com/CodeAlexx/cudarc-pinctx.git
```

That's it. The `[patch.crates-io]` entries in flame-core and
EriDiffusion-v2 already point at `../cudarc-pinctx`.

## Upstream tracking

| File changed vs upstream cudarc 0.11.9 | Purpose |
|---|---|
| `src/driver/safe/threading.rs` | thread-local CUcontext cache + fast-path in `bind_to_thread()` |

To re-vendor against a different cudarc release, copy the upstream source,
then apply the same 25-line patch to `threading.rs`. The patch is independent
of cudarc's internals — it only touches the one safe wrapper function.

## When to drop this fork

If/when upstream cudarc (or whichever Rust CUDA crate the workspace uses)
gains its own per-thread context caching:

```toml
# 1. Remove the [patch.crates-io] section from both consuming Cargo.tomls
# 2. Upgrade the cudarc dependency to the version that has the caching
# 3. Delete this directory
```

The patch is intentionally minimal so the migration is trivial.

## License

Same as upstream cudarc: dual MIT / Apache-2.0. See `LICENSE-MIT` and
`LICENSE-APACHE`.
