# baselines/

Canonical "known-good run" numbers per trainer. `/train-launch` reads
`<model>.toml` in preflight as the source of truth — instead of grepping
memory + handoffs.

## Schema

See any existing `<model>.toml` for the layout. Required keys:

- `model`, `config_canonical`, `last_validated`, `flame_core_commit`
- `[step_1] loss`
- `[steady_state] loss_mean_last_30`, `step_seconds`
- `[grad_flow] fires_at_step_1_or_later` (any nonzero = regression)
- `[lora_health] b_nonzero_ratio` + `at_step`
- `[known_quirks] notes` — list of one-liners (FLAME_ALLOC_POOL, timestep
  shifts, etc.)

Optional `[variants.<name>]` subtables for long-run / production numbers
that differ from smoke baselines.

## Update procedure

After a clean training run completes:

```
/baseline-set
```

The skill verifies the run was actually clean (no NaN, no grad-flow fires
at step ≥1, no dead LoRA-B), proposes a diff against the existing baseline
if any, asks for confirmation, writes the file.

**Never edit these by hand without bumping `last_validated` + the relevant
commit fields.** A baseline that doesn't say when it was valid is unfalsifiable.

## What's NOT a valid baseline

- A run that crashed (even if it ran 1000 steps before crashing)
- A LyCORIS run used to baseline a plain-LoRA model (+2.1 s/step bias)
- A run where `RUST_LOG=info` or `FLAME_ASSERT_GRAD_FLOW=1` wasn't set
- Any run prior to 2026-05-05 if Klein / ERNIE / Flux / SDXL / Z-Image — the
  HWC/CHW prepare bug invalidated those caches
