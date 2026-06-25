# SKEPTIC FINDINGS — Rust port of the Ideogram B/C captioner GLUE

Target: `/home/alex/EriTrainer/crates/ideogram-caption/src/glue.rs` (+ `templates/*.txt`).
Reference: ai-toolkit `extensions_built_in/captioner/Ideogram4Captioner.py` (B),
`ui_scripts/upsample_ideogram4_caption.py` (C), `toolkit/ideogram_caption.py` (module A).
Oracle env: `/home/alex/serenityflow-v2/.venv/bin/python` (CPython 3.12.3), no GPU/LLM.

Method: built a Python-vs-Rust differential harness over the **REAL ai-toolkit
deterministic code** (module A imported directly; B/C deterministic functions copied
verbatim and guarded against drift by asserting the snippets still exist in source).
Ran ~290 adversarial inputs the committed 63-fixture gate never reaches, byte-comparing
CPython output against the Rust glue. Harness lived in `tests/zz_skeptic_diff.rs`
(temporary, removed after the pass; reproduction recipe at the bottom).

## VERDICT: 2 BLOCKERS (conditional) / 3 FRAGILE / 2 STYLE — NOT clean.

The **core deterministic glue is faithful** on every in-schema input: serializer
(empty `{}`/`[]` inlining, control-char escapes, solidus, DEL, unicode-in-pretty —
all byte-identical across 23 structures × 2 modes), `extract_json` (multi-fence
first-wins, CRLF/CR, BOM, nested fences, braces-in-strings, two-object→None — 70+
cases), bbox clamp/sort/**banker's-round**/swap, duplicate-key last-wins-first-pos,
build_prompt substitution **order** (incl. the `{{aspect_ratio}}`-in-instructions and
`{{original_prompt}}` ordering traps), `compute_aspect_ratio` (31 cases incl.
tie-breaks/degenerate/large), `normalize_item`, art-style migration, palette
dedup/uppercase/#abc-expand, non-ASCII raw preservation, and the swap-in-text
fallback. Templates + both directive constants are **char-for-char** byte-equal to
what Python builds at runtime (B import-resolved 22831 chars sha `9e9ce2a3…`; C
verbatim slice 7892 chars sha `7bf215b2…`; FAITHFUL 340 sha `1217ecda…`; CREATIVE
579 sha `29bd3f2b…`).

The divergences are all in the **out-of-schema / garbage-model-output** corner — but
they are real, they reach the public glue functions, and the spec's blanket
"byte-exact glue gate: YES" claim never tested them.

---

## F1 — BLOCKER (conditional): serializer diverges on big-ints & extreme floats in passthrough positions

**The spec's #1 claim** ("`serde_json::to_string_pretty` / `PyDefaultFormatter` is
byte-identical to CPython `json.dumps`") is TRUE for every structural case (empty
containers, escapes, separators, indent, unicode) — but FALSE for numeric tokens that
CPython and serde_json format differently. These tokens reach the serializer because
`normalize_caption_dict` passes **any non-bbox value verbatim** (background,
high_level_description, unknown extra keys, element extras), so a garbage/adversarial
model emitting a number there survives to `to_default_string`/`to_pretty_string`.

Proven end-to-end through `b_full_glue` AND `c_full_glue`:

| input (raw model string) | CPython bytes | Rust bytes |
|---|---|---|
| `{"compositional_deconstruction":{"background":"x"},"weird":9999999999999999999999}` (C) | `…"weird": 9999999999999999999999}` | `…"weird": 1e+22}` |
| `{"compositional_deconstruction":{"background":1e-30,"elements":[]}}` (C) | `…"background": 1e-30, …` | `…"background": 9.999999999999999e-31, …` |
| `{"high_level_description":18446744073709551616,…}` (C) | `…: 18446744073709551616, …` | `…: 1.8446744073709552e+19, …` |
| same shapes, B `indent=2` | (multi-line, exact int/`1e-30`) | (multi-line, `1e+22` / `9.99…e-31`) |

Root cause: `serde_json::Value` stores any JSON number outside i64/u64 range as `f64`,
then formats it with Rust's float formatter (shortest round-trip, `e+NN` style, no
preservation of the source token). CPython preserves the **literal source digits** for
ints of arbitrary size and uses `repr(float)` (`1e-30`, not `9.999…e-31`).
`1e30` happens to agree (`1e+30` == `1e+30`); `1e-30`, `>u64` ints, and `>~17-digit`
ints do not.

Severity: BLOCKER **iff** the gate's contract is "byte-exact on arbitrary model
output." In practice every in-schema passthrough value is a string, so a well-behaved
model never triggers it → drop to FRAGILE if the contract is "byte-exact on schema-
valid captions only." Either way the code comment at `glue.rs:16-20` ("which
`serde_json::to_string_pretty` reproduces byte-for-byte … verified") **overclaims** —
it is only verified for non-numeric content.

Fix options: (a) enable serde_json `arbitrary_precision` so big-ints keep their source
token (handles the int cases; float `1e-30` still needs a CPython-`repr`-faithful float
writer); (b) a custom number formatter matching CPython `float.__repr__` + arbitrary-
int passthrough; (c) explicitly scope the contract to "string-valued schema fields"
and assert no non-bbox numbers can appear (they can — extra keys are preserved).

---

## F2 — FRAGILE: `float()` permissiveness — Python accepts `_` digit-grouping, Rust rejects

In the bbox path `value_to_f64` does `s.trim().parse::<f64>()`, but Python does
`float(v)`. Python's `float()` accepts **underscore digit separators** (PEP 515:
`float("1_000") == 1000.0`, `"1_0"`, `"1_000_000"`), which `f64::from_str` rejects.

| input | CPython `b_convert_bbox` | Rust `b_convert_bbox` |
|---|---|---|
| `["1_000","20","30","40"]` | `[20, 30, 40, 1000]` | `None` (whole bbox dropped) |
| `["1_0","2_0","3_0","4_0"]` | `[20, 10, 40, 30]` | `None` |
| `["1_000_000","2","3","4"]` | `[2, 3, 4, 1000]` | `None` |

Same for `c_sanitize_bbox` (different output order, same drop). Effect: a model that
writes grouped digits keeps its (clamped) box in Python but loses it entirely in Rust,
changing the element's normalized output. Low real-world exposure (models emit plain
digits), but it is a genuine `float()`-vs-`parse` gap the code comment at
`glue.rs:163-166` claims to have handled ("Python float() trims surrounding whitespace
and accepts Unicode digits") — it covers whitespace + Unicode digits but **misses
underscores**. (Note: `+5`, `1.`, `.5`, `1e2`, fullwidth `１０` all DO match — Rust f64
and `normalize_unicode_digits` cover those.)

---

## F3 — FRAGILE: `inf`/`nan` bbox strings — Python RAISES (aborts), Rust silently fabricates a box

The bbox inner `try/except` in B/C wraps ONLY `[float(v) for v in bbox]`, NOT the
following `round(...)`. So `float("inf")` succeeds, then `round(inf)` raises
**OverflowError** and `round(nan)` raises **ValueError**, escaping `_convert_bbox`/
`sanitize_bbox`:

- **B**: the exception bubbles to `get_caption_for_file`'s outer `try/except` → the
  whole file returns `None` (no caption saved).
- **C**: `upsample_one` has **no** try/except around `sanitize_caption` → the script
  **crashes** (uncaught OverflowError/ValueError).

Rust's `value_to_f64` parses `"inf"`/`"nan"` (Rust `f64::from_str` accepts them), then
`round_half_even(inf)` saturates → a fabricated box, and the glue continues:

| input bbox | CPython | Rust |
|---|---|---|
| `["inf",2,3,4]` | OverflowError → B:`None` / C:crash | `b_full_glue` → box `[2,3,4,1000]`, JSON emitted |
| `["nan",2,3,4]` | ValueError → B:`None` / C:crash | `b_full_glue` → box `[2,0,4,3]`, JSON emitted |
| `["1e400",…]` (→`inf` via float) | OverflowError | `[…,1000]` emitted |

This is the **same class** as module A's documented deliberate inf-deviation
(`lib.rs:535-557`), but: (1) it lives in a **separate code path** (the glue's
`clamp_1000_f`, not module-A `clamp_1000`) and is **undocumented** in `glue.rs`;
(2) the divergence here is observable on **normal JSON** (a string `"inf"` is valid
JSON inside a bbox), whereas module-A's text-regex path can never capture `inf`;
(3) the Python outcome is "abort the whole caption" (B→None, C→crash), not "clamp,"
so Rust producing a valid-looking box is a behavior change, not just a value change.
Defensible as robustness-over-crash, but it must be **stated** as a deviation; the
glue currently claims clean 1:1.

---

## F4 — STYLE/latent: `\s` and `.strip()` Unicode-class mismatch (currently masked)

Two real class mismatches that the pipeline happens to neutralize:

1. **fence regex `\s*`**: Python `re` `\s` matches `0x1c-0x1f` (FS/GS/RS/US) and
   `0x85`/`0xa0`/unicode spaces; Rust `regex` `\s` = `\p{White_Space}` does NOT include
   `0x1c-0x1f`. So `` ```json␜{...}``` `` captures differently between the two.
2. **`raw.trim()` vs Python `raw.strip()`**: Python `.strip()` strips `0x1c-0x1f`;
   Rust `.trim()` (White_Space) does not.

Tested 110+ inputs placing `0x1c-0x1f`/`0x0b`/`0x0c`/`0x85`/`0xa0` after the fence tag,
around the raw string, and as fence-only content. **All matched** — because the regex
capture is always `.strip()`-ed AND then reduced to the first-`{`..last-`}` span, so a
boundary control char never lands inside the final brace span. No observable divergence
found, but the underlying char classes are NOT equal; a future refactor that removes
the `.strip()` or the brace-span (or a `\s` that must match content, not just trim)
would expose it. The comment at `glue.rs:110-112` ("DOTALL … lazy") is right about the
DOTALL/lazy semantics but silent on the `\s`-class difference.

---

## F5 — STYLE/pathological: `normalize_item` non-string truthy `aspect_ratio`

Python `normalize_item({"prompt":"x","aspect_ratio":True}, …)` returns `("x", True)`
(the truthy non-string AR carried through unchanged, because `item.get("aspect_ratio")
or default` keeps a truthy value as-is). Rust returns `("x", "1:1")` (falls back to
default for any non-string AR). Already acknowledged in the code (`glue.rs:413-418`)
and AR is only ever consumed as a string downstream, so harmless — but it is a real
return-value difference, not byte-exact. Pathological input only.

---

## What was tested and is CLEAN (so the report is fair)

- Serializer default + `indent=2`: 23 structures (empty/nested-empty containers,
  arrays-of-objects, single-element arrays, control chars ` -`, solidus
  `/` un-escaped by both, DEL ``, empty-string key, unicode in pretty) — 46/46.
- Numeric serializer where it **agrees**: `1e30→1e+30`, `0.1`, `2.0`, `-0.0`, `1.5e-10`,
  i64-range ints, `0.3333333333333333`, `2.718281828459045` — all byte-equal.
- `extract_json`: 70+ cases — multi-fence (first wins), unclosed fence→brace-span,
  ` ``` ` vs ` ```json `, CRLF, CR-only, NBSP/line-sep/tab after tag, two-object span
  →None, nested braces in strings, braces-in-keys, BOM prefix, empty-fence-then-json,
  whitespace-only→None, prose-wrapped.
- Duplicate keys: last-value-at-first-position — matches CPython (serde preserve_order).
- bbox: banker's rounding battery (`.5` even/odd, `[10.6,20.4,300.5,400.5]`), clamp at
  0/1000 boundary, round-then-sort, degenerate `y2<=y1`/`x2<=x1`→None, wrong-len,
  non-list, bool→1.0/0.0, null/list/dict element→drop, string coercion `+5`/`1.`/`.5`/
  `1e2`/`  10.5  `/fullwidth `１０`.
- `compute_aspect_ratio`: 31 cases (degenerate `0`/negative→`1:1`, `1023×768→4:3`,
  `1920×1080→16:9`, Fibonacci-ish, snap-path tie-breaks, large `1000000×999999`).
- `normalize_item`: str / dict / dict-no-AR / empty-AR→default / null-AR / `0`-AR /
  empty-after-strip→None / non-str / non-dict / `{noprompt}` / non-str-prompt /
  idea-not-stripped.
- build_prompt B+C: 18 cases incl. None/empty/spaced instructions, and the ordering
  traps (`{{aspect_ratio}}`/`{{original_prompt}}`/`{{mode_directive}}` appearing inside
  substituted values, where C's later substitutions DO/DON'T re-hit them) — all exact.
- Full-glue realistic: art_style-branch migration, palette dedup+uppercase+`#abc`
  expand+invalid-drop, non-ASCII (CJK/Cyrillic/accents) raw in BOTH pretty & default,
  bad-bbox-drop-element-kept, old-format migration, malformed→`swap_bbox_xy_in_text`
  fallback with non-ASCII + bbox swap.
- Templates: B/C `.txt` and both directive consts byte-equal to Python runtime values.

---

## Reproduction

Oracle scripts in `/tmp/claude-1000/.../scratchpad/glue_diff/` (`oracle.py` =
verbatim B/C functions + real module A; `gen_serializer.py`, `*_oracle.json`
generators). Re-create the temporary Rust harness `tests/zz_skeptic_diff.rs` with
env-gated tests (`SER_ORACLE`/`FG_ORACLE`/`BP_ORACLE`) reading those JSONs and calling
`glue::*`, then e.g.:

```
# big-int / extreme-float serializer divergence (F1)
python gen_serializer.py > ser.json   # then num_oracle as in transcript
SER_ORACLE=num_oracle.json cargo test --offline --test zz_skeptic_diff skeptic_serializer_diff -- --nocapture
# underscore + inf/nan bbox (F2/F3)
FG_ORACLE=fg_num_oracle.json cargo test --offline --test zz_skeptic_diff skeptic_fullglue_diff -- --nocapture
```

The committed gate `tests/glue_byte_exact.rs` (63 cases) stays GREEN throughout —
confirming these divergences are real blind spots in the fixture set, not regressions.
