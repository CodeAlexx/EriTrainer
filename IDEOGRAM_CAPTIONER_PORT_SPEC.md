# Ideogram Captioner — Port Spec (Rust + Mojo)

Review-only spec. No code was ported or modified. Source: `/home/alex/ai-toolkit`.
Ground truth captured below was produced by running the real ai-toolkit code with
no GPU and no LLM (`/home/alex/serenityflow-v2/.venv/bin/python`).

---

## 0. TL;DR — the decisive finding

There is **no single "ideogram captioner" that turns a `.txt` into a `.json`.** That
framing conflates two separate things in ai-toolkit:

| Piece | What it does | Backend | Portable to Rust/Mojo? |
|---|---|---|---|
| **A. `toolkit/ideogram_caption.py`** | normalize / migrate / minify an *already-structured* caption (dict or JSON string) into the compact model-string; pass prose through unchanged | **deterministic** (json + re + string ops) | **YES — clean 1:1 port. This is the real target.** |
| **B. `extensions_built_in/captioner/Ideogram4Captioner.py`** | generate the structured caption **from an image** | **LLM/VLM** (Qwen3-VL-8B) | NO — it IS the VLM call; only the I/O + (A) glue is portable |
| **C. `ui_scripts/upsample_ideogram4_caption.py`** | generate the structured caption **from a short idea/prompt** | **LLM** (Qwen3-VL-8B, text-only) | NO — same as B |

**The `.txt` files are plain English prose. The `.json` files are the structured
Ideogram schema.** Nothing in ai-toolkit deterministically turns prose → structure;
that step is exclusively the VLM/LLM (B or C). The deterministic code (A) only
*normalizes already-structured* captions and *passes prose through untouched*.

**Recommended port target = module A (`toolkit/ideogram_caption.py`).** A working,
proven consumer of exactly this already exists in this repo:
`EriTrainer/trainer/parity/automagic3/minify_captions.py` — it walks a dir and
rewrites each structured `.json` to its minified model-string via `digest_caption_string`.
Porting A to Rust/Mojo makes that minify step native (no Python/ai-toolkit dependency).

If the user literally wants "prose `.txt` → structured `.json`", that is **B/C** and is
a **glue-only port around an LLM** — Rust/Mojo would supply the file I/O + prompt
templating + JSON extraction + module-A normalization, and call out to a separately
hosted Qwen3-VL (e.g. the existing `serenitymojo` magic-prompt llama-server path, or
the ported Ideogram4 Qwen3-VL encoder). The model itself does not "port" as a formatter.

---

## 1. Files located (every candidate)

```
toolkit/ideogram_caption.py                                  <- THE deterministic core (PORT THIS)
extensions_built_in/captioner/Ideogram4Captioner.py          <- VLM captioner (image -> JSON)
extensions_built_in/captioner/Qwen3VLCaptioner.py            <- parent: loads Qwen3-VL, generate()
extensions_built_in/captioner/BaseCaptioner.py               <- file walk, save, SQLite/UI plumbing
extensions_built_in/captioner/prompts/ideogram4_caption_prompt.py    <- 261-line VLM system prompt
extensions_built_in/captioner/prompts/ideogram4_upsample_prompt.py   <- 100-line LLM system prompt
ui_scripts/upsample_ideogram4_caption.py                     <- LLM upsampler (idea -> JSON)
extensions_built_in/diffusion_models/ideogram4/ideogram4.py  <- CONSUMER (line 522: digest at encode)
```

Why A is "the captioner formatter": `ideogram_caption.py`'s own docstring calls itself
"the single source of truth for the caption schema so the captioner, the prompt
upsampler, the dataloader, and the model encoder all agree." B and C both import from
A and end by calling `normalize_caption_dict`. The training encoder (`ideogram4.py:522`)
calls `digest_caption_string(p)` on every prompt at encode time.

---

## 2. Module A behavior, exactly (the portable surface)

### Public functions (all in `toolkit/ideogram_caption.py`)
- `canon_medium(medium)` — lower/strip, drop trailing `.`, map aliases → one of
  `photograph|illustration|3d_render|painting|graphic_design`; unknown mediums kept verbatim (stripped).
- `is_photo_medium(medium)` — `canon_medium(...) == "photograph"`.
- `normalize_hex(color)` — `#RGB`→`#RRGGBB`, uppercase; invalid → `None`.
- `sanitize_palette(palette, max_len)` — unique valid uppercase hex, in order, capped to
  `max_len`; empty → `None` (caller drops the key). Image cap = 16, element cap = 5.
- `normalize_style(style)` — pick **photo vs art_style branch** (recognized medium is
  authoritative, else infer from which key exists, default photo) and emit **strict key order**:
  - photo branch: `aesthetics, lighting, photo, medium, color_palette`
  - art branch:   `aesthetics, lighting, medium, art_style, color_palette`
  - migrates old shape (always `photo`) → correct branch; unknown extra keys appended at end.
- `normalize_element(el)` — strict key order:
  - `obj`:  `type, bbox, desc, color_palette`
  - `text`: `type, bbox, text, desc, color_palette`
  - bbox kept verbatim (already `[y1,x1,y2,x2]` stored form); palette capped to 5; extras appended.
- `normalize_caption_dict(data)` — drop `aspect_ratio`; top-level key order
  `high_level_description, style_description, compositional_deconstruction`; inside decon:
  `background` then `elements` (each via `normalize_element`); extras appended. Returns `OrderedDict`.
- `to_model_string(data)` — `json.dumps(data, ensure_ascii=False, separators=(",",":"))`
  (compact, **no spaces**, **no `\uXXXX`**).
- `digest_caption_string(text)` — **the top-level entry**: if text doesn't start with `{`
  → return unchanged (prose passes through). Else parse with `object_pairs_hook=OrderedDict`;
  if not a dict with a `compositional_deconstruction` object → return unchanged; else
  `to_model_string(normalize_caption_dict(parsed))`.
- `is_ideogram_caption_str(text)` — `True` iff parses to a dict with a
  `compositional_deconstruction` dict.
- `swap_bbox_xy_in_text(text)` — **regex** rewrite of every `"bbox":[a,b,c,d]` in *raw,
  possibly-malformed* text: swap `[x1,y1,x2,y2]`→`[y1,x1,y2,x2]`, clamp each to 0–1000,
  sort each axis pair. Touches only bbox arrays; everything else byte-for-byte.

### The schema (output `.json`)
Top level (key order strict): `high_level_description?`, `style_description?`,
`compositional_deconstruction` (required).
- `style_description` = EXACTLY ONE of `photo` / `art_style` (never both). Branch + key
  order per `normalize_style` above. `medium` ∈ the 5 tokens. `color_palette`: uppercase
  `#RRGGBB`, ≤16.
- `compositional_deconstruction` = `background` (string) + `elements` (list). Each element
  per `normalize_element`. `bbox` optional, normalized 0–1000, `[y_min,x_min,y_max,x_max]`,
  origin top-left. element `color_palette` ≤5.
- Serialize compact: `separators=(",",":")`, `ensure_ascii=False`.

---

## 3. Ground truth (parity oracle — produced from ai-toolkit's own code, no GPU)

### Oracle 1 — prose `.txt` passes through unchanged (the input most `.txt` files are)
Input = `datasets/eri2_with_trigger/10.txt` (plain prose, starts `vrtlEri2, The image depicts...`).
`digest_caption_string(input) == input` → **True**. (Prose is NOT structured; it is never
turned into JSON by deterministic code.)

### Oracle 2 — old-format structured JSON → migrated compact model-string
Input (old shape: `photo` key + `medium:"Illustration."` + `#f00` + element palette-first + `aspect_ratio`):
```json
{"aspect_ratio":"1:1","high_level_description":"A red apple on a table.",
 "style_description":{"aesthetics":"clean, minimal","lighting":"soft daylight",
   "photo":"studio product shot","medium":"Illustration.","color_palette":["#f00","#FFFFFF","#f00"]},
 "compositional_deconstruction":{"background":"plain white","elements":[
   {"type":"obj","color_palette":["#ff0000"],"desc":"a red apple","bbox":[100,200,300,400]},
   {"type":"text","color_palette":["#000"],"text":"FRESH","desc":"label","bbox":[10,20,30,40]}]}}
```
Exact output (`digest_caption_string`):
```
{"high_level_description":"A red apple on a table.","style_description":{"aesthetics":"clean, minimal","lighting":"soft daylight","medium":"illustration","art_style":"studio product shot","color_palette":["#FF0000","#FFFFFF"]},"compositional_deconstruction":{"background":"plain white","elements":[{"type":"obj","bbox":[100,200,300,400],"desc":"a red apple","color_palette":["#FF0000"]},{"type":"text","bbox":[10,20,30,40],"text":"FRESH","desc":"label","color_palette":["#000000"]}]}}
```
Proves, in one fixture: `aspect_ratio` dropped · `photo`+illustration-medium → **art_style branch** ·
medium canonicalized `Illustration.`→`illustration` · palette `#f00`→`#FF0000`, dedup, uppercase ·
element keys reordered to `type,bbox,desc,color_palette` · compact separators · branch key order.

### Oracle 3 — unit helpers
`canon_medium("Illustration.")="illustration"`; `canon_medium("3D Render")="3d_render"`;
`normalize_hex("#f00")="#FF0000"`; `is_ideogram_caption_str(prose)=False`,
`is_ideogram_caption_str(structured)=True`.

### Oracle 4 — `swap_bbox_xy_in_text` on malformed raw
Input: `garbage {"bbox":[200,100,400,300]} more {"bbox": [5, 6, 1, 2]} tail`
Output: `garbage {"bbox":[100,200,300,400]} more {"bbox":[2,1,6,5]} tail`
(swap+clamp+sort each axis; non-bbox text untouched.)

---

## 4. Dependencies

**Module A (the port target):** stdlib only — `json`, `re`, `collections.OrderedDict`.
No torch, no model, no network. Trivial.

**Module B/C (LLM-backed, NOT the formatter):** `torch`, `transformers`
(`Qwen3VLForConditionalGeneration` / `...MoeForConditionalGeneration`, `AutoProcessor`),
`PIL`, optional `optimum.quanto` (fp8), the Qwen3-VL-8B weights, plus the 261/100-line
prompt templates. `BaseCaptioner` additionally pulls `sqlite3` + asyncio UI plumbing
(irrelevant to a port — that's the ai-toolkit web UI, not the caption logic).

---

## 5. Risk / effort

**Module A — clean 1:1 port. Low risk, byte-exact gate is feasible.**

Hardest parts:
- **Rust:** (1) **Key order is load-bearing** and the output must be byte-identical —
  use `serde_json` with `preserve_order` (IndexMap) on both read and write, build output
  maps in the exact documented order, and reproduce Python's compact `separators=(",",":")`
  with `ensure_ascii=false` (no `\uXXXX`, no spaces) — a custom/compact serializer or a
  `serde_json` formatter, verified against the oracle bytes. (2) **The "pass prose through
  byte-for-byte" rule** — `digest_caption_string` must return the *original string
  unchanged* (not a reparse) for prose and for non-caption JSON; don't round-trip.
- **Mojo:** (1) **No mature JSON-with-preserved-order lib** — use the MOJO-libs json lib
  (see `[[reference-mojo-libs-repo]]`) and confirm it preserves insertion order, or
  carry ordered key lists explicitly. (2) **Compact serializer** matching Python's
  `separators=(",",":")` + non-ASCII passthrough + the exact float/int formatting for
  bbox ints. (3) `swap_bbox_xy_in_text` is **regex** over arbitrary text — Mojo's regex
  story is weak; likely a hand-written scanner for `"bbox":[n,n,n,n]` (ints/floats, signs,
  whitespace) per `_BBOX_TEXT_RE`. Same applies to Rust but `regex` crate makes it trivial there.

Shared gotchas to replicate exactly: dedupe palette by *normalized* hex (first occurrence
wins, order preserved); empty palette ⇒ **drop the key** (not `[]`); element bbox sort uses
the *stored* `[y1,x1,y2,x2]` order; medium alias table verbatim; unknown extra keys appended
at end in original order.

**Module B/C — glue-only port.** Porting "the captioner" in the image/idea→JSON sense is
**not a formatter port**; it's I/O + prompt-template substitution (`{{aspect_ratio}}`,
`{{user_instructions}}`, `{{mode_directive}}`, `{{original_prompt}}`) + tolerant JSON
extraction (strip ```json fences, take outermost `{...}`) + module-A normalization, wrapped
around a **separately hosted Qwen3-VL call**. serenitymojo already has the pieces for this
(magic-prompt llama-server: `[[project-ideogram4-magic-prompt]]`; ported Ideogram4 Qwen3-VL
encoder: `[[project-ideogram4-mojo-port]]`). Hardest parts are non-deterministic by nature
— a byte-exact gate is impossible; you can only gate the deterministic glue (template
substitution, `extract_json`, `sanitize_bbox`, normalization) and smoke-test the LLM call.

---

## 6. Is a byte-exact `.txt→.json` parity gate feasible?

- **For module A (recommended target): YES.** Oracles 1–4 above are exact. The natural gate:
  port `digest_caption_string` + helpers, run them on a fixture set, and assert
  **byte-identical** output to the four oracles (and to a sweep of the real
  `datasets/gigerver3_json/*.json` run through Python's `digest_caption_string`). The existing
  `EriTrainer/.../minify_captions.py` is the reference harness — reproduce its
  before/after bytes natively.
- **For module B/C (image/idea → JSON): NO byte-exact gate** (LLM is stochastic; even greedy
  decode is GPU/version-sensitive). A parity fixture there would be: freeze model + seed +
  greedy + a fixed image/idea, capture one reference JSON, and gate only that the *glue*
  (prompt build, JSON extraction, normalization) is byte-exact given a fixed raw model string.

---

## 7. Sample fixture produced

No new files written to the dataset. The fixtures are inline in §3 and are fully
reproducible with:
```
/home/alex/serenityflow-v2/.venv/bin/python - <<'PY'
import sys, json; sys.path.insert(0,'/home/alex/ai-toolkit')
from toolkit.ideogram_caption import digest_caption_string
# Oracle 2 input -> exact output (see §3)
PY
```
Real data available as oracles: `datasets/gigerver3_json/*.json` (already-structured, exercise
the migration/minify path) and `datasets/eri2_with_trigger/*.txt` (prose, exercise passthrough).

---

## PHASE 1 STATUS — COMPLETE (2026-06-24)

Module A (deterministic normalizer/minifier) ported to BOTH languages, BYTE-EXACT,
each via builder → skeptic → bugfix → orchestrator re-gate.

- Oracle: `EriTrainer/trainer/parity/ideogram_caption/fixtures.json` (126 cases from real Python; gen_fixtures.py).
- **Rust** (`EriTrainer/crates/ideogram-caption/`, standalone leaf crate, serde_json+regex, 2.6s build): 126/126 committed + 39/39 adversarial. Skeptic found 1 BLOCKER (Unicode bbox digits → Python `float()` accepts, Rust rejected) → fixed via Unicode-15 Nd digit table. FRAGILE huge-int documented (robust saturate, not Python's crash).
- **Mojo** (`serenitymojo/captioner/`, hand-rolled ordered-JSON + parser + compact serializer): 126/126 committed + 143/143 adversarial. Skeptic found 5 BLOCKERS (NUL truncation, dup keys, loose number parser, raw ctrl, float re-emit) — all fixed with non-vacuous per-fix proof; NaN/Inf documented. Both gates fail-loud.
- Both verified by orchestrator's own re-runs.

## PHASE 2 — PENDING (B/C LLM glue). The deterministic glue (prompt-template substitution,
extract_json, sanitize_bbox, module-A normalization) is byte-gateable; the LLM call (Qwen3-VL-8B)
is smoke-only. Reuses serenitymojo magic-prompt llama-server (Mojo) / a hosted LLM (Rust).

---

## PHASE 2 STATUS — COMPLETE (2026-06-25). FULL CAPTIONER PORT DONE (A+B+C, Rust + Mojo).

B+C deterministic glue ported both languages, byte-exact, builder→skeptic→bugfix→orchestrator-re-gate.
Fixtures: EriTrainer/trainer/parity/ideogram_bc_glue/fixtures.json (63) + per-language adversarial.

- **Rust** (crates/ideogram-caption/src/glue.rs): module A 126/126 + 39 adv; glue 63/63 + 70 adv. Skeptic found number-serialization divergence on extreme/non-schema numerics → FIXED via serde_json arbitrary_precision (non-vacuously verified); PEP-515 underscores + inf/nan-bbox documented as out-of-schema deviations. Serializer: B=to_string_pretty (==CPython indent=2, measured), C=custom PyDefaultFormatter (", "/": "). LLM = pluggable CaptionLlm trait (Stub + OpenAiCompat w/ injected transport — crate stays a fast leaf, no http dep).
- **Mojo** (serenitymojo/captioner/ideogram_bc_glue.mojo): module A 126/126 + 143 adv; glue 105/105 (incl. 42 adversarial). Hand-rolled 2-mode serializer (==CPython, incl. 1e2→100.0) + hand-rolled lazy-DOTALL fence scanner + _coerce_float (Python float() bbox coercion). Skeptic found bbox string/bool coercion gap → FIXED (non-vacuous). LLM = pluggable CaptionGenerator (Stub + HTTP-to-VL-server interface).
- bbox: B swaps x/y, C does not (both). LLM call is smoke-only (NOT gated), reuses Qwen3-VL-8B host (HTTP/llama-server).
- Both verified by orchestrator re-runs. All gates fail-loud. UNCOMMITTED (captioner/ untracked; Rust crate new) — commit on request.
