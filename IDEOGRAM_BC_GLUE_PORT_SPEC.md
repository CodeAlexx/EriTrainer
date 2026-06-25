# Ideogram B/C Captioner GLUE — Port Spec (Rust + Mojo)

Review-only spec. No code was ported or modified. Source: `/home/alex/ai-toolkit`.
Ground truth below was produced by running the REAL deterministic glue functions
(copied verbatim from source) plus the REAL module-A `toolkit/ideogram_caption.py`,
with **no GPU and no LLM** (`/home/alex/serenityflow-v2/.venv/bin/python`).
Sample fixtures live at
`/tmp/.../scratchpad/glue_fixtures.json` (regenerate via the generator in §6).

This spec covers the **glue around module A** (Phase-1 module A — the
normalizer/minifier — is already ported byte-exact in both languages:
`crates/ideogram-caption/src/lib.rs` and `serenitymojo/captioner/ideogram_caption.mojo`).
It does NOT re-spec module A; see `IDEOGRAM_CAPTIONER_PORT_SPEC.md` for that.

---

## 0. TL;DR

There are **two glue pipelines**, B (image→JSON) and C (idea/text→JSON). Each is:

```
build_prompt (template substitution)  →  [LLM.generate]  →  extract_json  →
  per-element bbox fix (convert/sanitize)  →  module-A normalize_caption_dict  →  json.dumps
```

- The **deterministic glue** = everything EXCEPT the `[LLM.generate]` box. It is
  **fully byte-gateable with no model** and is a clean 1:1 port to Rust + Mojo.
- The **LLM call** = `Qwen/Qwen3-VL-8B-Instruct` (a vision-language model). B feeds it
  an image + prompt; C feeds it text only. This is smoke-only, NOT byte-gated.
- B and C share `extract_json` (modulo one trivial difference), the
  `model.generate`/trim/decode wiring, and the module-A `normalize_caption_dict` tail.
  They differ in: input type (image vs text), which system prompt, the bbox transform
  (B **swaps x/y**; C does **not**), the malformed-JSON fallback (B has one; C returns
  `None`), and the OUTPUT serialization (B pretty `indent=2`; C compact-default `indent=None`).

**Feasibility of a byte-exact glue gate: YES.** Demonstrated below with real fixtures.

---

## 1. Files

```
extensions_built_in/captioner/Ideogram4Captioner.py            B glue (image -> JSON)   184 lines
extensions_built_in/captioner/Qwen3VLCaptioner.py              B parent: model load + generate wiring
extensions_built_in/captioner/BaseCaptioner.py                 file walk / save / SQLite-UI plumbing (mostly NOT ported)
extensions_built_in/captioner/prompts/ideogram4_caption_prompt.py   B system prompt (importable str, 22 831 chars after import)
extensions_built_in/captioner/prompts/ideogram4_upsample_prompt.py  C system prompt (verbatim-read str, 7 892 chars between """ """ )
ui_scripts/upsample_ideogram4_caption.py                       C glue (idea -> JSON)    392 lines
toolkit/ideogram_caption.py                                    module A (already ported) — the normalize the glue ends with
```

---

## 2. The deterministic glue — function by function (BYTE-GATEABLE)

### 2.1 Prompt-template substitution

**HOW it substitutes — plain `str.replace`, sequential, in this exact order. NOT
`.format`, NOT f-string.** Each `{{placeholder}}` is replaced literally. Order matters
only if a substituted value could itself contain a later placeholder — none of these do,
but match the order anyway for faithfulness.

**B — `Ideogram4Captioner.build_prompt(self, aspect_ratio)` (lines 61–69):**
```python
user_instructions = (self.caption_config.caption_prompt or "").strip()
if not user_instructions:
    user_instructions = "None."
prompt = ideogram4_caption_prompt.replace("{{aspect_ratio}}", aspect_ratio)
prompt = prompt.replace("{{user_instructions}}", user_instructions)
```
- Placeholders: **`{{aspect_ratio}}`, `{{user_instructions}}`** (exactly two; no
  `{{mode_directive}}`, no `{{original_prompt}}`).
- `user_instructions` = `config.caption_prompt`, `.strip()`-ed; empty/None → the literal
  `"None."`.
- Template source: imported as a Python str (`from .prompts.ideogram4_caption_prompt
  import ideogram4_caption_prompt`). The file is a normal `name = """…"""` and IS
  import-safe (no invalid escapes), so the imported value already has Python escapes
  resolved (`\\n` in the file → `\n` in the string, `\\uNNNN` → literal backslash-u, etc.).
  A port must embed the **post-import** string, not the raw file bytes.

**C — `upsample_ideogram4_caption.build_prompt(template, aspect_ratio, original_prompt,
creative, instructions)` (lines 83–95):**
```python
directive = CREATIVE_DIRECTIVE if creative else FAITHFUL_DIRECTIVE
prompt = template.replace("{{mode_directive}}", directive)
prompt = prompt.replace("{{user_instructions}}", instructions.strip() or "None.")
prompt = prompt.replace("{{aspect_ratio}}", aspect_ratio)
prompt = prompt.replace("{{original_prompt}}", original_prompt)
```
- Placeholders: **`{{mode_directive}}`, `{{user_instructions}}`, `{{aspect_ratio}}`,
  `{{original_prompt}}`** (exactly four).
- `{{mode_directive}}` ← one of two hard-coded constants
  (`FAITHFUL_DIRECTIVE` / `CREATIVE_DIRECTIVE`, lines 40–56 of the script — embed both
  verbatim in the port).
- `{{user_instructions}}` ← `instructions.strip()`, or the literal `"None."` if empty.
- `{{original_prompt}}` ← the raw idea string (NOTE: the caller, `upsample_one` line 163,
  passes `idea.strip()`, so the trailing/leading whitespace strip happens at the call
  site, not in `build_prompt`).
- Template source: read **verbatim** from the `.py` file (lines 72–80), NOT imported —
  the content has literal `\uNNNN`/`\n` that are invalid Python escapes:
  ```python
  src = open(_PROMPT_PATH, "r", encoding="utf-8").read()
  start = src.find('"""'); end = src.rfind('"""')
  return src[start + 3 : end]
  ```
  So the C template begins and ends with a literal `\n` (the chars right after the
  opening `"""` and before the closing `"""`), and its `\uNNNN`/`\n` sequences are
  **literal backslash sequences**, NOT resolved escapes. A port must embed exactly this
  7 892-char between-triple-quotes slice. (Anchors in §5.)

### 2.2 `extract_json` (B `_extract_json` 71–88 / C `extract_json` 98–112)

Identical logic in both (one trivial wording diff, behaviour identical):
```python
text = raw.strip()
fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)   # FIRST fenced block, lazy
if fence:
    text = fence.group(1).strip()
start = text.find("{")        # FIRST '{'
end   = text.rfind("}")       # LAST  '}'
if start == -1 or end == -1 or end <= start:
    return None
candidate = text[start : end + 1]
try:    return json.loads(candidate)      # B binds to `candidate`; C inlines text[start:end+1]
except json.JSONDecodeError:  return None
```
Steps, in order:
1. `raw.strip()`.
2. **Strip the FIRST ` ```json … ``` ` / ` ``` … ``` ` fence** via regex
   `` ```(?:json)?\s*(.*?)``` `` with `DOTALL`. The `json` tag is optional; `\s*` eats
   whitespace after the opening fence; `(.*?)` is **lazy** → captures up to the FIRST
   closing ` ``` `. If a fence matches, `text` = `group(1).strip()`. If no fence, `text`
   is left as the stripped raw.
3. Take the span from the **first `{`** to the **last `}`** (inclusive). If either is
   missing or `end <= start`, return `None`.
4. `json.loads` that span. On `JSONDecodeError`, return `None`.

**Edge cases (verified, see fixtures `b_extract_json` / `c_extract_json`):**
- `"no json at all"` → `None`.
- `"{this is not valid json}"` → span found, parse fails → `None`.
- `'{"a":1} trailing {"b":2}'` → span = first `{` … last `}` = the WHOLE string
  `{"a":1} trailing {"b":2}` → invalid → **`None`** (it does NOT pick the first object).
- ` ``` ` with no closing fence → regex doesn't match (needs a closing ` ``` `), falls
  through to the `{…}` span on the stripped raw.
- `"   "` (whitespace) → no `{` → `None`.
- Prose before/after a clean object → span trims prose, parses → dict.

**B vs C difference here:** none functional. B assigns `candidate` then `json.loads(candidate)`;
C calls `json.loads(text[start:end+1])` inline. Same result.

### 2.3 Per-element bbox fix — **B swaps, C does not**

**B `_convert_bbox` (90–106):** input boxes from Qwen3-VL are `[x1,y1,x2,y2]`, stored
format is `[y1,x1,y2,x2]`, so B **reorders** (no scaling):
```python
if not list/tuple or len != 4:                          return None
try: x1,y1,x2,y2 = [float(v) for v in bbox]             except: return None
x1,x2 = sorted((clamp0_1000(round(x1)), clamp0_1000(round(x2))))
y1,y2 = sorted((clamp0_1000(round(y1)), clamp0_1000(round(y2))))
if y2 <= y1 or x2 <= x1:                                return None
return [y1, x1, y2, x2]      # <-- y/x SWAPPED into stored order
```

**C `sanitize_bbox` (115–129):** the generation prompt already emits `[y1,x1,y2,x2]`,
so C does **NOT** swap — it clamps/sorts each axis pair in place:
```python
if not list/tuple or len != 4:                          return None
try: y1,x1,y2,x2 = [float(v) for v in bbox]             except: return None
y1,y2 = sorted((clamp0_1000(round(y1)), clamp0_1000(round(y2))))
x1,x2 = sorted((clamp0_1000(round(x1)), clamp0_1000(round(x2))))
if y2 <= y1 or x2 <= x1:                                return None
return [y1, x1, y2, x2]      # same order in
```
where `clamp0_1000(v) = max(0, min(1000, round(float(v))))`.

**Rounding detail (port trap):** `round()` is Python **banker's rounding**
(round-half-to-even): `round(0.5)==0`, `round(2.5)==2`, `round(10.6)==11`. Module A's
ported `_py_round`/`swap_bbox_xy_in_text` already implements this — reuse the same
half-to-even routine for the glue's bbox rounding. Verified cases in fixtures
`b_convert_bbox` / `c_sanitize_bbox` (e.g. `[10.6,20.4,300.5,400.5]`).

**Application loop (B `_normalize_caption` 108–124 / C `sanitize_caption` 132–147,
identical apart from the bbox fn name):**
```python
decon = data.get("compositional_deconstruction", {})
elements = decon.get("elements", []) if isinstance(decon, dict) else []
for el in elements:
    if isinstance(el, dict) and "bbox" in el:
        cleaned = <convert|sanitize>_bbox(el["bbox"])
        if cleaned is None: el.pop("bbox", None)     # drop bad bbox, keep element
        else:               el["bbox"] = cleaned
return normalize_caption_dict(data)                  # <-- module A (already ported)
```

### 2.4 The module-A tail

Both pipelines END by calling **`normalize_caption_dict(data)`** (already ported,
byte-exact). It drops `aspect_ratio`, enforces the photo/art_style branch + strict key
order, canonicalizes `medium`, swaps the style/element key order, and caps/uppercases
palettes. No new work — the glue just feeds it the bbox-fixed dict.

### 2.5 OUTPUT serialization — **a real port trap, differs B vs C, neither is module-A compact**

- **B** returns `json.dumps(data, ensure_ascii=False, indent=2)` (line 180) → **pretty
  2-space-indented** JSON, separators `,\n…` and `": "`. Saved to `<image>.json` by
  `BaseCaptioner.save_caption_for_file`. The dataloader RE-MINIFIES it at load via
  `digest_caption_string` (module A) — see `ideogram4.py:522` — so on-disk B is pretty.
- **C** prints `json.dumps(result, ensure_ascii=False, indent=indent)` where
  `indent = 2 if --pretty else None` (lines 331, 380/386). Default (no `--pretty`) →
  `indent=None` → **single line, default separators `", "` and `": "`** (note the SPACE
  after `,` and `:` — this is NOT module A's compact `(",",":")`).
- Both use `ensure_ascii=False` (CJK/Cyrillic/accents preserved literally, never `\uNNNN`).

**Port requirement:** the glue needs a `json.dumps`-faithful serializer that (a) preserves
the OrderedDict key order produced by module A, (b) emits Python's exact default/indent=2
separator+whitespace, (c) escapes strings exactly as Python's `json` does with
`ensure_ascii=False` (escape `"` `\` and control chars; leave non-ASCII raw). Module A's
existing compact `_serialize_into`/`to_model_string` is the right escape logic but the
WRONG separators — the glue serializer must support default (`", "/": "`) and pretty
(`indent=2`) modes.

### 2.6 B-only: malformed-JSON fallback

If `extract_json` returns `None`, **B** does NOT give up — it returns
`swap_bbox_xy_in_text(output_text)` (line 176; module A, already ported). This rewrites
every `"bbox":[x1,y1,x2,y2]` array in the raw text to `[y1,x1,y2,x2]` (clamp+sort) via
regex, leaving everything else byte-for-byte, and returns the **raw model string** with
boxes adapted (so the boxes still render even though the JSON is broken). **C** has no
fallback — on `None` it logs to stderr and returns `None` (the item becomes `null` in
the output list). Verified: fixture `b_full_glue` case 3 (truncated/unparseable) →
swapped-raw; `c_full_glue` "no json" → `None`.

### 2.7 C-only: `normalize_item` (input coercion, 192–204)

Accepts either a bare prompt string or `{"prompt": …, "aspect_ratio": …}`:
```python
if isinstance(item, str):                       idea, ar = item, default_ar
elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
    idea = item["prompt"]; ar = item.get("aspect_ratio") or default_ar
else:                                           return None
if not idea.strip():                            return None
return idea, ar
```
- Bare string → `(string, default_ar)`. Dict with str `prompt` → `(prompt, ar-or-default)`.
  `aspect_ratio` is taken only if **truthy** (`… or default_ar` — empty string falls back).
- Non-str/non-dict, or dict without a str `prompt`, or empty-after-strip → `None`
  (skipped). Verified: fixture `c_normalize_item` (incl. `{"prompt":"  "}` → `None`,
  `123` → `None`, `{"noprompt":1}` → `None`).

---

## 3. The LLM-call boundary (SMOKE-ONLY, not byte-gated)

The single non-deterministic box. **Interface:**

| | B (image → JSON) | C (idea → JSON) |
|---|---|---|
| Input | PIL image (RGB, resized to `max_res²` by `load_pil_image`) + built prompt | built prompt (text only) |
| Model | `Qwen/Qwen3-VL-8B-Instruct` (`Qwen3VLForConditionalGeneration`; the `…B-A…` MoE variant → `Qwen3VLMoeForConditionalGeneration`) | same |
| dtype / device | bf16 (config), cuda; optional fp8 quant via `optimum.quanto` | same |
| Prompt build | chat: `[{role:user, content:[{image}, {text:prompt}]}]` | chat: `[{role:user, content:[{text:prompt}]}]` |
| Tokenize | `processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")` | identical |
| Generate | `model.generate(**inputs, max_new_tokens=N)` — **greedy** (no sampling args); N≥3072 (B floors to `MIN_NEW_TOKENS=3072`) | `model.generate(**inputs, **gen_kwargs)`; `gen_kwargs` = `max_new_tokens=3072` + `do_sample=True,temperature=0.7` (default) OR `do_sample=False` if `temperature<=0` |
| Trim | `out_ids[len(in_ids):]` per row (drop the prompt tokens) | identical |
| Decode | `processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()` | identical |
| Output | raw model string → glue §2.2+ | raw model string → glue §2.2+ |

**What serenitymojo infra can host it:**
- The **prior** `pipeline/ideogram4_magic.mojo` runs **Qwen3-8B (text-only)**
  in-process via the ported `Qwen3Encoder.lm_logits_last` + `qwen3_magic.generate_greedy`
  (greedy). That can host the **C text-only path** model-wise, BUT (a) it's a different
  model (`Qwen3-8B`, not `Qwen3-VL-8B-Instruct`) and (b) it uses a different, older
  single-shot system prompt. To match ai-toolkit C you'd swap in the C template + the
  VL-8B weights/tokenizer. It is **greedy-only** — ai-toolkit C defaults to
  `temperature=0.7` sampling, so outputs won't match without a sampler (and sampling is
  non-deterministic regardless → smoke-only).
- **B (image path) has NO Mojo host.** `models/text_encoder/ideogram_qwen3vl.mojo` is the
  Ideogram-4 **conditioning encoder** (embeddings, no vision tower, no lm_head, no
  autoregressive generate). There is no ported Qwen3-VL **vision tower + generation loop**.
  B's LLM call would need either that port (large) or an external host.
- **Rust** has no in-process LLM at all → both B and C call out. Cleanest: **HTTP to an
  OpenAI-compatible server** (e.g. a `Qwen3-VL-8B-Instruct` llama.cpp/vLLM/TGI server, the
  same pattern as the existing magic-prompt `llama-server` flow referenced in
  `serenitymojo` docs). Rust supplies file I/O + template build + `extract_json` +
  bbox-fix + module-A normalize; the server supplies `generate`. For B, the HTTP request
  carries the base64 image in the chat content (`{"type":"image_url"…}` / provider
  multimodal format).

**Bottom line:** port the deterministic glue (§2) byte-exact and gate it; treat the LLM
as a pluggable backend behind a `generate(prompt[, image]) -> raw_string` trait/interface,
verified only by a smoke test (does a real call return parseable JSON that the glue
normalizes), never byte-gated.

---

## 4. Shared vs different (B vs C)

| Concern | Shared? | Detail |
|---|---|---|
| `extract_json` | **Shared** (behaviour-identical) | fence-strip → first`{`…last`}` → `json.loads`/None |
| `model.generate`/trim/decode wiring | **Shared** | same apply_chat_template + trim + batch_decode |
| module-A `normalize_caption_dict` tail | **Shared** | already ported |
| per-element bbox application loop | **Shared** | same `elements` walk + pop-on-None |
| bbox transform itself | **DIFFERS** | **B swaps x/y** (`_convert_bbox`), **C does not** (`sanitize_bbox`) |
| input | **DIFFERS** | B = image; C = text |
| system prompt | **DIFFERS** | B = 22 831-char observe-only (imported); C = 7 892-char upsampler (verbatim-read) |
| placeholders | **DIFFERS** | B: `{{aspect_ratio}}`,`{{user_instructions}}`. C: + `{{mode_directive}}`,`{{original_prompt}}` |
| mode directive | **C-only** | FAITHFUL vs CREATIVE constants → `{{mode_directive}}` |
| malformed-JSON fallback | **DIFFERS** | B → `swap_bbox_xy_in_text(raw)`; C → `None` |
| output serialization | **DIFFERS** | B `indent=2` pretty; C `indent=None` default-sep |
| aspect-ratio derivation | **B-only glue** | `compute_aspect_ratio(w,h)` snaps to ≤16 denominator (C takes AR as input) |
| input coercion | **C-only** | `normalize_item` (str | {prompt,aspect_ratio}) |

`compute_aspect_ratio` (B, 41–59) is extra deterministic glue worth porting+gating:
gcd-reduce; if both ≤16 return `rw:rh`; else search `q∈1..16`, `p=max(1,round(target*q))`,
keep min `|p/q − target|`. Verified cases in fixture `b_compute_aspect_ratio` (e.g.
`1023×768 → "4:3"`, `1920×1080 → "16:9"`, `0×100 → "1:1"`).

---

## 5. Template anchors (embed-verbatim checks)

| Template | Source | Chars | sha256 (first 16) | Head / tail |
|---|---|---|---|---|
| B `ideogram4_caption_prompt` | **imported** str | 22 831 | `9e9ce2a344f12de7…` | head `\n[META]\nfrozen: false…`; tail `…Analyze the provided image and emit the JSON caption.\n` |
| C `ideogram4_upsample_prompt` (between `"""`) | **verbatim** file slice | 7 892 | `7bf215b29f6587f7…` | head `\n[META]\nfrozen: false\ndescription: Faithful upsampler — lays a user prompt into …`; tail `…ASPECT RATIO: {{aspect_ratio}} (width:height).\nUser prompt: {{original_prompt}}\n` |

A port should assert its embedded template hashes to these. Note the **leading and
trailing `\n`** on both (they sit just inside the triple-quotes). The C template still
contains its 4 `{{…}}` placeholders un-substituted at embed time; B contains its 2.

---

## 6. Fixture plan (byte-exact glue gate — FEASIBLE, samples produced)

Mirror the existing module-A gate (`trainer/parity/ideogram_caption/gen_fixtures.py` +
`fixtures.json`, consumed byte-for-byte by `crates/ideogram-caption/tests/byte_exact.rs`
and the Mojo probe). Add a sibling `ideogram_bc_glue/` with:

`gen_glue_fixtures.py` — runs the deterministic glue (copied verbatim from B/C source) +
the REAL module-A `normalize_caption_dict`/`swap_bbox_xy_in_text`, no GPU/LLM, dumps
`{fn -> [{in,out}]}`. (Working generator already written; see scratchpad.)

Fixture groups (all produced and verified — sample I/O below):

| Group | Inputs | Gate |
|---|---|---|
| `b_compute_aspect_ratio` | `[w,h]` pairs incl. degenerate/odd ratios | exact string out |
| `b_build_prompt` | `(caption_prompt, aspect_ratio)` incl. None/empty/spaced | sha256+len of full prompt (+head/tail) |
| `b_convert_bbox` | clean / reversed / clamped / degenerate / wrong-len / wrong-type / half-round | exact list-or-None (note x/y SWAP) |
| `b_extract_json` | clean / fenced / bare-fence / prose-wrapped / no-json / invalid / two-objects / whitespace | exact dict-or-None |
| `b_full_glue` | fixed raw model strings: OK / fenced / malformed | exact final pretty JSON / swapped-raw |
| `c_build_prompt` | `(aspect_ratio, original_prompt, creative, instructions)` faithful+creative | sha256+len (+head/tail) |
| `c_sanitize_bbox` | same battery as B | exact list-or-None (NO swap) |
| `c_extract_json` | same battery as B | exact dict-or-None |
| `c_normalize_item` | str / dict / dict-no-ar / empty / non-str | exact `[idea,ar]`-or-None |
| `c_full_glue` | fixed raw model string: OK / no-json | exact compact-default JSON / None |
| `b_template_anchors`,`c_template_anchors` | — | embedded-template len+sha256 |

**Sample fixtures (real output, verbatim):**

`b_full_glue` (raw model string → final saved B JSON, `indent=2`, x/y swapped, module-A normalized):
```
in : {"high_level_description":"A red apple on a white table.","style_description":{"aesthetics":"clean","lighting":"soft daylight","photo":"studio shot","medium":"photograph","color_palette":["#ff0000","#ffffff"]},"compositional_deconstruction":{"background":"plain white","elements":[{"type":"obj","bbox":[200,100,400,300],"desc":"a red apple"},{"type":"text","bbox":[20,10,40,30],"text":"FRESH","desc":"label"}]}}
out: {
  "high_level_description": "A red apple on a white table.",
  "style_description": {
    "aesthetics": "clean",
    "lighting": "soft daylight",
    "photo": "studio shot",
    "medium": "photograph",
    "color_palette": [ "#FF0000", "#FFFFFF" ]      ← uppercased by module A; (rendered one-per-line at indent=2)
  },
  "compositional_deconstruction": {
    "background": "plain white",
    "elements": [
      { "type": "obj",  "bbox": [100,200,300,400], "desc": "a red apple" },   ← bbox x/y SWAPPED 200,100,400,300 -> 100,200,300,400
      { "type": "text", "bbox": [10, 20, 30, 40 ], "text": "FRESH", "desc": "label" }
    ]
  }
}
```
(elided to show the transforms; the byte-exact full string is in `glue_fixtures.json`.)

`b_full_glue` malformed (unparseable → `swap_bbox_xy_in_text(raw)`):
```
in : {"high_level_description":"broken","compositional_deconstruction":{"elements":[{"type":"obj","bbox":[200,100,400,300],"desc":"x"
out: {"high_level_description":"broken","compositional_deconstruction":{"elements":[{"type":"obj","bbox":[100,200,300,400],"desc":"x"
                                                                                          ↑ only the bbox array rewritten; rest byte-for-byte
```

`c_full_glue` (raw → compact-default JSON, NO x/y swap, palette uppercased, `#abc`→`#AABBCC`):
```
in : {"high_level_description":"sks dog running","style_description":{"aesthetics":"playful","lighting":"bright sun","medium":"illustration","art_style":"flat vector","color_palette":["#abc","#123456"]},"compositional_deconstruction":{"background":"grass field","elements":[{"type":"obj","bbox":[100,150,700,850],"desc":"sks dog"}]}}
out: {"high_level_description": "sks dog running", "style_description": {"aesthetics": "playful", "lighting": "bright sun", "medium": "illustration", "art_style": "flat vector", "color_palette": ["#AABBCC", "#123456"]}, "compositional_deconstruction": {"background": "grass field", "elements": [{"type": "obj", "bbox": [100, 150, 700, 850], "desc": "sks dog"}]}}
```
(note the `", "`/`": "` spaces — C default-sep, not module-A compact.)

`b_extract_json` two-objects edge: `'{"a":1} trailing {"b":2}'` → **`None`**.
`b_compute_aspect_ratio`: `1023×768 → "4:3"`, `1920×1080 → "16:9"`, `0×100 → "1:1"`.
`c_normalize_item`: `{"prompt":"  "}` → `None`; `123` → `None`; `{"prompt":"a dog","aspect_ratio":"16:9"}` → `["a dog","16:9"]`.

The full `glue_fixtures.json` (every case above with byte-exact `out`) is at
`/tmp/.../scratchpad/glue_fixtures.json`. Promote it to
`trainer/parity/ideogram_bc_glue/fixtures.json` and add the loaders to both ports.

---

## 7. Risk / effort — hardest parts, Rust vs Mojo

| Piece | Rust | Mojo |
|---|---|---|
| Template substitution (`str.replace` ×2/×4) | trivial | trivial |
| **Embedding the templates** (22 831 / 7 892 chars, must be byte-exact incl. leading/trailing `\n`, em-dashes, the C template's literal `\uNNNN`/`\n`) | **easy** (`include_str!` a snapshot, or a `const &str`) | **mild** — large multi-line raw string literal; the prior `_system_prompt()` in `ideogram4_magic.mojo` shows it works but is verbose. Embed via a generated file + hash-assert. |
| `extract_json` (regex fence + first/last brace + `json.loads`) | **easy** (`regex` + `serde_json`) | **moderate** — no regex stdlib; the fence-strip + brace-scan must be hand-rolled, but module A already hand-rolls a JSON parser + `swap_bbox_xy_in_text` regex-equivalent in pure Mojo, so the pattern exists. The lazy `(.*?)…``` ` first-fence and DOTALL semantics must be replicated by hand. |
| bbox transform (clamp/sort/**half-to-even round**/swap) | easy (reuse module A `_py_round`) | easy (reuse module A `_py_round`) |
| module-A normalize tail | done (ported) | done (ported) |
| **Output serializer** (`json.dumps` default-sep AND `indent=2`, `ensure_ascii=False`, OrderedDict order) | **easy** (`serde_json` `to_string` = default-sep; `to_string_pretty` is 2-space but verify it matches Python's exact pretty form — Python puts `": "` and `,\n`; serde pretty differs slightly → likely need a custom Python-faithful pretty writer) | **moderate** — extend module A's `_serialize_into` to emit default (`", "/": "`) and pretty (`indent=2`) variants; the escape logic is already there. |
| `compute_aspect_ratio` / `normalize_item` | trivial | trivial |
| LLM call | HTTP client to an OpenAI-compat VL server (B: base64 image in chat) — **the only real integration cost**, smoke-only | C text-only could reuse in-process `generate_greedy` (different model/prompt → smoke-only); B needs the unported VL vision tower or an external host |

**Biggest byte-exact trap (both langs): the output serializer.** Python's
`json.dumps(indent=2)` and `indent=None` (default `", "/": "`) are BOTH different from
module A's compact `(",",":")`. Get the separators + pretty whitespace + string-escape
(`ensure_ascii=False`) bit-identical to CPython or the gate fails. `serde_json`'s
`to_string` matches the default-sep C form out of the box; its `to_string_pretty` is
2-space but should be diffed against CPython's pretty form before trusting it. Mojo must
extend the existing serializer with two new modes.

**Second trap: the lazy-regex fence strip in `extract_json`** for Mojo (no regex stdlib) —
hand-roll the DOTALL lazy `` ```(?:json)?\s*(.*?)``` `` first-match.

**Lowest risk:** template substitution, bbox math, `normalize_item`, `compute_aspect_ratio`,
and the entire module-A tail (already ported). A byte-exact glue gate covering everything
except the LLM is straightforwardly feasible and the sample fixtures above prove it.
