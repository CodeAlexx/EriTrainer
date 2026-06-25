//! Deterministic B/C captioner GLUE — 1:1 Rust port of ai-toolkit
//! `extensions_built_in/captioner/Ideogram4Captioner.py` (B, image→JSON) and
//! `ui_scripts/upsample_ideogram4_caption.py` (C, idea→JSON), built on module A
//! (`super::normalize_caption_dict` / `super::swap_bbox_xy_in_text` /
//! `super::round_half_even`).
//!
//! Pipeline (both): `build_prompt` → [LLM.generate] → `extract_json` →
//! per-element bbox fix → module-A `normalize_caption_dict` → serialize.
//!
//! The LLM box is the only non-deterministic step; everything else here is
//! byte-gateable with no model. B and C diverge in: input (image vs text), which
//! system prompt, the bbox transform (B **swaps x/y**, C does **not**), the
//! malformed-JSON fallback (B → `swap_bbox_xy_in_text(raw)`, C → `None`), and the
//! OUTPUT serialization (B pretty `indent=2`; C default `", "/": "` separators).
//!
//! Byte-exactness notes vs the Python oracle (scope: ALL structural cases —
//! containers, separators, indent, key order, string escaping — AND every
//! IN-SCHEMA value; for non-schema EXTREME NUMERIC tokens see the F1 note below):
//! - **B output** = CPython `json.dumps(data, ensure_ascii=False, indent=2)`,
//!   which `serde_json::to_string_pretty` reproduces byte-for-byte for all
//!   structural + in-schema cases (verified: 2-space indent, one array element
//!   per line, empty `[]`/`{}` on one line, `": "` key sep, `,\n` item sep,
//!   non-ASCII raw; non-schema extreme numerics: see the F1/F3 notes).
//! - **C output** = CPython `json.dumps(result, ensure_ascii=False)` with the
//!   default separators `", "` / `": "` (note the SPACE after `,` and `:` — this
//!   is NOT module A's compact `(",",":")` and NOT serde's compact `to_string`).
//!   We drive a custom `serde_json` `Formatter` (`PyDefaultFormatter`) for it.
//! - String escaping under `ensure_ascii=False` matches between serde_json and
//!   CPython (escape `"` `\` and control chars; leave non-ASCII raw) — verified
//!   exhaustively in the module-A skeptic pass.
//!
//! ## F1 — number serialization on NON-SCHEMA numeric tokens
//! The captioner contract is schema-valid captions: bbox values are numeric
//! (0-1000 ints after the bbox fix), and EVERY other field (background,
//! high_level_description, desc, extras, …) is a STRING. So no non-bbox number
//! reaches the serializer in practice. A garbage/adversarial model COULD emit a
//! number in a passthrough position (extras are preserved verbatim by module A),
//! and CPython vs Rust format some numeric tokens differently. We enable
//! serde_json's `arbitrary_precision` feature, which preserves the original
//! numeric TOKEN on round-trip, so the cases an adversary is most likely to hit
//! match CPython: big-ints beyond i64/u64 (`9999999999999999999999`,
//! `18446744073709551616`, `-123…890`) stay literal, and canonical floats
//! (`0.1`, `2.0`, `1e-30`, `1e+30`, `0.333…`) match CPython `repr`. (Without it,
//! serde stores these as f64 and emits e.g. `1e+22` / `9.99…e-31`.) RESIDUAL
//! (documented, NOT fixed — equally out-of-schema, opposite direction): a
//! NON-canonical source token round-trips verbatim instead of being normalized
//! the way CPython `float.__repr__` would — e.g. `1.50`→Rust `1.50` vs CPython
//! `1.5`; `2e3`→`2e+3` vs `2000.0`; over-precision digits kept vs truncated to
//! 17 sig-figs. Bounded to non-bbox numeric passthrough (never occurs in a
//! schema-valid caption); not worth a bespoke CPython-`repr` float writer.

use crate::round_half_even;
use regex::Regex;
use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};
use serde_json::Value;
use std::sync::OnceLock;

// ===========================================================================
// Templates (embedded byte-exact; sha256/len gated by the template-anchor test)
// ===========================================================================

/// B system prompt — the **post-import** value of
/// `extensions_built_in/captioner/prompts/ideogram4_caption_prompt.py`'s
/// `ideogram4_caption_prompt` str (Python escapes already resolved). 22 831
/// chars; sha256 `9e9ce2a3…`. Still contains its 2 `{{…}}` placeholders.
pub const B_CAPTION_PROMPT: &str = include_str!("../templates/ideogram4_caption_prompt.txt");

/// C system prompt — the **verbatim** triple-quoted body of
/// `ideogram4_upsample_prompt.py` (the `src[find('"""')+3 : rfind('"""')]` slice;
/// NOT imported, so its literal `\uNNNN`/`\n` stay as backslash sequences). 7 892
/// chars; sha256 `7bf215b2…`. Still contains its 4 `{{…}}` placeholders. Note the
/// leading and trailing `\n` that sit just inside the triple-quotes.
pub const C_UPSAMPLE_PROMPT: &str = include_str!("../templates/ideogram4_upsample_prompt.txt");

/// C mode directive for the faithful (default) branch — `FAITHFUL_DIRECTIVE`
/// constant, verbatim from `upsample_ideogram4_caption.py` lines 40-46.
pub const FAITHFUL_DIRECTIVE: &str = "- **Fill in ONLY what the structure needs.** Add a concrete background shell, bounding boxes, and the required elements/text -- nothing else. Do NOT add new subjects, props, narrative, mood, or a setting the user did not specify. If the prompt names no location, keep the background minimal. If the prompt is sparse, the scene stays sparse.";

/// C mode directive for the `--creative` branch — `CREATIVE_DIRECTIVE` constant,
/// verbatim from `upsample_ideogram4_caption.py` lines 48-56.
pub const CREATIVE_DIRECTIVE: &str = "- **Expand the scene while keeping the user's idea intact.** Place the subject in a specific, believable setting and build a real background environment with fitting secondary details (props, depth layers, atmosphere) that serve the idea -- never a blank or 'plain' background when a setting can be implied. Everything you add must support, never replace or contradict, what the user asked for, and you must not introduce a different main subject. The FIDELITY rules above still hold: triggers verbatim, no invented appearance for a named person, no elaboration of a named style.";

// ===========================================================================
// Prompt-template substitution (sequential str.replace, exact order)
// ===========================================================================

/// B `Ideogram4Captioner.build_prompt(aspect_ratio)`: substitute
/// `{{aspect_ratio}}` then `{{user_instructions}}` into `B_CAPTION_PROMPT`.
/// `caption_prompt` is the user's additional-instructions block; empty/None →
/// the literal `"None."`. (Sequential `str.replace`, not `.format`.)
pub fn b_build_prompt(caption_prompt: Option<&str>, aspect_ratio: &str) -> String {
    let mut user_instructions = caption_prompt.unwrap_or("").trim().to_string();
    if user_instructions.is_empty() {
        user_instructions = "None.".to_string();
    }
    let p = B_CAPTION_PROMPT.replace("{{aspect_ratio}}", aspect_ratio);
    p.replace("{{user_instructions}}", &user_instructions)
}

/// C `upsample_ideogram4_caption.build_prompt(...)`: substitute
/// `{{mode_directive}}` (FAITHFUL/CREATIVE), `{{user_instructions}}`,
/// `{{aspect_ratio}}`, `{{original_prompt}}` into `template` — in that exact
/// order. `original_prompt` is passed already-stripped by the caller in Python
/// (`upsample_one` passes `idea.strip()`), so we do NOT strip it here; the caller
/// is responsible for matching that.
pub fn c_build_prompt(
    template: &str,
    aspect_ratio: &str,
    original_prompt: &str,
    creative: bool,
    instructions: &str,
) -> String {
    let directive = if creative {
        CREATIVE_DIRECTIVE
    } else {
        FAITHFUL_DIRECTIVE
    };
    let p = template.replace("{{mode_directive}}", directive);
    let trimmed = instructions.trim();
    let ui = if trimmed.is_empty() { "None." } else { trimmed };
    let p = p.replace("{{user_instructions}}", ui);
    let p = p.replace("{{aspect_ratio}}", aspect_ratio);
    p.replace("{{original_prompt}}", original_prompt)
}

// ===========================================================================
// extract_json — shared by B and C (behaviour identical)
// ===========================================================================

fn fence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Python: re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    //   (?s) = DOTALL so `.` matches newlines; lazy `(.*?)` => FIRST closing ```.
    RE.get_or_init(|| Regex::new(r"(?s)```(?:json)?\s*(.*?)```").unwrap())
}

/// Pull the JSON object out of raw model output, tolerating ```json fences and
/// stray preamble. Returns the parsed value or `None`. 1:1 with B `_extract_json`
/// / C `extract_json` (behaviour identical).
///
/// Steps: `strip()` → strip the FIRST ```…``` fence (lazy, DOTALL) if present →
/// span from the FIRST `{` to the LAST `}` (inclusive) → `json.loads` that span,
/// `None` on missing braces / `end <= start` / parse error. Key order preserved
/// (serde_json preserve_order) so the downstream normalize/serialize is exact.
pub fn extract_json(raw: &str) -> Option<Value> {
    let mut text: &str = raw.trim();
    let captured;
    if let Some(caps) = fence_re().captures(text) {
        // group(1).strip()
        captured = caps.get(1).map(|m| m.as_str().trim().to_string());
        if let Some(ref c) = captured {
            text = c.as_str();
        }
    }
    // first '{' .. last '}', byte indices (ASCII braces; safe on UTF-8).
    let start = text.find('{');
    let end = text.rfind('}');
    let (start, end) = match (start, end) {
        (Some(s), Some(e)) if e > s => (s, e),
        _ => return None,
    };
    let candidate = &text[start..=end];
    serde_json::from_str::<Value>(candidate).ok()
}

// ===========================================================================
// bbox fix — B swaps x/y, C does not. Both clamp 0-1000, sort, banker's round.
// ===========================================================================

/// `max(0, min(1000, round(float(v))))` — module-A banker's rounding + clamp.
/// Takes an already-parsed f64 (the typed callers pass numbers, not text).
///
/// F3 — DELIBERATE DEVIATION on non-schema `inf`/`nan` bbox values (robustness
/// over crash-parity): Python's bbox guard wraps ONLY `[float(v) for v in bbox]`,
/// not the following `round(...)`, so `float("inf")` succeeds and then
/// `round(inf)` raises `OverflowError` (`round(nan)` raises `ValueError`) —
/// which in ai-toolkit ABORTS the caption (B → the file returns `None`; C →
/// `upsample_one` has no inner guard, so the whole script crashes). Rust instead
/// saturates: `round_half_even(±inf)`/`nan` → clamped to the 0..1000 range, so a
/// pathological `"inf"`/`"nan"` bbox yields a clamped box and the caption still
/// emits. We intentionally do NOT replicate the crash/abort — a captioner must
/// not die on garbage model output. This is the same class as module A's
/// documented inf deviation (`lib.rs` `clamp_1000`), but a SEPARATE code path and
/// reachable from normal JSON (a string `"inf"` is valid inside a bbox array).
/// In a schema-valid caption bbox values are finite 0-1000 numbers, so this
/// never triggers in practice.
fn clamp_1000_f(v: f64) -> i64 {
    round_half_even(v).max(0).min(1000)
}

/// Coerce a JSON value to f64 exactly as Python `float(v)` would for the values
/// that reach `[float(x) for x in bbox]`: numbers convert; bool → 1.0/0.0
/// (Python `float(True)==1.0`); strings parse if numeric; everything else (null,
/// list, object, non-numeric string) raises in Python → here returns `None`,
/// which makes the caller drop the whole bbox.
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            // Python float() trims surrounding whitespace and accepts Unicode
            // digits; reuse module-A's Unicode-digit normalization for parity.
            // Covered & matching Python: `+5`, `1.`, `.5`, `1e2`, surrounding
            // whitespace, fullwidth/Arabic/Devanagari digits.
            //
            // F2 — DOCUMENTED DEVIATION on non-schema PEP-515 underscore digit
            // grouping: Python `float("1_000") == 1000.0` (also `"1_0"`,
            // `"1_000_000"`), but Rust `f64::from_str` rejects underscores → the
            // whole bbox is dropped (vs Python keeping the clamped box). We do NOT
            // strip underscores: bbox values in a schema-valid caption are JSON
            // NUMBERS (0-1000), not grouped-digit STRINGS, so this position never
            // carries `"1_000"` in practice; matching it would mean re-deriving
            // CPython's exact float-grammar acceptance for a garbage-only input.
            // (F3 — `"inf"`/`"nan"` strings parse here as ±inf/nan; the
            // saturate-instead-of-abort deviation is documented at `clamp_1000_f`.)
            let norm = crate::normalize_unicode_digits(s.trim());
            norm.parse::<f64>().ok()
        }
        _ => None,
    }
}

/// Parse a 4-element bbox into f64s, or `None` if it isn't a 4-list/tuple of
/// floatable values. Mirrors the shared guard in B `_convert_bbox` / C
/// `sanitize_bbox`.
fn parse_bbox4(bbox: &Value) -> Option<[f64; 4]> {
    let arr = match bbox {
        Value::Array(a) => a,
        _ => return None, // not list/tuple
    };
    if arr.len() != 4 {
        return None;
    }
    let mut out = [0.0f64; 4];
    for (i, el) in arr.iter().enumerate() {
        out[i] = value_to_f64(el)?; // any non-floatable => whole bbox dropped
    }
    Some(out)
}

/// B `_convert_bbox`: Qwen3-VL emits `[x1,y1,x2,y2]`; stored format is
/// `[y1,x1,y2,x2]`, so SWAP after clamp+sort. Returns the stored box or `None`.
pub fn b_convert_bbox(bbox: &Value) -> Option<Value> {
    let [x1, y1, x2, y2] = parse_bbox4(bbox)?;
    let (cx1, cx2) = sort2(clamp_1000_f(x1), clamp_1000_f(x2));
    let (cy1, cy2) = sort2(clamp_1000_f(y1), clamp_1000_f(y2));
    if cy2 <= cy1 || cx2 <= cx1 {
        return None;
    }
    // stored order [y1, x1, y2, x2]
    Some(int_array(&[cy1, cx1, cy2, cx2]))
}

/// C `sanitize_bbox`: the generation prompt already emits `[y1,x1,y2,x2]`, so do
/// NOT swap — clamp/sort each axis pair in place. Returns the box or `None`.
pub fn c_sanitize_bbox(bbox: &Value) -> Option<Value> {
    let [y1, x1, y2, x2] = parse_bbox4(bbox)?;
    let (cy1, cy2) = sort2(clamp_1000_f(y1), clamp_1000_f(y2));
    let (cx1, cx2) = sort2(clamp_1000_f(x1), clamp_1000_f(x2));
    if cy2 <= cy1 || cx2 <= cx1 {
        return None;
    }
    Some(int_array(&[cy1, cx1, cy2, cx2]))
}

fn sort2(a: i64, b: i64) -> (i64, i64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn int_array(vals: &[i64]) -> Value {
    Value::Array(vals.iter().map(|&v| Value::Number(v.into())).collect())
}

/// Apply the per-element bbox fix over `compositional_deconstruction.elements`
/// in place, then hand off to module-A `normalize_caption_dict`. The bbox fn is
/// `b_convert_bbox` (B) or `c_sanitize_bbox` (C). Bad bbox → drop the `"bbox"`
/// key, keep the element. Mirrors B `_normalize_caption` / C `sanitize_caption`.
fn apply_bbox_fix_and_normalize(mut data: Value, bbox_fix: impl Fn(&Value) -> Option<Value>) -> Value {
    if let Some(decon) = data.get_mut("compositional_deconstruction") {
        if let Some(decon_obj) = decon.as_object_mut() {
            if let Some(Value::Array(elements)) = decon_obj.get_mut("elements") {
                for el in elements.iter_mut() {
                    if let Some(obj) = el.as_object_mut() {
                        if obj.contains_key("bbox") {
                            let cleaned = obj.get("bbox").and_then(&bbox_fix);
                            match cleaned {
                                Some(b) => {
                                    obj.insert("bbox".to_string(), b);
                                }
                                None => {
                                    obj.remove("bbox");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    crate::normalize_caption_dict(&data)
}

// ===========================================================================
// Output serialization (B pretty indent=2; C default ", "/": " separators)
// ===========================================================================

/// CPython `json.dumps(..., indent=None)` default separators: `", "` between
/// items and `": "` after object keys. (serde_json's default `to_string` is the
/// compact `","`/`":"`, so we override the separators here.)
struct PyDefaultFormatter;

impl Formatter for PyDefaultFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Serialize like CPython `json.dumps(v, ensure_ascii=False)` (default
/// `indent=None`): single line, `", "`/`": "` separators, non-ASCII raw, key
/// order preserved. Used by C.
pub fn to_default_string(v: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PyDefaultFormatter);
    v.serialize(&mut ser).expect("Value serializes");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// Serialize like CPython `json.dumps(v, ensure_ascii=False, indent=2)`. Used by
/// B. `serde_json::to_string_pretty` is byte-identical to CPython's `indent=2`
/// (2-space indent, one element per line, empty `[]`/`{}` inline, `": "` key
/// sep) — verified against the oracle.
pub fn to_pretty_string(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("Value serializes")
}

// ===========================================================================
// Full post-LLM glue (deterministic; takes the raw model string)
// ===========================================================================

/// B full post-LLM glue: `extract_json` → if `None`, fall back to module-A
/// `swap_bbox_xy_in_text(raw)` (returns the raw string with bboxes adapted) →
/// else convert-bbox (x/y swap) + module-A normalize → pretty `indent=2` JSON.
/// Mirrors `get_caption_for_file`'s post-generation tail.
pub fn b_full_glue(raw_model_string: &str) -> String {
    match extract_json(raw_model_string) {
        None => crate::swap_bbox_xy_in_text(raw_model_string),
        Some(data) => {
            let normed = apply_bbox_fix_and_normalize(data, b_convert_bbox);
            to_pretty_string(&normed)
        }
    }
}

/// C full post-LLM glue: `extract_json` → if `None`, return `None` (no fallback)
/// → else sanitize-bbox (NO swap) + module-A normalize → default-separator
/// single-line JSON. Mirrors `upsample_one`'s tail + the `indent=None` print.
pub fn c_full_glue(raw_model_string: &str) -> Option<String> {
    let data = extract_json(raw_model_string)?;
    let normed = apply_bbox_fix_and_normalize(data, c_sanitize_bbox);
    Some(to_default_string(&normed))
}

// ===========================================================================
// B-only: compute_aspect_ratio
// ===========================================================================

/// Largest denominator allowed when snapping an image's aspect ratio (B's
/// `MAX_AR_DENOMINATOR`).
pub const MAX_AR_DENOMINATOR: i64 = 16;

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// B `compute_aspect_ratio(width, height)`: clean `W:H` snapped to a small
/// denominator. `<=0` either dim → `"1:1"`; gcd-reduce; if both reduced terms
/// `<= 16` return them; else search `q in 1..=16`, `p = max(1, round(target*q))`,
/// keep the `(p,q)` with min `|p/q - target|` (FIRST-wins on ties, matching the
/// Python `err < best[0]` strict-less comparison).
pub fn compute_aspect_ratio(width: i64, height: i64) -> String {
    if width <= 0 || height <= 0 {
        return "1:1".to_string();
    }
    let g = gcd(width, height);
    let rw = width / g;
    let rh = height / g;
    if rw <= MAX_AR_DENOMINATOR && rh <= MAX_AR_DENOMINATOR {
        return format!("{}:{}", rw, rh);
    }
    let target = width as f64 / height as f64;
    let mut best: Option<(f64, i64, i64)> = None;
    for q in 1..=MAX_AR_DENOMINATOR {
        // p = max(1, round(target * q)) — Python round() is banker's rounding.
        let p = round_half_even(target * q as f64).max(1);
        let err = (p as f64 / q as f64 - target).abs();
        match best {
            None => best = Some((err, p, q)),
            Some((be, _, _)) if err < be => best = Some((err, p, q)),
            _ => {}
        }
    }
    let (_, p, q) = best.expect("loop runs at least once");
    format!("{}:{}", p, q)
}

// ===========================================================================
// C-only: normalize_item
// ===========================================================================

/// C `normalize_item(item, default_aspect_ratio)`: accept a bare prompt string
/// or `{"prompt": str, "aspect_ratio"?: str}`. Returns `(idea, aspect_ratio)` or
/// `None` (malformed / non-str prompt / empty-after-strip). `aspect_ratio` is
/// taken from the dict only if TRUTHY (`item.get("aspect_ratio") or default` —
/// empty string / null / missing all fall back to default).
pub fn normalize_item(item: &Value, default_aspect_ratio: &str) -> Option<(String, String)> {
    let (idea, aspect_ratio): (String, String) = match item {
        Value::String(s) => (s.clone(), default_aspect_ratio.to_string()),
        Value::Object(map) => {
            // require a STRING `prompt`
            let prompt = match map.get("prompt") {
                Some(Value::String(s)) => s.clone(),
                _ => return None,
            };
            // `item.get("aspect_ratio") or default` — only a truthy value wins.
            // For aspect_ratio the realistic truthy value is a non-empty string;
            // Python `or` also rejects empty string / None / 0 / [] / {}.
            let ar = match map.get("aspect_ratio") {
                Some(v) if py_truthy(v) => match v {
                    Value::String(s) => s.clone(),
                    // A truthy non-string aspect_ratio is pathological; Python
                    // would carry the object through unchanged, but downstream it
                    // is only ever used as a string. Fall back to default to keep
                    // the (String,String) contract; gated cases are all strings.
                    _ => default_aspect_ratio.to_string(),
                },
                _ => default_aspect_ratio.to_string(),
            };
            (prompt, ar)
        }
        _ => return None,
    };
    if idea.trim().is_empty() {
        return None;
    }
    Some((idea, aspect_ratio))
}

/// Python `bool(x)` for the JSON values that can appear as an aspect_ratio.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
    }
}

// ===========================================================================
// LLM boundary (SMOKE-ONLY — never byte-gated)
// ===========================================================================

/// Optional image payload for the B (vision) path: raw bytes + a MIME type, sent
/// to an OpenAI-compatible multimodal server as a base64 `image_url` data URI.
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub mime: String, // e.g. "image/png", "image/jpeg"
}

/// The single non-deterministic step, behind a trait so the deterministic glue
/// (everything else in this module) can be unit-tested with a stub and wired to a
/// real Qwen3-VL-8B server in production. NOT byte-gated.
pub trait CaptionLlm {
    /// Run one generation. `prompt` is the fully-substituted system+user prompt;
    /// `image` is `Some(..)` for the B image path, `None` for the C text path.
    /// Returns the raw model string (post-decode, stripped) for the glue to feed
    /// into `extract_json`. Errors surface as `Err` (the caller decides whether
    /// to skip/log), matching B/C wrapping each call in try/except.
    fn generate(&self, prompt: &str, image: Option<&ImageInput>) -> Result<String, String>;
}

/// Test/dev stub: returns a canned string regardless of input. Lets the
/// deterministic glue be exercised end-to-end with no network/model.
pub struct StubLlm {
    pub response: String,
}

impl CaptionLlm for StubLlm {
    fn generate(&self, _prompt: &str, _image: Option<&ImageInput>) -> Result<String, String> {
        Ok(self.response.clone())
    }
}

/// HTTP client to an OpenAI-compatible vision-language chat-completions server
/// (e.g. a `Qwen/Qwen3-VL-8B-Instruct` llama.cpp / vLLM / TGI endpoint), matching
/// the existing serenitymojo magic-prompt `llama-server` pattern.
///
/// This crate has NO HTTP dependency (it must stay a fast leaf — serde_json +
/// serde + regex only). To avoid pulling reqwest/hyper here, the request is
/// emitted via a user-supplied transport closure that takes the JSON request
/// body string and returns the JSON response body string (the caller wires its
/// own HTTP client — the trainer already has one). The OpenAI request/response
/// SHAPING (chat messages, base64 image content, greedy params, choice parsing)
/// lives here so it is consistent and testable; only the socket I/O is injected.
pub struct OpenAiCompatLlm<F>
where
    F: Fn(&str) -> Result<String, String>,
{
    pub model: String,
    pub max_new_tokens: u32,
    /// Greedy by default (B is greedy; C defaults to temperature 0.7 but sampling
    /// is non-deterministic regardless, so for reproducible smoke tests prefer 0).
    pub temperature: f32,
    /// `transport(request_body_json) -> response_body_json`.
    pub transport: F,
}

impl<F> OpenAiCompatLlm<F>
where
    F: Fn(&str) -> Result<String, String>,
{
    /// Build the OpenAI `/v1/chat/completions` request body. For the B path the
    /// image is embedded as a base64 `image_url` data URI alongside the text, in
    /// a single user message (matching ai-toolkit's `[{image},{text}]` content).
    pub fn build_request(&self, prompt: &str, image: Option<&ImageInput>) -> String {
        let mut content: Vec<Value> = Vec::new();
        if let Some(img) = image {
            let data_uri = format!("data:{};base64,{}", img.mime, base64_encode(&img.bytes));
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": data_uri}
            }));
        }
        content.push(serde_json::json!({"type": "text", "text": prompt}));
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        });
        serde_json::to_string(&body).expect("request serializes")
    }

    /// Parse `choices[0].message.content` from an OpenAI response body, stripped
    /// (B/C both `.strip()` the decoded output).
    fn parse_response(&self, body: &str) -> Result<String, String> {
        let v: Value = serde_json::from_str(body).map_err(|e| format!("bad LLM response JSON: {e}"))?;
        v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| "LLM response missing choices[0].message.content".to_string())
    }
}

impl<F> CaptionLlm for OpenAiCompatLlm<F>
where
    F: Fn(&str) -> Result<String, String>,
{
    fn generate(&self, prompt: &str, image: Option<&ImageInput>) -> Result<String, String> {
        let req = self.build_request(prompt, image);
        let resp = (self.transport)(&req)?;
        self.parse_response(&resp)
    }
}

/// Standard base64 (RFC 4648, `+/`, `=` padding) for the image data URI. Kept
/// local so the crate needs no base64 dependency.
fn base64_encode(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(T[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(T[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ===========================================================================
// End-to-end driver (LLM + deterministic glue) — convenience entry points
// ===========================================================================

/// Full B pipeline given a prompt-built string, an LLM, and the image: generate
/// then run the deterministic post-LLM glue. (Prompt building / aspect-ratio
/// derivation are separate so the caller controls image dimensions.)
pub fn b_caption_from_raw(llm: &dyn CaptionLlm, prompt: &str, image: &ImageInput) -> Result<String, String> {
    let raw = llm.generate(prompt, Some(image))?;
    Ok(b_full_glue(&raw))
}

/// Full C pipeline given a prompt-built string and an LLM: generate then run the
/// deterministic post-LLM glue. Returns `None` if the model output had no
/// parseable JSON (matching C's behaviour).
pub fn c_caption_from_raw(llm: &dyn CaptionLlm, prompt: &str) -> Result<Option<String>, String> {
    let raw = llm.generate(prompt, None)?;
    Ok(c_full_glue(&raw))
}
