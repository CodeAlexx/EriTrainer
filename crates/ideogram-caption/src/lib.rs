//! Shared helpers for Ideogram-4 structured JSON captions.
//!
//! This is a faithful 1:1 Rust port of ai-toolkit `toolkit/ideogram_caption.py`
//! (module A). It is the single source of truth for the caption schema so the
//! captioner, the prompt upsampler, the dataloader, and the model encoder all
//! agree. It encodes the official Ideogram-4 rules and, crucially, MIGRATES the
//! old caption format into the new one ("digest" old, emit new).
//!
//! Byte-exactness with the Python original is the contract:
//! - object key order is preserved on BOTH parse and emit (serde_json
//!   `preserve_order` / IndexMap), and output maps are built in the exact key
//!   order the Python emits;
//! - serialization matches Python `json.dumps(data, ensure_ascii=False,
//!   separators=(",", ":"))` — compact (no spaces after `,`/`:`) and non-ASCII
//!   passes through raw (no `\uXXXX`). serde_json's default `to_string` already
//!   produces exactly this byte layout;
//! - prose / non-caption input is returned UNCHANGED (the original string, not a
//!   reparse).
//!
//! Official schema (summary):
//! - three top-level keys: high_level_description (optional), style_description
//!   (optional), compositional_deconstruction (required).
//! - style_description holds EXACTLY ONE of `photo` (photographs) or `art_style`
//!   (illustration/painting/3D/graphic design), never both. Key order is strict
//!   and branch-dependent:
//!     photo branch:     aesthetics, lighting, photo, medium, color_palette
//!     non-photo branch: aesthetics, lighting, medium, art_style, color_palette
//! - medium is one of: photograph, illustration, 3d_render, painting,
//!   graphic_design
//! - color_palette: UPPERCASE #RRGGBB only, up to 16 per image / 5 per element.
//! - elements, strict key order:
//!     obj:  type, bbox, desc, color_palette
//!     text: type, bbox, text, desc, color_palette
//!   bbox is optional, normalized 0-1000, [y_min, x_min, y_max, x_max], top-left.

use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

/// The deterministic B/C captioner glue (1:1 port of ai-toolkit
/// `Ideogram4Captioner.py` and `ui_scripts/upsample_ideogram4_caption.py`),
/// built on top of module A above. See `glue` for the public surface.
pub mod glue;

/// style_description.color_palette cap.
pub const MAX_IMAGE_PALETTE: usize = 16;
/// per-element color_palette cap.
pub const MAX_ELEMENT_PALETTE: usize = 5;

/// Canonical medium tokens (official set).
pub const MEDIUM_OPTIONS: [&str; 5] = [
    "photograph",
    "illustration",
    "3d_render",
    "painting",
    "graphic_design",
];

/// Map common variants (including the old "Title." style) to the canonical
/// token. Anything not listed is treated as a custom medium and preserved
/// verbatim. Mirrors `_MEDIUM_ALIASES` exactly (insertion order is irrelevant
/// for lookup; we use a match for the same key/value pairs).
fn medium_alias(key: &str) -> Option<&'static str> {
    match key {
        "photograph" => Some("photograph"),
        "photo" => Some("photograph"),
        "illustration" => Some("illustration"),
        "3d render" => Some("3d_render"),
        "3d_render" => Some("3d_render"),
        "3d-render" => Some("3d_render"),
        "3drender" => Some("3d_render"),
        "render" => Some("3d_render"),
        "3d" => Some("3d_render"),
        "painting" => Some("painting"),
        "graphic design" => Some("graphic_design"),
        "graphic_design" => Some("graphic_design"),
        "graphic-design" => Some("graphic_design"),
        "graphic" => Some("graphic_design"),
        _ => None,
    }
}

/// Python `str.strip()` / `str.rstrip(...)` operate on Unicode whitespace. The
/// inputs here are caption mediums; Python's default `.strip()` strips
/// ASCII+Unicode whitespace. `str::trim` (Unicode whitespace) matches Python's
/// `.strip()` for these. We only need the trailing-`.` strip for `rstrip(".")`.
fn py_strip(s: &str) -> &str {
    s.trim()
}

// --- medium ----------------------------------------------------------------

/// Canonicalize a medium string to an official token when recognized, otherwise
/// return it stripped (custom mediums are allowed, preserved as-is).
///
/// Python operates on a `str`; non-str inputs return unchanged. In Rust the
/// typed entry point takes `&str`, so this always canonicalizes. The `Value`
/// variant (`canon_medium_value`) reproduces the "non-str returns unchanged"
/// branch for the dict-walking paths.
pub fn canon_medium(medium: &str) -> String {
    // key = medium.strip().rstrip(".").strip().lower()
    let s1 = py_strip(medium);
    let s2 = s1.trim_end_matches('.');
    let s3 = py_strip(s2);
    let key = s3.to_lowercase();
    if let Some(canon) = medium_alias(&key) {
        return canon.to_string();
    }
    // return medium.strip()
    py_strip(medium).to_string()
}

/// `Value`-typed `canon_medium`: non-string values are returned unchanged
/// (mirrors `if not isinstance(medium, str): return medium`).
fn canon_medium_value(medium: &Value) -> Value {
    match medium {
        Value::String(s) => Value::String(canon_medium(s)),
        other => other.clone(),
    }
}

/// True for the photograph branch (uses `photo`), False for the art_style
/// branch. Mirrors `canon_medium(...) == "photograph"`.
pub fn is_photo_medium(medium: &str) -> bool {
    canon_medium(medium) == "photograph"
}

// --- hex / palette ---------------------------------------------------------

fn hex6_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^#[0-9a-fA-F]{6}$").unwrap())
}

fn hex3_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^#[0-9a-fA-F]{3}$").unwrap())
}

/// Return an UPPERCASE #RRGGBB string, expanding #RGB -> #RRGGBB. None if
/// invalid. Mirrors `normalize_hex`.
pub fn normalize_hex(color: &str) -> Option<String> {
    let s = py_strip(color);
    if hex6_re().is_match(s) {
        // "#" + s[1:].upper()
        let mut out = String::with_capacity(7);
        out.push('#');
        out.push_str(&s[1..].to_uppercase());
        return Some(out);
    }
    if hex3_re().is_match(s) {
        // "#" + "".join(ch*2 for ch in s[1:]).upper()
        let mut body = String::with_capacity(6);
        for ch in s[1..].chars() {
            body.push(ch);
            body.push(ch);
        }
        let mut out = String::with_capacity(7);
        out.push('#');
        out.push_str(&body.to_uppercase());
        return Some(out);
    }
    None
}

/// `Value`-typed normalize_hex: non-string values yield None (mirrors
/// `if not isinstance(color, str): return None`).
fn normalize_hex_value(color: &Value) -> Option<String> {
    match color {
        Value::String(s) => normalize_hex(s),
        _ => None,
    }
}

/// Keep unique, valid, UPPERCASE hex colors in order, capped to `max_len`.
/// Returns the cleaned list, or None if nothing valid remains (drop the key).
/// Mirrors `sanitize_palette`. Operates on a JSON value: only lists/tuples are
/// considered (`isinstance(palette, (list, tuple))`); anything else → None.
fn sanitize_palette(palette: Option<&Value>, max_len: usize) -> Option<Vec<String>> {
    let arr = match palette {
        Some(Value::Array(a)) => a,
        _ => return None,
    };
    // dedupe by NORMALIZED hex (first occurrence wins, order preserved).
    let mut seen: Vec<String> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for c in arr {
        let h = match normalize_hex_value(c) {
            Some(h) => h,
            None => continue,
        };
        if seen.iter().any(|x| x == &h) {
            continue;
        }
        seen.push(h.clone());
        out.push(h);
        if out.len() >= max_len {
            break;
        }
    }
    // `return out or None` — empty list drops the key.
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn string_vec_to_value(v: Vec<String>) -> Value {
    Value::Array(v.into_iter().map(Value::String).collect())
}

// --- style -----------------------------------------------------------------

/// Reorder/clean style_description into the correct branch (photo vs art_style)
/// with the strict key order, canonical medium, and uppercase palette. Accepts
/// the old shape (always `photo`) and migrates it based on the medium.
/// Mirrors `normalize_style`. Non-dict input returns unchanged.
fn normalize_style(style: &Value) -> Value {
    let map = match style {
        Value::Object(m) => m,
        other => return other.clone(),
    };

    // raw_medium = style.get("medium")
    let raw_medium = map.get("medium");
    // medium = canon_medium(raw_medium) if raw_medium is not None else None
    // (Python: a present-but-null `medium` is treated as None.)
    let medium: Option<Value> = match raw_medium {
        None | Some(Value::Null) => None,
        Some(v) => Some(canon_medium_value(v)),
    };

    // has_photo = bool(style.get("photo")); has_art = bool(style.get("art_style"))
    let has_photo = truthy(map.get("photo"));
    let has_art = truthy(map.get("art_style"));

    // medium_in_options: only a *string* canon medium can be in MEDIUM_OPTIONS.
    let medium_str: Option<&str> = match &medium {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let medium_in_options = medium_str.map(|s| MEDIUM_OPTIONS.contains(&s)).unwrap_or(false);

    // Decide the branch.
    let photo_branch = if medium_in_options {
        medium_str == Some("photograph")
    } else if has_art && !has_photo {
        false
    } else {
        true
    };

    // photo_val = style.get("photo") if has_photo else None
    let photo_val: Option<&Value> = if has_photo { map.get("photo") } else { None };
    // art_val = style.get("art_style") if has_art else None
    let art_val: Option<&Value> = if has_art { map.get("art_style") } else { None };

    let mut out: Map<String, Value> = Map::new();
    if let Some(v) = map.get("aesthetics") {
        out.insert("aesthetics".to_string(), v.clone());
    }
    if let Some(v) = map.get("lighting") {
        out.insert("lighting".to_string(), v.clone());
    }

    if photo_branch {
        // aesthetics, lighting, photo, medium, color_palette
        // val = photo_val if photo_val is not None else art_val
        let val = if photo_val.is_some() { photo_val } else { art_val };
        if let Some(v) = val {
            out.insert("photo".to_string(), v.clone());
        }
        if let Some(m) = &medium {
            out.insert("medium".to_string(), m.clone());
        }
    } else {
        // aesthetics, lighting, medium, art_style, color_palette
        if let Some(m) = &medium {
            out.insert("medium".to_string(), m.clone());
        }
        // val = art_val if art_val is not None else photo_val
        let val = if art_val.is_some() { art_val } else { photo_val };
        if let Some(v) = val {
            out.insert("art_style".to_string(), v.clone());
        }
    }

    let pal = sanitize_palette(map.get("color_palette"), MAX_IMAGE_PALETTE);
    if let Some(p) = pal {
        out.insert("color_palette".to_string(), string_vec_to_value(p));
    }

    // Preserve any unexpected extra keys at the end rather than dropping them.
    const KNOWN: [&str; 6] = [
        "aesthetics",
        "lighting",
        "photo",
        "art_style",
        "medium",
        "color_palette",
    ];
    for (k, v) in map.iter() {
        if !KNOWN.contains(&k.as_str()) {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

// --- element ---------------------------------------------------------------

/// Reorder an element's keys to the strict schema order and uppercase its
/// palette. obj: type, bbox, desc, color_palette. text: type, bbox, text, desc,
/// color_palette. bbox is kept verbatim. Mirrors `normalize_element`. Non-dict
/// input returns unchanged.
fn normalize_element(el: &Value) -> Value {
    let map = match el {
        Value::Object(m) => m,
        other => return other.clone(),
    };

    // etype = el.get("type", "obj")
    let etype_val: Value = map.get("type").cloned().unwrap_or(Value::String("obj".to_string()));
    let etype_is_text = matches!(&etype_val, Value::String(s) if s == "text");

    let mut out: Map<String, Value> = Map::new();
    out.insert("type".to_string(), etype_val);

    // if el.get("bbox") is not None: out["bbox"] = el["bbox"]
    if let Some(b) = map.get("bbox") {
        if !b.is_null() {
            out.insert("bbox".to_string(), b.clone());
        }
    }

    if etype_is_text {
        if let Some(v) = map.get("text") {
            out.insert("text".to_string(), v.clone());
        }
        if let Some(v) = map.get("desc") {
            out.insert("desc".to_string(), v.clone());
        }
    } else {
        if let Some(v) = map.get("desc") {
            out.insert("desc".to_string(), v.clone());
        }
    }

    let pal = sanitize_palette(map.get("color_palette"), MAX_ELEMENT_PALETTE);
    if let Some(p) = pal {
        out.insert("color_palette".to_string(), string_vec_to_value(p));
    }

    // Preserve any extras (e.g. future keys) at the end.
    // `for k, v in el.items(): if k not in out and k != "color_palette": out[k] = v`
    for (k, v) in map.iter() {
        if !out.contains_key(k) && k != "color_palette" {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

// --- caption dict ----------------------------------------------------------

/// Normalize a parsed caption dict: drop input-only aspect_ratio, enforce
/// top-level key order, normalize style and every element. Returns a new ordered
/// object. Accepts old-format captions and emits new. Mirrors
/// `normalize_caption_dict`. Non-dict input returns unchanged.
pub fn normalize_caption_dict(data: &Value) -> Value {
    let map = match data {
        Value::Object(m) => m,
        other => return other.clone(),
    };

    let mut out: Map<String, Value> = Map::new();
    if let Some(v) = map.get("high_level_description") {
        out.insert("high_level_description".to_string(), v.clone());
    }
    if let Some(v) = map.get("style_description") {
        out.insert("style_description".to_string(), normalize_style(v));
    }

    // decon = data.get("compositional_deconstruction")
    match map.get("compositional_deconstruction") {
        Some(Value::Object(decon)) => {
            let mut nd: Map<String, Value> = Map::new();
            if let Some(v) = decon.get("background") {
                nd.insert("background".to_string(), v.clone());
            }
            // els = decon.get("elements"); if isinstance(els, list): ...
            if let Some(Value::Array(els)) = decon.get("elements") {
                let normed: Vec<Value> = els.iter().map(normalize_element).collect();
                nd.insert("elements".to_string(), Value::Array(normed));
            }
            for (k, v) in decon.iter() {
                if k != "background" && k != "elements" {
                    nd.insert(k.clone(), v.clone());
                }
            }
            out.insert("compositional_deconstruction".to_string(), Value::Object(nd));
        }
        Some(other) if !other.is_null() => {
            // elif decon is not None: out[...] = decon
            out.insert("compositional_deconstruction".to_string(), other.clone());
        }
        _ => {}
    }

    // Preserve trailing extras (aspect_ratio is dropped — never re-added here).
    const KNOWN: [&str; 4] = [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
        "aspect_ratio",
    ];
    for (k, v) in map.iter() {
        if !KNOWN.contains(&k.as_str()) {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

// --- truthiness ------------------------------------------------------------

/// Python `bool(x)` for the JSON values that can appear here. `None`/missing →
/// false; `null` → false; `""`/`[]`/`{}`/`0`/`0.0`/`false` → false; otherwise
/// true. Used for `bool(style.get("photo"))` etc.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => {
            // 0 / 0.0 are falsy.
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                true
            }
        }
    }
}

// --- bbox text rewrite -----------------------------------------------------

fn bbox_text_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Mirror Python `_BBOX_TEXT_RE` exactly:
    //   r'"bbox"\s*:\s*\[\s*'
    //   r"(-?\d+(?:\.\d+)?)\s*,\s*" x3
    //   r"(-?\d+(?:\.\d+)?)\s*\]"
    RE.get_or_init(|| {
        Regex::new(
            r#""bbox"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"#,
        )
        .unwrap()
    })
}

/// Block-zero code points for every Unicode `Nd` (decimal-digit) script.
/// AUTO-GENERATED from the Unicode 15.0.0 character database (the same data
/// CPython's `float()` consults via `Py_UNICODE_TODECIMAL`). For any `Nd`
/// digit `d`, its value 0-9 equals `d - zero`, where `zero` is the largest
/// entry `<= d`. Verified exhaustively (68 blocks, every `Nd` codepoint, zero
/// mismatches). The `regex` crate's `\d` (= `\p{Nd}`) matches exactly this set,
/// so this table covers everything `_BBOX_TEXT_RE` can capture. Regenerate from
/// `unicodedata` if the Unicode version is bumped.
pub(crate) const ND_DIGIT_ZEROS: [u32; 68] = [
    0x0030, 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6,
    0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0DE6, 0x0E50, 0x0ED0,
    0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80,
    0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900,
    0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066,
    0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0,
    0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60,
    0x16AC0, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140,
    0x1E2F0, 0x1E4F0, 0x1E950, 0x1FBF0,
];

/// Value 0-9 of a Unicode `Nd` decimal digit, or `None` if `c` is not one.
/// Mirrors CPython's per-character decimal-digit conversion in `float()`.
fn nd_digit_value(c: char) -> Option<u32> {
    let cp = c as u32;
    // Largest block-zero <= cp (table is sorted ascending).
    let mut zero = None;
    for &z in ND_DIGIT_ZEROS.iter() {
        if z <= cp {
            zero = Some(z);
        } else {
            break;
        }
    }
    let z = zero?;
    let v = cp - z;
    if v <= 9 {
        Some(v)
    } else {
        None
    }
}

/// Replicate the digit acceptance of CPython `float()`: convert every Unicode
/// `Nd` decimal digit (fullwidth, Arabic-Indic, Devanagari, ...) to its ASCII
/// 0-9 form, leaving the only structural chars the bbox regex can produce —
/// ASCII `-` and `.` — untouched. The regex `_BBOX_TEXT_RE` matches a number as
/// `-?\d+(?:\.\d+)?` with ASCII-literal `-`/`.` and Unicode `\d`, so a captured
/// group can only contain ASCII `-`/`.` plus `Nd` digits; nothing else needs
/// handling (e.g. fullwidth `．`/`－` are never matched). After this rewrite the
/// string parses with `f64::from_str` exactly as Python's `float()` would.
pub(crate) fn normalize_unicode_digits(v: &str) -> std::borrow::Cow<'_, str> {
    if v.is_ascii() {
        // Fast path: pure-ASCII numbers (the overwhelmingly common case) need no
        // conversion.
        return std::borrow::Cow::Borrowed(v);
    }
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match nd_digit_value(c) {
            Some(d) => out.push((b'0' + d as u8) as char),
            None => out.push(c), // ASCII '-' / '.' (or anything else) verbatim
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Python `round(float(v))` then clamp to [0,1000]. Mirrors `_clamp_1000`.
/// Python's `round` is banker's rounding (round-half-to-even) on floats.
///
/// Two byte-exactness notes vs the Python oracle:
/// - Unicode decimal digits: Python `float("１")` accepts any `Nd` digit, so we
///   first normalize them to ASCII (see `normalize_unicode_digits`) before
///   `f64::from_str`, which rejects them. Without this, a Unicode-digit bbox
///   would parse to 0 and rewrite to `[0,0,0,0]` instead of the real values.
/// - DELIBERATE DEVIATION (robustness over crash-parity): for an astronomically
///   large integer (e.g. ~400 nines), `f64::from_str` yields `+inf`; Python's
///   `round(float(...))` instead raises `OverflowError`, which would propagate
///   out of `swap_bbox_xy_in_text` and ABORT the whole digest/caption call. We
///   intentionally do NOT replicate that crash — `round_half_even(inf)` saturates
///   and we clamp to 1000, so a pathological box yields a clamped value rather
///   than taking down the captioner. This is the one place the port is not
///   byte-/behavior-identical to Python, and it is deliberate (a captioner must
///   not crash on garbage model output). Exposure is nil in practice (a model
///   would have to emit a 400-digit bbox coordinate).
fn clamp_1000(v: &str) -> i64 {
    let normalized = normalize_unicode_digits(v);
    let f: f64 = normalized.parse().unwrap_or(0.0);
    let r = round_half_even(f);
    r.max(0).min(1000)
}

/// Round-half-to-even (Python 3 `round`) to an integer. Shared with the B/C
/// glue's bbox rounding (`pub(crate)` so `glue` can reuse the exact routine).
pub(crate) fn round_half_even(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // exactly .5 -> round to even
        let fl = floor as i64;
        if fl % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    };
    rounded as i64
}

/// Swap every [x1,y1,x2,y2] bbox to the stored [y1,x1,y2,x2] order directly in
/// the raw model output -- clamping each value to 0-1000 and ordering each axis
/// pair. Never parses the surrounding JSON, so it works even on malformed
/// output. Only `"bbox":[n,n,n,n]` arrays are touched; everything else is left
/// byte-for-byte. Mirrors `swap_bbox_xy_in_text`.
pub fn swap_bbox_xy_in_text(text: &str) -> String {
    let re = bbox_text_re();
    re.replace_all(text, |caps: &regex::Captures| {
        let x1 = clamp_1000(&caps[1]);
        let y1 = clamp_1000(&caps[2]);
        let x2 = clamp_1000(&caps[3]);
        let y2 = clamp_1000(&caps[4]);
        // cx1, cx2 = sorted((clamp(x1), clamp(x2)))
        let (cx1, cx2) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        // cy1, cy2 = sorted((clamp(y1), clamp(y2)))
        let (cy1, cy2) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        format!("\"bbox\":[{},{},{},{}]", cy1, cx1, cy2, cx2)
    })
    .into_owned()
}

// --- top-level entry points ------------------------------------------------

/// True if text parses as a JSON object with a compositional_deconstruction
/// block. Mirrors `is_ideogram_caption_str`.
pub fn is_ideogram_caption_str(text: &str) -> bool {
    let t = text.trim();
    if !t.starts_with('{') {
        return false;
    }
    match serde_json::from_str::<Value>(t) {
        Ok(Value::Object(map)) => matches!(map.get("compositional_deconstruction"), Some(Value::Object(_))),
        _ => false,
    }
}

/// Serialize a caption value to the compact, model-ready string the renderer
/// wants. Mirrors `to_model_string` =
/// `json.dumps(data, ensure_ascii=False, separators=(",",":"))`.
///
/// serde_json's default `to_string` already produces compact (no-space) output
/// and passes non-ASCII through raw (no `\uXXXX`), matching Python exactly.
pub fn to_model_string(data: &Value) -> String {
    serde_json::to_string(data).expect("caption Value is always serializable")
}

/// Parse, normalize (migrating old format), and return the compact model-ready
/// string. Returns the input UNCHANGED if it is not an Ideogram structured
/// caption (plain-text captions pass straight through). Mirrors
/// `digest_caption_string`.
pub fn digest_caption_string(text: &str) -> String {
    let t = text.trim();
    if !t.starts_with('{') {
        return text.to_string();
    }
    // object_pairs_hook=OrderedDict -> serde_json preserve_order keeps key order.
    let data: Value = match serde_json::from_str(t) {
        Ok(v) => v,
        Err(_) => return text.to_string(),
    };
    let is_caption = match &data {
        Value::Object(map) => matches!(map.get("compositional_deconstruction"), Some(Value::Object(_))),
        _ => false,
    };
    if !is_caption {
        return text.to_string();
    }
    to_model_string(&normalize_caption_dict(&data))
}
