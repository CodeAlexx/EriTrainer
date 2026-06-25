//! Byte-exact parity gate for the B/C captioner GLUE against the ai-toolkit
//! oracle. Loads `trainer/parity/ideogram_bc_glue/fixtures.json` (groups:
//! b_build_prompt, b_extract_json, b_convert_bbox, b_compute_aspect_ratio,
//! b_full_glue, c_build_prompt, c_extract_json, c_sanitize_bbox, c_normalize_item,
//! c_full_glue, + b/c template_anchors) and asserts byte-identical output for
//! every case. Fail-loud: any miss panics with the diff.

use ideogram_caption::glue;
use serde_json::Value;

fn load_fixtures() -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path =
        std::path::Path::new(manifest).join("../../trainer/parity/ideogram_bc_glue/fixtures.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read glue fixtures at {}: {}", path.display(), e));
    serde_json::from_slice(&bytes).expect("glue fixtures.json must parse")
}

fn arr<'a>(fx: &'a Value, key: &str) -> &'a Vec<Value> {
    fx.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixtures missing array key {key}"))
}

/// Char-slice helper matching Python `s[:n]` / `s[-n:]` (Unicode codepoints).
fn char_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
fn char_suffix(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(n);
    chars[start..].iter().collect()
}

fn record(name: &str, ok: bool, total: &mut usize, pass: &mut usize, fails: &mut Vec<String>, detail: impl FnOnce() -> String) {
    *total += 1;
    if ok {
        *pass += 1;
    } else {
        fails.push(format!("[{name}] {}", detail()));
    }
}

#[test]
fn glue_byte_exact_all() {
    let fx = load_fixtures();
    let mut total = 0usize;
    let mut pass = 0usize;
    let mut fails: Vec<String> = Vec::new();

    // ---- b_compute_aspect_ratio: in [w,h] -> out "W:H" ----
    for c in arr(&fx, "b_compute_aspect_ratio") {
        let wh = c["in"].as_array().unwrap();
        let w = wh[0].as_i64().unwrap();
        let h = wh[1].as_i64().unwrap();
        let expected = c["out"].as_str().unwrap();
        let got = glue::compute_aspect_ratio(w, h);
        record("b_compute_aspect_ratio", got == expected, &mut total, &mut pass, &mut fails, || {
            format!("in=[{w},{h}] expected={expected:?} got={got:?}")
        });
    }

    // ---- b_build_prompt / c_build_prompt: sha256 + char-len + head/tail ----
    for c in arr(&fx, "b_build_prompt") {
        let inp = &c["in"];
        let cp = inp.get("caption_prompt");
        let cp_opt = match cp {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None, // null/None
        };
        let ar = inp["aspect_ratio"].as_str().unwrap();
        let got = glue::b_build_prompt(cp_opt, ar);
        check_build_prompt("b_build_prompt", c, &got, &mut total, &mut pass, &mut fails);
    }
    for c in arr(&fx, "c_build_prompt") {
        let inp = &c["in"];
        let ar = inp["aspect_ratio"].as_str().unwrap();
        let op = inp["original_prompt"].as_str().unwrap();
        let cr = inp["creative"].as_bool().unwrap();
        let ins = inp["instructions"].as_str().unwrap();
        let got = glue::c_build_prompt(glue::C_UPSAMPLE_PROMPT, ar, op, cr, ins);
        check_build_prompt("c_build_prompt", c, &got, &mut total, &mut pass, &mut fails);
    }

    // ---- b_convert_bbox / c_sanitize_bbox: in bbox -> out [..]|null ----
    for c in arr(&fx, "b_convert_bbox") {
        let got = glue::b_convert_bbox(&c["in"]);
        let ok = value_opt_eq(&c["out"], &got);
        record("b_convert_bbox", ok, &mut total, &mut pass, &mut fails, || {
            format!("in={} expected={} got={:?}", c["in"], c["out"], got)
        });
    }
    for c in arr(&fx, "c_sanitize_bbox") {
        let got = glue::c_sanitize_bbox(&c["in"]);
        let ok = value_opt_eq(&c["out"], &got);
        record("c_sanitize_bbox", ok, &mut total, &mut pass, &mut fails, || {
            format!("in={} expected={} got={:?}", c["in"], c["out"], got)
        });
    }

    // ---- b_extract_json / c_extract_json: in str -> out dict|null ----
    for (key, group) in [("b_extract_json", "b_extract_json"), ("c_extract_json", "c_extract_json")] {
        for c in arr(&fx, group) {
            let input = c["in"].as_str().unwrap();
            let got = glue::extract_json(input);
            let ok = value_opt_eq(&c["out"], &got);
            record(key, ok, &mut total, &mut pass, &mut fails, || {
                format!("in={input:?} expected={} got={:?}", c["out"], got)
            });
        }
    }

    // ---- b_full_glue: in str -> out str (byte-exact pretty/swapped-raw) ----
    for c in arr(&fx, "b_full_glue") {
        let input = c["in"].as_str().unwrap();
        let expected = c["out"].as_str().unwrap();
        let got = glue::b_full_glue(input);
        record("b_full_glue", got == expected, &mut total, &mut pass, &mut fails, || {
            format!(
                "in={:?}\n  EXPECTED ({} bytes): {:?}\n  GOT ({} bytes): {:?}",
                input, expected.len(), expected, got.len(), got
            )
        });
    }

    // ---- c_full_glue: in str -> out str|null ----
    for c in arr(&fx, "c_full_glue") {
        let input = c["in"].as_str().unwrap();
        let got = glue::c_full_glue(input);
        let ok = match &c["out"] {
            Value::Null => got.is_none(),
            Value::String(s) => got.as_deref() == Some(s.as_str()),
            other => panic!("unexpected c_full_glue out type {other:?}"),
        };
        record("c_full_glue", ok, &mut total, &mut pass, &mut fails, || {
            format!("in={input:?}\n  EXPECTED={}\n  GOT={:?}", c["out"], got)
        });
    }

    // ---- c_normalize_item: in (any), default_ar -> out [idea,ar]|null ----
    for c in arr(&fx, "c_normalize_item") {
        let default_ar = c["default_ar"].as_str().unwrap();
        let got = glue::normalize_item(&c["in"], default_ar);
        let ok = match &c["out"] {
            Value::Null => got.is_none(),
            Value::Array(a) => {
                let exp_idea = a[0].as_str().unwrap();
                let exp_ar = a[1].as_str().unwrap();
                got.as_ref().map(|(i, r)| i == exp_idea && r == exp_ar).unwrap_or(false)
            }
            other => panic!("unexpected c_normalize_item out type {other:?}"),
        };
        record("c_normalize_item", ok, &mut total, &mut pass, &mut fails, || {
            format!("in={} default_ar={default_ar:?} expected={} got={:?}", c["in"], c["out"], got)
        });
    }

    // ---- b/c template anchors: char-len + sha256 + head/tail ----
    check_template_anchor("b_template_anchors", &fx, glue::B_CAPTION_PROMPT, &mut total, &mut pass, &mut fails);
    check_template_anchor("c_template_anchors", &fx, glue::C_UPSAMPLE_PROMPT, &mut total, &mut pass, &mut fails);

    eprintln!("glue byte-exact parity: {pass}/{total} cases pass");
    if !fails.is_empty() {
        let shown: Vec<String> = fails.iter().take(20).cloned().collect();
        panic!("{} / {} glue cases FAILED:\n\n{}\n", fails.len(), total, shown.join("\n\n"));
    }
    assert_eq!(pass, total);
    // 61 list cases + 2 template-anchor groups = 63 assertions.
    assert_eq!(total, 63, "glue fixture count drift");
}

/// build_prompt fixtures store `out_sha256_len: [hex, char_len]`, `out_head`,
/// `out_tail`. Verify all four against the freshly built prompt.
fn check_build_prompt(
    name: &str,
    c: &Value,
    got: &str,
    total: &mut usize,
    pass: &mut usize,
    fails: &mut Vec<String>,
) {
    let sl = c["out_sha256_len"].as_array().unwrap();
    let exp_sha = sl[0].as_str().unwrap();
    let exp_len = sl[1].as_u64().unwrap() as usize;
    let exp_head = c["out_head"].as_str().unwrap();
    let exp_tail = c["out_tail"].as_str().unwrap();

    let got_sha = sha256_hex(got.as_bytes());
    let got_len = got.chars().count(); // Python len() = char count
    // head=120, tail=160 (B) / tail=200 (C) per the generator; derive from the
    // recorded anchor lengths so we don't hard-code per-group.
    let got_head = char_prefix(got, exp_head.chars().count());
    let got_tail = char_suffix(got, exp_tail.chars().count());

    let ok = got_sha == exp_sha && got_len == exp_len && got_head == exp_head && got_tail == exp_tail;
    *total += 1;
    if ok {
        *pass += 1;
    } else {
        fails.push(format!(
            "[{name}] in={}\n  sha exp={exp_sha} got={got_sha}\n  len exp={exp_len} got={got_len}\n  head_match={} tail_match={}",
            c["in"],
            got_head == exp_head,
            got_tail == exp_tail,
            name = name,
        ));
    }
}

fn check_template_anchor(
    name: &str,
    fx: &Value,
    template: &str,
    total: &mut usize,
    pass: &mut usize,
    fails: &mut Vec<String>,
) {
    let a = fx.get(name).unwrap_or_else(|| panic!("missing {name}"));
    let exp_len = a["len"].as_u64().unwrap() as usize;
    let exp_sha = a["sha256"].as_str().unwrap();
    let exp_head = a["head"].as_str().unwrap();
    let exp_tail = a["tail"].as_str().unwrap();

    let got_len = template.chars().count();
    let got_sha = sha256_hex(template.as_bytes());
    let got_head = char_prefix(template, exp_head.chars().count());
    let got_tail = char_suffix(template, exp_tail.chars().count());

    let ok = got_len == exp_len && got_sha == exp_sha && got_head == exp_head && got_tail == exp_tail;
    *total += 1;
    if ok {
        *pass += 1;
    } else {
        fails.push(format!(
            "[{name}] len exp={exp_len} got={got_len}; sha exp={exp_sha} got={got_sha}; head_match={} tail_match={}",
            got_head == exp_head,
            got_tail == exp_tail,
        ));
    }
}

/// Compare an expected fixture value (`null` or a JSON value) with an
/// `Option<Value>` result, value-equal (key order-insensitive for dicts is fine
/// here — we separately byte-gate full_glue output where order matters).
fn value_opt_eq(expected: &Value, got: &Option<Value>) -> bool {
    match (expected, got) {
        (Value::Null, None) => true,
        (Value::Null, Some(_)) => false,
        (exp, Some(g)) => exp == g,
        (_, None) => false,
    }
}

// --- minimal SHA-256 (no external dep; the crate stays a fast leaf) ---------

fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[test]
fn sha256_self_check() {
    // "abc" -> ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}
