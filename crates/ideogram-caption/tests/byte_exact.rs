//! Byte-exact parity gate against the ai-toolkit oracle.
//!
//! Loads `trainer/parity/ideogram_caption/fixtures.json` (126 cases produced by
//! running the real Python `toolkit/ideogram_caption.py`) and asserts that each
//! ported function reproduces the oracle output BYTE-IDENTICALLY.
//!
//! Keys: digest_caption_string, swap_bbox_xy_in_text, canon_medium,
//! normalize_hex, is_ideogram_caption_str. Each is a list of {in, out}.

use ideogram_caption::{
    canon_medium, digest_caption_string, is_ideogram_caption_str, normalize_hex,
    swap_bbox_xy_in_text,
};
use serde_json::Value;

fn load_fixtures() -> Value {
    // The fixtures live in the trainer parity tree, a few levels up from this
    // crate. Resolve relative to CARGO_MANIFEST_DIR so the test is location-
    // independent.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join("../../trainer/parity/ideogram_caption/fixtures.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read fixtures at {}: {}", path.display(), e));
    serde_json::from_slice(&bytes).expect("fixtures.json must parse")
}

fn cases<'a>(fx: &'a Value, key: &str) -> &'a Vec<Value> {
    fx.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixtures missing key {key}"))
}

#[test]
fn byte_exact_all() {
    let fx = load_fixtures();

    let mut total = 0usize;
    let mut pass = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // ---- digest_caption_string: in (str) -> out (str) ----
    for c in cases(&fx, "digest_caption_string") {
        total += 1;
        let input = c["in"].as_str().expect("digest in is str");
        let expected = c["out"].as_str().expect("digest out is str");
        let got = digest_caption_string(input);
        if got == expected {
            pass += 1;
        } else {
            failures.push(format!(
                "[digest_caption_string]\n  IN ({} bytes): {:?}\n  EXPECTED ({} bytes): {:?}\n  GOT ({} bytes): {:?}",
                input.len(),
                truncate(input),
                expected.len(),
                truncate(expected),
                got.len(),
                truncate(&got),
            ));
        }
    }

    // ---- swap_bbox_xy_in_text: in (str) -> out (str) ----
    for c in cases(&fx, "swap_bbox_xy_in_text") {
        total += 1;
        let input = c["in"].as_str().expect("swap in is str");
        let expected = c["out"].as_str().expect("swap out is str");
        let got = swap_bbox_xy_in_text(input);
        if got == expected {
            pass += 1;
        } else {
            failures.push(format!(
                "[swap_bbox_xy_in_text]\n  IN: {:?}\n  EXPECTED: {:?}\n  GOT: {:?}",
                input, expected, got
            ));
        }
    }

    // ---- canon_medium: in (str) -> out (str) ----
    for c in cases(&fx, "canon_medium") {
        total += 1;
        let input = c["in"].as_str().expect("canon in is str");
        let expected = c["out"].as_str().expect("canon out is str");
        let got = canon_medium(input);
        if got == expected {
            pass += 1;
        } else {
            failures.push(format!(
                "[canon_medium]\n  IN: {:?}\n  EXPECTED: {:?}\n  GOT: {:?}",
                input, expected, got
            ));
        }
    }

    // ---- normalize_hex: in (str) -> out (str|null) ----
    for c in cases(&fx, "normalize_hex") {
        total += 1;
        let input = c["in"].as_str().expect("hex in is str");
        let got = normalize_hex(input);
        // Python None -> JSON null; Some(s) -> JSON string.
        let ok = match &c["out"] {
            Value::Null => got.is_none(),
            Value::String(s) => got.as_deref() == Some(s.as_str()),
            other => panic!("unexpected normalize_hex expected type: {other:?}"),
        };
        if ok {
            pass += 1;
        } else {
            failures.push(format!(
                "[normalize_hex]\n  IN: {:?}\n  EXPECTED: {:?}\n  GOT: {:?}",
                input, c["out"], got
            ));
        }
    }

    // ---- is_ideogram_caption_str: in (str) -> out (bool) ----
    for c in cases(&fx, "is_ideogram_caption_str") {
        total += 1;
        let input = c["in"].as_str().expect("is_ideogram in is str");
        let expected = c["out"].as_bool().expect("is_ideogram out is bool");
        let got = is_ideogram_caption_str(input);
        if got == expected {
            pass += 1;
        } else {
            failures.push(format!(
                "[is_ideogram_caption_str]\n  IN: {:?}\n  EXPECTED: {}\n  GOT: {}",
                truncate(input), expected, got
            ));
        }
    }

    eprintln!("byte-exact parity: {pass}/{total} cases pass");
    if !failures.is_empty() {
        let shown: Vec<String> = failures.iter().take(20).cloned().collect();
        panic!(
            "{} / {} cases FAILED:\n\n{}\n",
            failures.len(),
            total,
            shown.join("\n\n")
        );
    }
    assert_eq!(pass, total, "all cases must pass byte-identically");
    // Sanity: expected 126 total cases per the fixture spec.
    assert_eq!(total, 126, "fixture count drift: expected 126 cases");
}

fn truncate(s: &str) -> String {
    if s.len() <= 200 {
        s.to_string()
    } else {
        format!("{}…(+{} bytes)", &s[..s.char_indices().nth(200).map(|(i, _)| i).unwrap_or(s.len())], s.len() - 200)
    }
}
