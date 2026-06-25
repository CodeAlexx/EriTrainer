"""Ideogram caption module-A byte-exact parity oracle.

Runs the REAL ai-toolkit `toolkit/ideogram_caption.py` on a curated input set
(the 4 review oracles + the tricky unit helpers + a sweep of real dataset
captions) and dumps {fn -> [{in, out}]} to fixtures.json. The Rust (EriTrainer)
and Mojo (serenitymojo) ports load THIS file and must produce BYTE-IDENTICAL
output for every case.

stdlib + ai-toolkit only, no GPU/LLM. Run:
  /home/alex/serenityflow-v2/.venv/bin/python \
    /home/alex/EriTrainer/trainer/parity/ideogram_caption/gen_fixtures.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, "/home/alex/ai-toolkit")
from toolkit.ideogram_caption import (  # noqa: E402
    canon_medium,
    digest_caption_string,
    is_ideogram_caption_str,
    normalize_hex,
    swap_bbox_xy_in_text,
)

OUT = os.path.join(os.path.dirname(__file__), "fixtures.json")

fixtures = {
    "digest_caption_string": [],
    "swap_bbox_xy_in_text": [],
    "canon_medium": [],
    "normalize_hex": [],
    "is_ideogram_caption_str": [],
}

# ── Oracle 2: old-format structured JSON -> migrated compact model-string ──
oracle2_in = json.dumps(
    {
        "aspect_ratio": "1:1",
        "high_level_description": "A red apple on a table.",
        "style_description": {
            "aesthetics": "clean, minimal",
            "lighting": "soft daylight",
            "photo": "studio product shot",
            "medium": "Illustration.",
            "color_palette": ["#f00", "#FFFFFF", "#f00"],
        },
        "compositional_deconstruction": {
            "background": "plain white",
            "elements": [
                {"type": "obj", "color_palette": ["#ff0000"], "desc": "a red apple", "bbox": [100, 200, 300, 400]},
                {"type": "text", "color_palette": ["#000"], "text": "FRESH", "desc": "label", "bbox": [10, 20, 30, 40]},
            ],
        },
    }
)

# ── digest_caption_string cases ──
digest_inputs = [
    "vrtlEri2, The image depicts a plain prose caption that must pass through unchanged.",  # prose
    "",  # empty
    "   ",  # whitespace
    "{not really json",  # non-{ -> passthrough? starts with '{' so parses; malformed -> passthrough
    '{"foo":"bar"}',  # json but no compositional_deconstruction -> passthrough
    oracle2_in,  # the migration oracle
]
for s in digest_inputs:
    fixtures["digest_caption_string"].append({"in": s, "out": digest_caption_string(s)})
    fixtures["is_ideogram_caption_str"].append({"in": s, "out": is_ideogram_caption_str(s)})

# Real dataset sweep: all structured .json (digest/migrate path) + sample prose .txt (passthrough)
for p in sorted(glob.glob("/home/alex/ai-toolkit/datasets/gigerver3_json/*.json")):
    try:
        with open(p, "r", encoding="utf-8") as f:
            s = f.read()
    except Exception:
        continue
    fixtures["digest_caption_string"].append({"in": s, "out": digest_caption_string(s)})

for p in sorted(glob.glob("/home/alex/ai-toolkit/datasets/eri2_with_trigger/*.txt"))[:25]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            s = f.read()
    except Exception:
        continue
    fixtures["digest_caption_string"].append({"in": s, "out": digest_caption_string(s)})

# ── swap_bbox_xy_in_text (regex over raw text) ──
for s in [
    'garbage {"bbox":[200,100,400,300]} more {"bbox": [5, 6, 1, 2]} tail',
    'no bbox here at all',
    '{"bbox":[0,0,1000,1000]}',
    '{"bbox":[-5,2000,3,4]}',  # clamp
    'a {"bbox":[1,2,3,4]} b {"bbox":[4,3,2,1]} c',
]:
    fixtures["swap_bbox_xy_in_text"].append({"in": s, "out": swap_bbox_xy_in_text(s)})

# ── canon_medium ──
for s in ["Illustration.", "3D Render", "photograph", "PAINTING ", "Graphic Design", "weird-medium", ""]:
    fixtures["canon_medium"].append({"in": s, "out": canon_medium(s)})

# ── normalize_hex (None for invalid) ──
for s in ["#f00", "#FFFFFF", "#abc", "#GGGGGG", "red", "#1234567", ""]:
    fixtures["normalize_hex"].append({"in": s, "out": normalize_hex(s)})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(fixtures, f, ensure_ascii=False, indent=1)

counts = {k: len(v) for k, v in fixtures.items()}
print("OK wrote", OUT)
print("counts:", counts)
print("sample digest passthrough (prose):", fixtures["digest_caption_string"][0]["in"][:40],
      "==", fixtures["digest_caption_string"][0]["out"][:40],
      "->", fixtures["digest_caption_string"][0]["in"] == fixtures["digest_caption_string"][0]["out"])
