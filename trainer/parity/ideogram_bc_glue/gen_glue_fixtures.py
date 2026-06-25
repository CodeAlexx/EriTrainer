"""Generate BYTE-EXACT fixtures for the deterministic glue around module A
(B = Ideogram4Captioner image->JSON, C = upsample idea->JSON), WITHOUT loading
any LLM/torch. The glue functions (_extract_json, _convert_bbox, compute_aspect_ratio,
build_prompt for B; extract_json, sanitize_bbox, build_prompt, load_generation_prompt,
normalize_item for C) are pure Python. We copy them VERBATIM from source here so the
oracle is exactly what the real files do, then run module A's real normalize on top.
"""
import json, os, re, sys
from math import gcd

sys.path.insert(0, "/home/alex/ai-toolkit")
from toolkit.ideogram_caption import normalize_caption_dict, swap_bbox_xy_in_text  # module A (real)

AITK = "/home/alex/ai-toolkit"

# ============================ B-path glue (verbatim) ============================
MAX_AR_DENOMINATOR = 16

def b_compute_aspect_ratio(width, height):
    if width <= 0 or height <= 0:
        return "1:1"
    g = gcd(width, height)
    rw, rh = width // g, height // g
    if rw <= MAX_AR_DENOMINATOR and rh <= MAX_AR_DENOMINATOR:
        return f"{rw}:{rh}"
    target = width / height
    best = None
    for q in range(1, MAX_AR_DENOMINATOR + 1):
        p = max(1, round(target * q))
        err = abs(p / q - target)
        if best is None or err < best[0]:
            best = (err, p, q)
    return f"{best[1]}:{best[2]}"

def _b_template():
    src = open(os.path.join(AITK, "extensions_built_in/captioner/prompts/ideogram4_caption_prompt.py"), "r", encoding="utf-8").read()
    # mirror: from .prompts...import ideogram4_caption_prompt (it's a normal importable str)
    ns = {}
    exec(compile(src, "ideogram4_caption_prompt.py", "exec"), ns)
    return ns["ideogram4_caption_prompt"]

def b_build_prompt(caption_prompt, aspect_ratio):
    user_instructions = (caption_prompt or "").strip()
    if not user_instructions:
        user_instructions = "None."
    prompt = _b_template().replace("{{aspect_ratio}}", aspect_ratio)
    prompt = prompt.replace("{{user_instructions}}", user_instructions)
    return prompt

def b_extract_json(raw):
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start: end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def b_convert_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(1000, round(x1))), max(0, min(1000, round(x2)))))
    y1, y2 = sorted((max(0, min(1000, round(y1))), max(0, min(1000, round(y2)))))
    if y2 <= y1 or x2 <= x1:
        return None
    return [y1, x1, y2, x2]

def b_normalize_caption(data):
    decon = data.get("compositional_deconstruction", {})
    elements = decon.get("elements", []) if isinstance(decon, dict) else []
    if isinstance(elements, list):
        for el in elements:
            if isinstance(el, dict) and "bbox" in el:
                cleaned = b_convert_bbox(el["bbox"])
                if cleaned is None:
                    el.pop("bbox", None)
                else:
                    el["bbox"] = cleaned
    return normalize_caption_dict(data)

def b_full_glue(raw_model_string):
    """The full B post-LLM glue: extract_json -> (None? swap_bbox_xy_in_text raw)
    else normalize -> pretty JSON (indent=2). Mirrors get_caption_for_file tail."""
    data = b_extract_json(raw_model_string)
    if data is None:
        return swap_bbox_xy_in_text(raw_model_string)
    data = b_normalize_caption(data)
    return json.dumps(data, ensure_ascii=False, indent=2)

# ============================ C-path glue (verbatim) ============================
FAITHFUL_DIRECTIVE = (
    "- **Fill in ONLY what the structure needs.** Add a concrete background shell, "
    "bounding boxes, and the required elements/text -- nothing else. Do NOT add new "
    "subjects, props, narrative, mood, or a setting the user did not specify. If the "
    "prompt names no location, keep the background minimal. If the prompt is sparse, "
    "the scene stays sparse."
)
CREATIVE_DIRECTIVE = (
    "- **Expand the scene while keeping the user's idea intact.** Place the subject in "
    "a specific, believable setting and build a real background environment with fitting "
    "secondary details (props, depth layers, atmosphere) that serve the idea -- never a "
    "blank or 'plain' background when a setting can be implied. Everything you add must "
    "support, never replace or contradict, what the user asked for, and you must not "
    "introduce a different main subject. The FIDELITY rules above still hold: triggers "
    "verbatim, no invented appearance for a named person, no elaboration of a named style."
)

def c_load_generation_prompt():
    p = os.path.join(AITK, "extensions_built_in/captioner/prompts/ideogram4_upsample_prompt.py")
    src = open(p, "r", encoding="utf-8").read()
    start = src.find('"""')
    end = src.rfind('"""')
    if start == -1 or end <= start:
        raise RuntimeError("parse")
    return src[start + 3: end]

def c_build_prompt(template, aspect_ratio, original_prompt, creative=False, instructions=""):
    directive = CREATIVE_DIRECTIVE if creative else FAITHFUL_DIRECTIVE
    prompt = template.replace("{{mode_directive}}", directive)
    prompt = prompt.replace("{{user_instructions}}", instructions.strip() or "None.")
    prompt = prompt.replace("{{aspect_ratio}}", aspect_ratio)
    prompt = prompt.replace("{{original_prompt}}", original_prompt)
    return prompt

def c_extract_json(raw):
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None

def c_sanitize_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        y1, x1, y2, x2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    y1, y2 = sorted((max(0, min(1000, round(y1))), max(0, min(1000, round(y2)))))
    x1, x2 = sorted((max(0, min(1000, round(x1))), max(0, min(1000, round(x2)))))
    if y2 <= y1 or x2 <= x1:
        return None
    return [y1, x1, y2, x2]

def c_sanitize_caption(data):
    decon = data.get("compositional_deconstruction", {})
    elements = decon.get("elements", []) if isinstance(decon, dict) else []
    if isinstance(elements, list):
        for el in elements:
            if isinstance(el, dict) and "bbox" in el:
                cleaned = c_sanitize_bbox(el["bbox"])
                if cleaned is None:
                    el.pop("bbox", None)
                else:
                    el["bbox"] = cleaned
    return normalize_caption_dict(data)

def c_normalize_item(item, default_aspect_ratio):
    if isinstance(item, str):
        idea, aspect_ratio = item, default_aspect_ratio
    elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
        idea = item["prompt"]
        aspect_ratio = item.get("aspect_ratio") or default_aspect_ratio
    else:
        return None
    if not idea.strip():
        return None
    return [idea, aspect_ratio]

def c_full_glue(raw_model_string):
    """C full post-LLM glue: extract_json -> None? None else sanitize -> compact-list output.
    The script prints json.dumps(result) with indent=None for non-pretty single."""
    data = c_extract_json(raw_model_string)
    if data is None:
        return None
    return json.dumps(c_sanitize_caption(data), ensure_ascii=False)

# ============================ build fixtures ============================
EXTRACT_RAWS = [
    '{"high_level_description":"a red apple","compositional_deconstruction":{"background":"plain","elements":[]}}',
    '```json\n{"high_level_description":"fenced","compositional_deconstruction":{"background":"x","elements":[]}}\n```',
    '```\n{"high_level_description":"bare fence","compositional_deconstruction":{"background":"x","elements":[]}}\n```',
    'Here is your caption:\n{"high_level_description":"prose before","compositional_deconstruction":{"background":"x","elements":[]}}\nDone.',
    'no json at all, just prose',
    '{this is not valid json}',
    '{"a":1} trailing prose {"b":2}',  # outermost {..} spans both -> invalid parse -> None
    '   ',
]

CONVERT_BBOXES = [
    [100, 200, 300, 400],
    [400, 300, 200, 100],           # reversed -> sorted
    [-50, 50, 1200, 900],           # clamp
    [10, 10, 10, 400],              # zero-height after order -> None (y2<=y1)
    [1, 2, 3],                      # wrong length -> None
    "notalist",                     # wrong type -> None
    [10.6, 20.4, 300.5, 400.5],     # rounding
]

# B: a realistic raw VLM string (x1,y1,x2,y2 boxes, old-format-ish) and a malformed one.
B_RAW_OK = '{"high_level_description":"A red apple on a white table.","style_description":{"aesthetics":"clean","lighting":"soft daylight","photo":"studio shot","medium":"photograph","color_palette":["#ff0000","#ffffff"]},"compositional_deconstruction":{"background":"plain white","elements":[{"type":"obj","bbox":[200,100,400,300],"desc":"a red apple"},{"type":"text","bbox":[20,10,40,30],"text":"FRESH","desc":"label"}]}}'
B_RAW_FENCED = "```json\n" + B_RAW_OK + "\n```"
B_RAW_MALFORMED = '{"high_level_description":"broken","compositional_deconstruction":{"elements":[{"type":"obj","bbox":[200,100,400,300],"desc":"x"'  # truncated, unparseable

# C: faithful + creative directive build, and a sanitize chain.
C_RAW_OK = '{"high_level_description":"sks dog running","style_description":{"aesthetics":"playful","lighting":"bright sun","medium":"illustration","art_style":"flat vector","color_palette":["#abc","#123456"]},"compositional_deconstruction":{"background":"grass field","elements":[{"type":"obj","bbox":[100,150,700,850],"desc":"sks dog"}]}}'

fixtures = {
    "b_compute_aspect_ratio": [
        {"in": [w, h], "out": b_compute_aspect_ratio(w, h)}
        for (w, h) in [(1024,1024),(1920,1080),(1080,1920),(1023,768),(512,512),(0,100),(100,0),(3,1),(1000,1000),(1366,768),(2,3),(4000,3000)]
    ],
    "b_build_prompt": [
        {"in": {"caption_prompt": cp, "aspect_ratio": ar},
         "out_sha256_len": [__import__("hashlib").sha256(b_build_prompt(cp, ar).encode("utf-8")).hexdigest(), len(b_build_prompt(cp, ar))],
         "out_head": b_build_prompt(cp, ar)[:120],
         "out_tail": b_build_prompt(cp, ar)[-160:]}
        for (cp, ar) in [(None, "1:1"), ("", "16:9"), ("Focus on the dog.", "9:16"), ("  spaced  ", "4:5")]
    ],
    "b_convert_bbox": [{"in": bb, "out": b_convert_bbox(bb)} for bb in CONVERT_BBOXES],
    "b_extract_json": [{"in": r, "out": b_extract_json(r)} for r in EXTRACT_RAWS],
    "b_full_glue": [
        {"in": B_RAW_OK, "out": b_full_glue(B_RAW_OK)},
        {"in": B_RAW_FENCED, "out": b_full_glue(B_RAW_FENCED)},
        {"in": B_RAW_MALFORMED, "out": b_full_glue(B_RAW_MALFORMED)},
    ],
    "c_build_prompt": [
        {"in": {"aspect_ratio": ar, "original_prompt": op, "creative": cr, "instructions": ins},
         "out_sha256_len": [__import__("hashlib").sha256(c_build_prompt(c_load_generation_prompt(), ar, op, cr, ins).encode("utf-8")).hexdigest(), len(c_build_prompt(c_load_generation_prompt(), ar, op, cr, ins))],
         "out_head": c_build_prompt(c_load_generation_prompt(), ar, op, cr, ins)[:120],
         "out_tail": c_build_prompt(c_load_generation_prompt(), ar, op, cr, ins)[-200:]}
        for (ar, op, cr, ins) in [("1:1","a cat",False,""),("16:9","sks man on a beach",True,"Keep it minimal."),("auto","[trigger] castle",False,"No people.")]
    ],
    "c_sanitize_bbox": [{"in": bb, "out": c_sanitize_bbox(bb)} for bb in CONVERT_BBOXES],
    "c_extract_json": [{"in": r, "out": c_extract_json(r)} for r in EXTRACT_RAWS],
    "c_normalize_item": [
        {"in": it, "default_ar": "1:1", "out": c_normalize_item(it, "1:1")}
        for it in ["a cat", {"prompt": "a dog", "aspect_ratio": "16:9"}, {"prompt": "x"}, {"prompt": "  "}, "", {"noprompt": 1}, 123]
    ],
    "c_full_glue": [
        {"in": C_RAW_OK, "out": c_full_glue(C_RAW_OK)},
        {"in": "no json", "out": c_full_glue("no json")},
    ],
    "c_template_anchors": {
        "len": len(c_load_generation_prompt()),
        "sha256": __import__("hashlib").sha256(c_load_generation_prompt().encode("utf-8")).hexdigest(),
        "head": c_load_generation_prompt()[:80],
        "tail": c_load_generation_prompt()[-80:],
    },
    "b_template_anchors": {
        "len": len(_b_template()),
        "sha256": __import__("hashlib").sha256(_b_template().encode("utf-8")).hexdigest(),
        "head": _b_template()[:80],
        "tail": _b_template()[-80:],
    },
}

out = os.path.join(os.path.dirname(__file__), "fixtures.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixtures, f, ensure_ascii=False, indent=1)
print("wrote", out)
# Print a few decisive samples
print("=== b_full_glue OK out (head) ===")
print(fixtures["b_full_glue"][0]["out"][:400])
print("=== b_full_glue MALFORMED out ===")
print(fixtures["b_full_glue"][2]["out"][:200])
print("=== c_full_glue OK out ===")
print(fixtures["c_full_glue"][0]["out"][:400])
print("=== b extract '{a:1} trailing {b:2}' ->", fixtures["b_extract_json"][6]["out"])
print("=== b_compute_aspect_ratio 1023:768 ->", b_compute_aspect_ratio(1023,768))
