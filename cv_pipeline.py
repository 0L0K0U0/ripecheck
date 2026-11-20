"""
cv_pipeline.py - RipeCheck computer vision pipeline.

Grades an oil palm fresh fruit bunch (FFB) from one RGB photo using classical
computer vision (OpenCV + NumPy). There is no trained model: grading is a short
set of rules over colour and texture features, with thresholds that can be
calibrated on labelled images (see calibrate.py).

Techniques used (AIT204 Assignment 2):
  1. Filtering                  bilateral filter (smooths noise, keeps edges)
  2. Contrast enhancement       CLAHE on the LAB lightness channel
  3. Colour space conversion    BGR -> HSV and LAB
  4. Segmentation               GrabCut from the user's box (or HSV colour
                                rules, or k-means in LAB), foliage removed by hue
  5. Morphological operations   opening, closing, largest connected component
  6. Edge detection             Canny, edge density inside the bunch
  7. Contour analysis           holes (empty sockets), solidity; colour and
                                texture features -> rule-based grade

Flow: resize -> crop to the user's box -> bilateral -> [gray-world, optional]
-> CLAHE -> HSV/LAB -> segmentation -> morphology -> features -> quality
checks -> rules -> grade (+ margin) .
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("unripe", "ripe", "overripe")
GRADE_NAMES = {0: "Unripe", 1: "Ripe", 2: "Overripe"}
SEG_MODES = ("grabcut", "hsv", "kmeans")
COLOUR_BGR = {                      # colours used by the colour map (and the web legend)
    "dark": (112, 44, 92), "orange": (40, 140, 255), "red": (50, 50, 200), "other": (176, 176, 176),
}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Image-processing settings. OpenCV HSV: H 0-179, S and V 0-255."""
    width: int = 640
    bilateral: bool = True
    bilateral_d: int = 7
    bilateral_sigma: float = 40.0
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_colour: bool = False         # measure colour on the CLAHE image (default: before CLAHE)
    white_balance: bool = False        # gray-world; off by default (see README)
    glare_filter: bool = True
    gates: bool = True                 # quality checks before grading
    seg_mode: str = "grabcut"
    grabcut_iter: int = 4
    green_h: tuple = (33, 90)
    green_s_min: int = 40
    sat_min: int = 50                  # hsv mode: saturation of "fruit-like" pixels
    morph_k: int = 5
    canny_low: int = 60
    canny_high: int = 160
    # colour classes for pixels inside the bunch
    dark_v: int = 70                   # V below this = dark (black / violet fruit)
    orange_h: tuple = (7, 26)
    red_h_lo: int = 6                  # red = H <= this ...
    red_h_hi: int = 168                # ... or H >= this
    color_s_min: int = 90
    # pixels ignored when measuring colour
    shadow_v: int = 25
    glare_v: int = 235
    glare_s: int = 40
    # quality checks
    min_area_ratio: float = 0.10
    min_valid_ratio: float = 0.40
    blur_min: float = 15.0
    mean_v_min: float = 35.0
    mean_v_max: float = 240.0
    max_green_ratio: float = 0.30
    min_fruit_like: float = 0.30


@dataclass
class Thresholds:
    """Grading rules. Calibrate on labelled images; the defaults are placeholders."""
    t_dark: float = 0.40
    t_orange: float = 0.30
    t_over: float = 0.50
    w_red: float = 0.50
    w_hole: float = 0.25
    w_edge: float = 0.25
    s_red: float = 0.60
    s_hole: float = 0.04
    s_edge: float = 0.15
    margin_min: float = 0.10
    note: str = "demo defaults, not validated on real fruit"


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    kw = {k: v for k, v in (d or {}).items() if k in names}
    for f in fields(cls):                     # JSON lists -> tuples
        if f.name in kw and isinstance(getattr(cls(), f.name), tuple):
            kw[f.name] = tuple(kw[f.name])
    return cls(**kw)


def params_from_dict(d: dict) -> Params:
    return _from_dict(Params, d)


def load_thresholds(path: Optional[str] = None) -> Thresholds:
    path = path or os.path.join(HERE, "thresholds.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return _from_dict(Thresholds, json.load(fh))
    return Thresholds()


def save_thresholds(t: Thresholds, path: Optional[str] = None) -> str:
    path = path or os.path.join(HERE, "thresholds.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(t), fh, indent=2)
    return path


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    s = width / w
    return cv2.resize(img, (width, max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def box_to_pixels(box, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
    y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
    px = [round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)]
    if px[2] - px[0] < 24 or px[3] - px[1] < 24:
        raise ValueError("The box is too small. Draw it around the whole bunch.")
    return tuple(px)


def odd(k: int, minimum: int = 3) -> int:
    k = max(minimum, int(k))
    return k if k % 2 else k + 1


# --------------------------------------------------------------------------
# Techniques 1-3: filtering, enhancement, colour spaces
# --------------------------------------------------------------------------
def gray_world(bgr: np.ndarray, ref: np.ndarray, limit=(0.75, 1.33)) -> np.ndarray:
    """Scale channels so the average colour of `ref` becomes neutral.

    Risky when one colour dominates the scene (an orange bunch fills the frame),
    which is why it is off by default; compare with and without in your ablation.
    """
    means = ref.reshape(-1, 3).astype(np.float32).mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-3), *limit)
    return np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def clahe_lab(bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile))).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def preprocess(crop: np.ndarray, ref: np.ndarray, p: Params):
    """Returns (filtered, enhanced). Filtering first so CLAHE does not amplify noise."""
    filtered = cv2.bilateralFilter(crop, p.bilateral_d, p.bilateral_sigma, p.bilateral_sigma) if p.bilateral else crop.copy()
    if p.white_balance:
        filtered = gray_world(filtered, ref)
    enhanced = clahe_lab(filtered, p.clahe_clip, p.clahe_tile) if p.clahe else filtered.copy()
    return filtered, enhanced


def green_mask(hsv: np.ndarray, p: Params) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    g = (h >= p.green_h[0]) & (h <= p.green_h[1]) & (s >= p.green_s_min) & (v >= 30)
    return g.astype(np.uint8) * 255


def colour_classes(hsv: np.ndarray, p: Params):
    """Boolean maps (dark, orange, red). Mutually exclusive, tested in that order."""
    h, s, v = cv2.split(hsv)
    dark = v < p.dark_v
    sat = (s >= p.color_s_min) & ~dark
    orange = sat & (h >= p.orange_h[0]) & (h <= p.orange_h[1])
    red = sat & ((h <= p.red_h_lo) | (h >= p.red_h_hi))
    return dark, orange, red


# --------------------------------------------------------------------------
# Technique 4-5: segmentation and morphology
# --------------------------------------------------------------------------
def segment_grabcut(img: np.ndarray, green: np.ndarray, p: Params) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    s = 320.0 / max(h, w) if max(h, w) > 320 else 1.0
    small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    m = max(2, int(0.04 * min(sh, sw)))
    mask = np.zeros((sh, sw), np.uint8)
    try:
        cv2.grabCut(small, mask, (m, m, sw - 2 * m, sh - 2 * m), np.zeros((1, 65)), np.zeros((1, 65)),
                    int(p.grabcut_iter), cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    fg[green > 0] = 0
    return fg


def segment_hsv(hsv: np.ndarray, green: np.ndarray, p: Params) -> np.ndarray:
    """Keep non-green pixels that are dark or colourful."""
    _, s, v = cv2.split(hsv)
    return (((v < p.dark_v) | (s >= p.sat_min)) & (green == 0)).astype(np.uint8) * 255


def segment_kmeans(enhanced: np.ndarray, green: np.ndarray, p: Params) -> np.ndarray:
    """Experimental: 3 colour clusters in LAB; keep the one that fills the centre of the box."""
    h, w = enhanced.shape[:2]
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    ok = green == 0
    pts = lab[ok]
    if len(pts) < 200:
        return np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(0)
    sample = pts[rng.choice(len(pts), size=min(len(pts), 15000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centres = cv2.kmeans(sample, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    dist = ((lab[:, :, None, :] - centres[None, None, :, :]) ** 2).sum(axis=3)
    lbl = dist.argmin(axis=2)
    lbl[~ok] = -1
    centre = lbl[int(h * .25):int(h * .75), int(w * .25):int(w * .75)]
    keep = int(np.argmax([(centre == k).sum() for k in range(3)]))
    return (lbl == keep).astype(np.uint8) * 255


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == i) * 255).astype(np.uint8)


def refine_mask(mask: np.ndarray, p: Params) -> np.ndarray:
    k = odd(p.morph_k)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return largest_component(m)


# --------------------------------------------------------------------------
# Techniques 6-7: edges, contours and features
# --------------------------------------------------------------------------
FEATURE_KEYS = ("dark_ratio", "orange_ratio", "red_ratio", "hole_ratio", "edge_density")


def _empty_features() -> dict:
    keys = ("area_ratio", "valid_ratio", "sharpness", "mean_v", "green_ratio", "fruit_like", "dark_ratio",
            "orange_ratio", "red_ratio", "other_ratio", "mean_a", "mean_b", "mean_s", "edge_density",
            "hole_ratio", "solidity")
    return {k: 0.0 for k in keys}


def extract_features(enhanced, crop_gray, mask, edges, green, p: Params) -> dict:
    f = _empty_features()
    area = cv2.countNonZero(mask)
    f["area_ratio"] = area / mask.size
    if area == 0:
        return f
    # outline: holes (empty sockets, gaps), solidity, filled region
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer = holes = hull = 0.0
    filled = np.zeros_like(mask)
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if hier[0][i][3] == -1:
            outer += a
            hull += cv2.contourArea(cv2.convexHull(c))
            cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
        elif a >= 6:
            holes += a
    f["hole_ratio"] = holes / max(outer, 1.0)
    f["solidity"] = outer / max(hull, 1.0)
    f["green_ratio"] = float(cv2.countNonZero(cv2.bitwise_and(green, filled))) / max(cv2.countNonZero(filled), 1)
    # colour statistics on pixels away from the outline, ignoring deep shadow and glare
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    sel = (inner if cv2.countNonZero(inner) else mask) > 0
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, s, v = cv2.split(hsv)
    valid = sel & (v >= p.shadow_v)
    if p.glare_filter:
        valid &= ~((v >= p.glare_v) & (s <= p.glare_s))
    n_sel, n_valid = int(sel.sum()), int(valid.sum())
    f["valid_ratio"] = n_valid / max(n_sel, 1)
    f["sharpness"] = float(cv2.Laplacian(crop_gray, cv2.CV_64F)[sel].var())
    f["edge_density"] = float((edges[sel] > 0).mean())
    if n_valid:
        dark, orange, red = (m[valid].mean() for m in colour_classes(hsv, p))
        f.update(dark_ratio=float(dark), orange_ratio=float(orange), red_ratio=float(red),
                 fruit_like=float(dark + orange + red), other_ratio=float(max(0.0, 1 - dark - orange - red)),
                 mean_a=float(lab[..., 1][valid].mean() - 128), mean_b=float(lab[..., 2][valid].mean() - 128),
                 mean_s=float(s[valid].mean()), mean_v=float(v[valid].mean()))
    return f


def quality_gates(f: dict, p: Params) -> list:
    def g(name, label, value, limit, ok, fix):
        return {"name": name, "label": label, "value": round(float(value), 4), "limit": limit, "ok": bool(ok), "fix": fix}
    return [
        g("area", "Bunch fills enough of the box", f["area_ratio"], f">= {p.min_area_ratio:.0%}",
          f["area_ratio"] >= p.min_area_ratio, "Drag the box closer around the bunch."),
        g("valid", "Enough evenly lit pixels", f["valid_ratio"], f">= {p.min_valid_ratio:.0%}",
          f["valid_ratio"] >= p.min_valid_ratio, "Avoid deep shadow and glare. Shoot in even light."),
        g("sharp", "Photo is sharp", f["sharpness"], f">= {p.blur_min:g}",
          f["sharpness"] >= p.blur_min, "Hold the phone steady and tap the bunch to focus."),
        g("light", "Exposure is workable", f["mean_v"], f"{p.mean_v_min:g} to {p.mean_v_max:g}",
          p.mean_v_min <= f["mean_v"] <= p.mean_v_max, "The photo is too dark or too bright. Retake it."),
        g("green", "Little foliage inside the outline", f["green_ratio"], f"<= {p.max_green_ratio:.0%}",
          f["green_ratio"] <= p.max_green_ratio, "Leaves are inside the outline. Draw the box tighter."),
        g("fruit", "Colours look like palm fruit", f["fruit_like"], f">= {p.min_fruit_like:.0%}",
          f["fruit_like"] >= p.min_fruit_like, "Check that the box is on a fruit bunch."),
    ]


# --------------------------------------------------------------------------
# Grading rules
# --------------------------------------------------------------------------
def over_score(red, hole, edge, t: Thresholds):
    """0-1 score of over-ripe cues: deep red fruit, gaps (empty sockets), rough surface."""
    ws = (t.w_red + t.w_hole + t.w_edge) or 1.0
    return (t.w_red * np.clip(np.asarray(red, float) / t.s_red, 0, 1)
            + t.w_hole * np.clip(np.asarray(hole, float) / t.s_hole, 0, 1)
            + t.w_edge * np.clip(np.asarray(edge, float) / t.s_edge, 0, 1)) / ws


def grade_arrays(dark, orange, red, hole, edge, t: Thresholds):
    """Vectorised rules. Returns (code, margin, over): code 0/1/2 or -1 when no rule fires."""
    dark, orange = np.asarray(dark, float), np.asarray(orange, float)
    over = over_score(red, hole, edge, t)
    e = 1e-6
    is_un = dark >= t.t_dark
    is_ov = ~is_un & (over >= t.t_over)
    is_rp = ~is_un & ~is_ov & (orange >= t.t_orange)
    code = np.full(dark.shape, -1)
    code[is_un], code[is_rp], code[is_ov] = 0, 1, 2
    not_dark = (t.t_dark - dark) / max(t.t_dark, e)
    m_un = (dark - t.t_dark) / max(t.t_dark, e)
    m_ov = np.minimum((over - t.t_over) / max(t.t_over, e), not_dark)
    m_rp = np.minimum.reduce([(orange - t.t_orange) / max(t.t_orange, e), (t.t_over - over) / max(t.t_over, e), not_dark])
    margin = np.where(is_un, m_un, np.where(is_ov, m_ov, np.where(is_rp, m_rp, 0.0)))
    return code, margin, over


ADVICE = {
    "Unripe": "Leave this bunch for now. Harvesting too early gives less oil.",
    "Ripe": "Ready to harvest.",
    "Overripe": "Harvest soon. Waiting longer lowers fruit quality.",
    "Uncertain": "The colours sit close to a boundary. Add another view of the bunch, or check it by eye.",
    "Not gradable": "This photo cannot be graded. Use the fixes listed under photo checks.",
}


def decide(f: dict, t: Thresholds, gates_ok: bool = True) -> dict:
    code, margin, over = grade_arrays(f["dark_ratio"], f["orange_ratio"], f["red_ratio"], f["hole_ratio"],
                                      f["edge_density"], t)
    code, margin, over = int(np.atleast_1d(code)[0]), float(np.atleast_1d(margin)[0]), float(np.atleast_1d(over)[0])
    leaning = GRADE_NAMES.get(code)
    if not gates_ok:
        grade, leaning = "Not gradable", None
    elif code < 0 or margin < t.margin_min:
        grade = "Uncertain"
    else:
        grade = GRADE_NAMES[code]
    return {"grade": grade, "leaning": leaning, "code": code, "margin": margin, "over": over}


def explain_rules(f: dict, t: Thresholds, d: dict) -> list:
    return [
        {"id": "unripe", "label": "Dark share", "value": f["dark_ratio"], "threshold": t.t_dark,
         "met": f["dark_ratio"] >= t.t_dark, "fired": d["code"] == 0,
         "note": "Unripe fruit is dark violet or black."},
        {"id": "overripe", "label": "Over-ripe score", "value": d["over"], "threshold": t.t_over,
         "met": d["over"] >= t.t_over, "fired": d["code"] == 2,
         "note": f"From deep red {f['red_ratio']:.0%}, gaps {f['hole_ratio']:.1%} and rough surface {f['edge_density']:.0%}."},
        {"id": "ripe", "label": "Orange share", "value": f["orange_ratio"], "threshold": t.t_orange,
         "met": f["orange_ratio"] >= t.t_orange, "fired": d["code"] == 1,
         "note": "Ripe fruit turns orange. Checked only if the two rules above are not met."},
    ]


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
@dataclass
class Analysis:
    grade: str
    leaning: Optional[str]
    code: int
    margin: float
    over: float
    advice: str
    features: dict
    composition: dict
    gates: list
    rules: list
    thresholds: dict
    seg_mode: str
    ms: float
    gates_ok: bool = True
    images: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "images"}
        d["fixes"] = [g["fix"] for g in self.gates if not g["ok"]]
        return d


def colour_map(crop, enhanced, mask, p: Params):
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    dark, orange, red = colour_classes(hsv, p)
    out = (crop * 0.30).astype(np.uint8)
    sel = mask > 0
    other = sel & ~(dark | orange | red)
    for name, m in (("other", other), ("dark", sel & dark), ("orange", sel & orange), ("red", sel & red)):
        out[m] = COLOUR_BGR[name]
    return out


def outline_overlay(crop, mask):
    out = crop.copy()
    tint = np.zeros_like(out)
    tint[:] = (80, 220, 120)
    blend = cv2.addWeighted(out, 0.82, tint, 0.18, 0)
    out[mask > 0] = blend[mask > 0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def analyse(img_bgr, box=(0, 0, 1, 1), params: Optional[Params] = None, thr: Optional[Thresholds] = None,
            want_images: bool = True) -> Analysis:
    t0 = time.perf_counter()
    p, t = params or Params(), thr or load_thresholds()
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("The image could not be read as a colour photo.")
    img = resize_to_width(img_bgr, p.width)
    x0, y0, x1, y1 = box_to_pixels(box, img.shape)
    crop = img[y0:y1, x0:x1].copy()

    filtered, enhanced = preprocess(crop, img, p)
    # CLAHE equalises each tile, so it changes absolute brightness. Colour is measured on the
    # filtered image by default; CLAHE is used for edges (and for colour only if clahe_colour).
    colour_img = enhanced if p.clahe_colour else filtered
    hsv = cv2.cvtColor(colour_img, cv2.COLOR_BGR2HSV)
    green = green_mask(hsv, p)
    cv2.setRNGSeed(12345)                              # GrabCut and k-means use OpenCV's RNG: keep runs repeatable

    raw = None
    if p.seg_mode == "grabcut":
        raw = segment_grabcut(filtered, green, p)
    if raw is None and p.seg_mode == "kmeans":
        raw = segment_kmeans(colour_img, green, p)
    if raw is None:                                   # "hsv", or fallback when GrabCut fails
        raw = segment_hsv(hsv, green, p)
    mask = refine_mask(raw, p)

    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_enh, p.canny_low, max(p.canny_high, p.canny_low + 1))
    f = extract_features(colour_img, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), mask, edges, green, p)
    gates = quality_gates(f, p) if p.gates else []
    gates_ok = all(g["ok"] for g in gates)
    d = decide(f, t, gates_ok)
    comp = {k: f[k + "_ratio"] for k in ("dark", "orange", "red", "other")}

    res = Analysis(grade=d["grade"], leaning=d["leaning"] if d["grade"] == "Uncertain" else None, code=d["code"],
                   margin=d["margin"], over=d["over"], advice=ADVICE[d["grade"]], features=f, composition=comp,
                   gates=gates, rules=explain_rules(f, t, d), thresholds=asdict(t), seg_mode=p.seg_mode,
                   ms=0.0, gates_ok=gates_ok)
    if want_images:
        res.images = {
            "crop": crop, "filtered": filtered, "enhanced": enhanced, "mask": mask, "edges": edges,
            "overlay": outline_overlay(crop, mask), "colormap": colour_map(crop, colour_img, mask, p),
        }
    res.ms = round((time.perf_counter() - t0) * 1000, 1)
    return res


# --------------------------------------------------------------------------
# Several photos of one bunch
# --------------------------------------------------------------------------
def fuse_views(views: list, thr: Optional[Thresholds] = None) -> dict:
    """views: [{'features': {...}, 'gates_ok': bool}, ...]. Average the features, then grade."""
    t = thr or load_thresholds()
    use = [v["features"] for v in views if v.get("gates_ok", True)]
    if not use:
        return {"grade": "Not gradable", "leaning": None, "margin": 0.0, "advice": ADVICE["Not gradable"],
                "used": 0, "votes": {}}
    avg = {k: float(np.mean([u[k] for u in use])) for k in FEATURE_KEYS}
    d = decide(avg, t, True)
    votes = {}
    for u in use:
        g = decide(u, t, True)["grade"]
        votes[g] = votes.get(g, 0) + 1
    return {"grade": d["grade"], "leaning": d["leaning"] if d["grade"] == "Uncertain" else None,
            "margin": d["margin"], "advice": ADVICE[d["grade"]], "used": len(use), "votes": votes,
            "features": avg, "over": d["over"]}
