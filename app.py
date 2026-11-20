"""
app.py - RipeCheck web app (Flask).

Run:   python app.py        then open http://127.0.0.1:5057
All computer vision is in cv_pipeline.py. This file only moves photos and
results between the browser and that module.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import webbrowser
from dataclasses import asdict

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from cv_pipeline import (CLASSES, Params, Thresholds, analyse, fuse_views, load_thresholds, params_from_dict,
                         SEG_MODES, COLOUR_BGR)
from metrics import summarise

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "sample_images")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024


# ---------------------------------------------------------------- helpers
def decode(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def data_url(img, ext=".jpg", quality=84):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    return "data:image/%s;base64,%s" % ("jpeg" if ext == ".jpg" else "png", base64.b64encode(buf).decode())


def parse_box(text):
    try:
        x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in (text or "0,0,1,1").split(","))
    except ValueError:
        return 0.0, 0.0, 1.0, 1.0
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def parse_params(text):
    try:
        d = json.loads(text) if text else {}
    except json.JSONDecodeError:
        d = {}
    allowed = {"seg_mode", "bilateral", "clahe", "clahe_colour", "white_balance", "glare_filter", "gates"}
    d = {k: v for k, v in d.items() if k in allowed}
    if d.get("seg_mode") not in SEG_MODES:
        d.pop("seg_mode", None)
    return params_from_dict(d)


def parse_thresholds(text):
    base = asdict(load_thresholds())
    try:
        over = json.loads(text) if text else {}
    except json.JSONDecodeError:
        over = {}
    for k in ("t_dark", "t_orange", "t_over", "margin_min"):
        if k in over:
            try:
                base[k] = min(max(float(over[k]), 0.0), 1.5)
            except (TypeError, ValueError):
                pass
    return Thresholds(**base)


# ---------------------------------------------------------------- pages and files
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/samples/<path:name>")
def sample_file(name):
    if not name.lower().endswith(IMAGE_EXT):
        abort(404)
    return send_from_directory(SAMPLE_DIR, name)


@app.get("/api/samples")
def samples():
    names = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(IMAGE_EXT)) if os.path.isdir(SAMPLE_DIR) else []
    return jsonify(names)


@app.get("/api/config")
def config():
    t = load_thresholds()
    return jsonify({"thresholds": asdict(t), "params": {k: v for k, v in asdict(Params()).items()
                    if k in ("seg_mode", "bilateral", "clahe", "clahe_colour", "white_balance", "glare_filter", "gates")},
                    "seg_modes": list(SEG_MODES),
                    "legend": {k: "rgb(%d,%d,%d)" % (v[2], v[1], v[0]) for k, v in COLOUR_BGR.items()}})


# ---------------------------------------------------------------- analysis
@app.post("/api/analyse")
def api_analyse():
    f = request.files.get("image")
    if f is None:
        return jsonify(ok=False, error="No photo was sent."), 400
    img = decode(f)
    if img is None:
        return jsonify(ok=False, error="That file could not be read as an image. Use a JPG or PNG photo."), 400
    try:
        res = analyse(img, parse_box(request.form.get("box")), parse_params(request.form.get("opts")),
                      parse_thresholds(request.form.get("thr")))
    except ValueError as err:
        return jsonify(ok=False, error=str(err)), 422
    out = res.to_json()
    out["ok"] = True
    out["images"] = {k: data_url(v, ".png" if k in ("mask", "edges") else ".jpg") for k, v in res.images.items()}
    return jsonify(out)


@app.post("/api/fuse")
def api_fuse():
    body = request.get_json(silent=True) or {}
    views = [v for v in body.get("views", []) if isinstance(v, dict) and isinstance(v.get("features"), dict)]
    if not views:
        return jsonify(ok=False, error="Add at least one view first."), 400
    try:
        out = fuse_views(views, parse_thresholds(json.dumps(body.get("thr") or {})))
    except KeyError:
        return jsonify(ok=False, error="A view is missing its features."), 400
    out["ok"] = True
    return jsonify(out)


@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify(ok=False, error="Choose some photos first."), 400
    p, t = parse_params(request.form.get("opts")), parse_thresholds(request.form.get("thr"))
    rows = []
    for f in files[:200]:
        name = os.path.basename(f.filename or "photo")
        low = name.lower()
        label = next((c for c in CLASSES if low.startswith(c + "_") or low.startswith(c + "-")), None)
        img = decode(f)
        if img is None:
            rows.append({"file": name, "label": label, "result": "not gradable", "margin": None, "note": "unreadable"})
            continue
        try:
            r = analyse(img, (0, 0, 1, 1), p, t, want_images=False)
            rows.append({"file": name, "label": label, "result": r.grade.lower(), "margin": round(r.margin, 2),
                         "note": ", ".join(g["name"] for g in r.gates if not g["ok"])})
        except ValueError as err:
            rows.append({"file": name, "label": label, "result": "not gradable", "margin": None, "note": str(err)})
    scored = [r for r in rows if r["label"]]
    summary = summarise([r["label"] for r in scored], [r["result"] for r in scored]) if scored else None
    return jsonify(ok=True, rows=rows, summary=summary, unscored=len(rows) - len(scored))


def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5057"))
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, open_browser, args=(port,)).start()
    print(f"RipeCheck is running at http://127.0.0.1:{port}  (press Ctrl+C to stop)")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False, threaded=True)
