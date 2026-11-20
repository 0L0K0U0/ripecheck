"""
dataset.py - load labelled images and compute their features (with a cache).

Folder layout (one folder per class, names must match):
    data/train/unripe/*.jpg    data/train/ripe/*.jpg    data/train/overripe/*.jpg
    data/test/unripe/*.jpg     ...
Optional data/train/labels.csv (and data/test/labels.csv) with columns
    file,label,bunch          e.g.   ripe/img_001.jpg,ripe,tree12_bunch3
`bunch` tells the scripts which photos show the same bunch, so train and
validation folds never share a bunch. Without it every photo is treated as its
own bunch, which can leak and inflate results if one bunch has several photos.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict

import cv2

from cv_pipeline import CLASSES, Params, analyse

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_items(root: str):
    root = os.path.abspath(root)
    meta, have_csv = {}, False
    csv_path = os.path.join(root, "labels.csv")
    if os.path.exists(csv_path):
        have_csv = True
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                meta[r["file"].replace("\\", "/")] = (r.get("label") or "", r.get("bunch") or "")
    items = []
    for cls in CLASSES:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(EXTS):
                rel = f"{cls}/{fn}"
                label, bunch = meta.get(rel, ("", ""))
                items.append({"path": os.path.join(d, fn), "rel": rel, "label": label or cls, "bunch": bunch or rel})
    if not items:
        sys.exit(f"No images found under {root}. Expected {root}/unripe, {root}/ripe and {root}/overripe.")
    if not have_csv:
        print(f"Note: {csv_path} not found. Each photo is treated as its own bunch; if several photos show the "
              "same bunch, results may be inflated.")
    return items, have_csv


def featurize(items, params: Params, cache_dir: str | None = None, box=(0, 0, 1, 1), thr=None, quiet=False):
    """Features for every item. Cached on disk, keyed by the settings and the files."""
    cache_dir = cache_dir or os.path.join(HERE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    sig = json.dumps({"p": asdict(params), "box": list(box),
                      "f": [(i["rel"], os.path.getmtime(i["path"]), os.path.getsize(i["path"])) for i in items]},
                     sort_keys=True, default=str)
    path = os.path.join(cache_dir, "features_" + hashlib.md5(sig.encode()).hexdigest()[:12] + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        return [{**it, **r} for it, r in zip(items, rows)]
    rows = []
    for n, it in enumerate(items, 1):
        img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"ok": False, "failed": ["unreadable"], "features": None})
            continue
        try:
            a = analyse(img, box, params, thr, want_images=False)
            rows.append({"ok": a.gates_ok, "failed": [g["name"] for g in a.gates if not g["ok"]],
                         "features": a.features})
        except ValueError:
            rows.append({"ok": False, "failed": ["box"], "features": None})
        if not quiet and n % 20 == 0:
            print(f"  features {n}/{len(items)}", flush=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return [{**it, **r} for it, r in zip(items, rows)]
