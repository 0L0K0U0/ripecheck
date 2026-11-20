"""
evaluate.py - score the grader on a labelled TEST folder.

    python evaluate.py data/test
    python evaluate.py data/test --fuse-views               # one grade per bunch (needs labels.csv)
    python evaluate.py data/test --ablation --train data/train
    python evaluate.py data/test --tree data/train          # compare with a small decision tree

Reports per-class precision / recall / F1, accuracy, the confusion matrix, and
coverage (the share of photos that received a grade). "Uncertain" and "not
gradable" count as misses, so abstaining never improves the headline numbers.
Run it on the test set once, after thresholds are frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

import numpy as np

from calibrate import LABEL_CODE, search_thresholds, stack
from cv_pipeline import FEATURE_KEYS, Params, Thresholds, decide, load_thresholds
from dataset import featurize, load_items
from metrics import LABELS, OUTCOMES, summarise


def predict(rows, thr: Thresholds):
    out = []
    for r in rows:
        if not r.get("features"):
            out.append("not gradable")
            continue
        out.append(decide(r["features"], thr, r["ok"])["grade"].lower())
    return out


def fmt(v, pct=False):
    return "n/a" if v is None else (f"{v:.1%}" if pct else f"{v:.3f}")


def print_report(s, title):
    print(f"\n== {title} ==")
    print(f"photos: {s['n']}   macro F1: {fmt(s['macro_f1'])}   accuracy (all): {fmt(s['accuracy_all'], True)}   "
          f"accuracy (graded only): {fmt(s['accuracy_graded'], True)}   coverage: {fmt(s['coverage'], True)}")
    print(f"{'class':10s} {'precision':>10s} {'recall':>8s} {'F1':>7s} {'n':>5s}")
    for c in LABELS:
        m = s["per_class"][c]
        print(f"{c:10s} {fmt(m['precision']):>10s} {fmt(m['recall']):>8s} {fmt(m['f1']):>7s} {m['support']:>5d}")
    print("confusion (rows = true, columns = result)")
    print(f"{'':10s}" + "".join(f"{o:>14s}" for o in OUTCOMES))
    for t in LABELS:
        print(f"{t:10s}" + "".join(f"{s['confusion'][t][o]:>14d}" for o in OUTCOMES))


def fuse_by_bunch(rows, thr):
    from cv_pipeline import fuse_views
    groups = {}
    for r in rows:
        groups.setdefault(r["bunch"], []).append(r)
    y_true, y_pred = [], []
    for g in groups.values():
        views = [{"features": r["features"], "gates_ok": r["ok"]} for r in g if r.get("features")]
        y_true.append(g[0]["label"])
        y_pred.append(fuse_views(views, thr)["grade"].lower() if views else "not gradable")
    return y_true, y_pred


VARIANTS = [
    ("default pipeline", {}),
    ("no bilateral filter", {"bilateral": False}),
    ("no CLAHE", {"clahe": False}),
    ("colour measured on CLAHE image", {"clahe_colour": True}),
    ("gray-world white balance", {"white_balance": True}),
    ("no glare/shadow filter", {"glare_filter": False}),
    ("HSV colour segmentation", {"seg_mode": "hsv"}),
    ("k-means segmentation", {"seg_mode": "kmeans"}),
    ("no quality checks", {"gates": False}),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_dir")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fuse-views", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--train", default=None, help="train folder: re-calibrate every ablation variant on it")
    ap.add_argument("--tree", default=None, help="train folder for the optional decision-tree comparison")
    a = ap.parse_args(argv)

    thr = load_thresholds(a.thresholds)
    items, have_csv = load_items(a.test_dir)
    os.makedirs(a.out, exist_ok=True)
    print("Thresholds:", thr.note)

    rows = featurize(items, Params())
    y_true, y_pred = [r["label"] for r in rows], predict(rows, thr)
    s = summarise(y_true, y_pred)
    print_report(s, "per photo")
    with open(os.path.join(a.out, "predictions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "bunch", "true", "result", "quality_checks_failed"])
        for r, p in zip(rows, y_pred):
            w.writerow([r["rel"], r["bunch"], r["label"], p, "/".join(r["failed"])])
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=2)

    print("\n== accuracy against coverage (margin sweep) ==")
    print(f"{'margin':>7s} {'coverage':>9s} {'acc (graded)':>13s} {'acc (all)':>10s}")
    for m in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        sm = summarise(y_true, predict(rows, replace(thr, margin_min=m)))
        print(f"{m:>7.2f} {fmt(sm['coverage'], True):>9s} {fmt(sm['accuracy_graded'], True):>13s} {fmt(sm['accuracy_all'], True):>10s}")

    if a.fuse_views:
        if not have_csv:
            print("\n--fuse-views needs labels.csv with a bunch column. Skipped.")
        else:
            yt, yp = fuse_by_bunch(rows, thr)
            print_report(summarise(yt, yp), f"per bunch (views fused, {len(yt)} bunches)")

    if a.ablation:
        print("\n== ablation ==")
        print("Thresholds are " + ("re-calibrated on the train folder for each variant." if a.train else
              "FIXED from the default pipeline, which can unfairly penalise variants that shift the features. "
              "Pass --train to re-calibrate each one."))
        train_items = load_items(a.train)[0] if a.train else None
        print(f"{'variant':34s} {'macro F1':>9s} {'acc (all)':>10s} {'acc (graded)':>13s} {'coverage':>9s}")
        for name, change in VARIANTS:
            p = replace(Params(), **change)
            rr = featurize(items, p, quiet=True)
            t = thr
            if train_items:
                tr = [r for r in featurize(train_items, p, quiet=True) if r["ok"] and r["features"]]
                X, y = stack(tr)
                t, _ = search_thresholds(X, y, thr)
            sv = summarise([r["label"] for r in rr], predict(rr, t))
            print(f"{name:34s} {fmt(sv['macro_f1']):>9s} {fmt(sv['accuracy_all'], True):>10s} "
                  f"{fmt(sv['accuracy_graded'], True):>13s} {fmt(sv['coverage'], True):>9s}")

    if a.tree:
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            print("\n--tree needs scikit-learn (pip install scikit-learn). Skipped.")
        else:
            tr = [r for r in featurize(load_items(a.tree)[0], Params(), quiet=True) if r["ok"] and r["features"]]
            Xtr = np.array([[r["features"][k] for k in FEATURE_KEYS] for r in tr])
            ytr = np.array([LABEL_CODE[r["label"]] for r in tr])
            clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
            pred = []
            for r in rows:
                if not r["ok"] or not r.get("features"):
                    pred.append("not gradable")
                else:
                    pred.append(LABELS[int(clf.predict([[r["features"][k] for k in FEATURE_KEYS]])[0])])
            print_report(summarise(y_true, pred), "decision tree (depth 3) on the same features")
    print(f"\nWrote {a.out}/predictions.csv and {a.out}/summary.json")


if __name__ == "__main__":
    main()
