"""
calibrate.py - search grading thresholds on a labelled TRAINING folder.

    python calibrate.py data/train
    python calibrate.py data/train --folds 5 --out thresholds.json

What it does
  1. Computes features for every training photo that passes the quality checks.
  2. Grid-searches the rule thresholds (and the weights of the over-ripe score)
     to maximise the macro F1 over unripe / ripe / overripe.
  3. Estimates how well that search generalises with grouped k-fold
     cross-validation: whole bunches are held out, never single photos.
  4. Writes thresholds.json.
Keep your TEST folder out of this step. Evaluate on it once, with evaluate.py.
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from cv_pipeline import Params, Thresholds, grade_arrays, load_thresholds, save_thresholds
from dataset import featurize, load_items

LABEL_CODE = {"unripe": 0, "ripe": 1, "overripe": 2}
WEIGHT_SETS = ((1, 0, 0), (0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0), (0.5, 0, 0.5))


def fast_macro_f1(y: np.ndarray, code: np.ndarray) -> float:
    """Macro F1 where code -1 (no rule fired) counts as a miss."""
    f1s = []
    for c in (0, 1, 2):
        tp = np.sum((y == c) & (code == c))
        fp = np.sum((y != c) & (code == c))
        fn = np.sum((y == c) & (code != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def stack(rows):
    keys = ("dark_ratio", "orange_ratio", "red_ratio", "hole_ratio", "edge_density")
    X = {k: np.array([r["features"][k] for r in rows], float) for k in keys}
    y = np.array([LABEL_CODE[r["label"]] for r in rows])
    return X, y


def scales_from(X) -> dict:
    return {"s_red": max(0.05, float(np.percentile(X["red_ratio"], 90))),
            "s_hole": max(0.005, float(np.percentile(X["hole_ratio"], 90))),
            "s_edge": max(0.02, float(np.percentile(X["edge_density"], 90)))}


def search_thresholds(X, y, base: Thresholds | None = None) -> tuple[Thresholds, float]:
    """Best thresholds on (X, y). Ties are broken by the average margin of correct grades."""
    base = base or Thresholds()
    sc = scales_from(X)
    best, best_key = None, (-1.0, -1.0)
    grid_d = np.round(np.arange(0.10, 0.81, 0.05), 2)
    grid_o = np.round(np.arange(0.10, 0.81, 0.05), 2)
    grid_v = np.round(np.arange(0.20, 0.91, 0.05), 2)
    for w in WEIGHT_SETS:
        for td, to, tv in itertools.product(grid_d, grid_o, grid_v):
            t = Thresholds(t_dark=float(td), t_orange=float(to), t_over=float(tv), w_red=w[0], w_hole=w[1],
                           w_edge=w[2], margin_min=0.0, **sc)
            code, margin, _ = grade_arrays(X["dark_ratio"], X["orange_ratio"], X["red_ratio"], X["hole_ratio"],
                                           X["edge_density"], t)
            key = (fast_macro_f1(y, code), float(np.clip(margin[code == y], 0, 1).mean()) if np.any(code == y) else 0.0)
            if key > best_key:
                best, best_key = t, key
    best.margin_min = base.margin_min
    return best, best_key[0]


def grouped_folds(groups, k, seed=0):
    ug = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_dir")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None, help="where to write thresholds (default: ./thresholds.json)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)

    items, _ = load_items(a.train_dir)
    rows = featurize(items, Params())
    use = [r for r in rows if r["ok"] and r["features"]]
    print(f"{len(use)} of {len(rows)} photos passed the quality checks and are used.")
    skipped = [r for r in rows if not (r["ok"] and r["features"])]
    if skipped:
        print("Skipped:", ", ".join(f"{r['rel']} ({'/'.join(r['failed'])})" for r in skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    X, y = stack(use)
    groups = np.array([r["bunch"] for r in use])
    counts = {c: int((y == i).sum()) for c, i in LABEL_CODE.items()}
    print("Photos per class:", counts, "| bunches:", len(set(groups)))
    if min(counts.values()) < 10:
        print("Warning: fewer than 10 photos in a class. The thresholds will be unreliable.")

    k = min(a.folds, len(set(groups)))
    cv_scores = []
    if k >= 3:
        fold = grouped_folds(groups, k)
        for i in range(k):
            tr, va = fold != i, fold == i
            if len(set(y[tr])) < 3 or not va.any():
                continue
            t, _ = search_thresholds({kk: v[tr] for kk, v in X.items()}, y[tr])
            code, _, _ = grade_arrays(*(X[kk][va] for kk in ("dark_ratio", "orange_ratio", "red_ratio", "hole_ratio", "edge_density")), t)
            cv_scores.append(fast_macro_f1(y[va], code))
        if cv_scores:
            print(f"Grouped {k}-fold CV macro F1: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f} (folds: {', '.join(f'{s:.2f}' for s in cv_scores)})")
    else:
        print("Too few bunches for cross-validation. Add labels.csv with bunch ids.")

    best, f1 = search_thresholds(X, y)
    cv_txt = f"grouped CV macro F1 {np.mean(cv_scores):.2f}" if cv_scores else "no CV"
    best.note = f"calibrated on {len(use)} photos ({len(set(groups))} bunches), train macro F1 {f1:.2f}, {cv_txt}"
    print(f"Best thresholds (training macro F1 {f1:.3f}):")
    for kk in ("t_dark", "t_orange", "t_over", "w_red", "w_hole", "w_edge", "s_red", "s_hole", "s_edge", "margin_min"):
        print(f"  {kk:10s} {getattr(best, kk):.3f}")
    if not a.no_write:
        print("Wrote", save_thresholds(best, a.out))
    return best


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
