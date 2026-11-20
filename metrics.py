"""metrics.py - scoring helpers shared by evaluate.py, calibrate.py and the web app."""
from __future__ import annotations

LABELS = ("unripe", "ripe", "overripe")
OUTCOMES = LABELS + ("uncertain", "not gradable")


def summarise(y_true: list, y_pred: list) -> dict:
    """y_true in LABELS, y_pred in OUTCOMES. 'uncertain' and 'not gradable' count as misses."""
    n = len(y_true)
    cm = {t: {p: 0 for p in OUTCOMES} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in OUTCOMES:
            cm[t][p] += 1
    per = {}
    for c in LABELS:
        tp = cm[c][c]
        support = sum(cm[c].values())
        predicted = sum(cm[t][c] for t in LABELS)
        prec = tp / predicted if predicted else None
        rec = tp / support if support else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if support else None)
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    graded = sum(cm[t][p] for t in LABELS for p in LABELS)
    correct = sum(cm[c][c] for c in LABELS)
    f1s = [per[c]["f1"] for c in LABELS if per[c]["f1"] is not None]
    return {
        "n": n, "per_class": per, "confusion": cm,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "accuracy_all": correct / n if n else None,
        "accuracy_graded": correct / graded if graded else None,
        "coverage": graded / n if n else None,
    }
