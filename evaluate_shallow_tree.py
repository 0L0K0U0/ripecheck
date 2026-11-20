"""Train a depth-three tree on real-photo features and evaluate both existing sets."""

import argparse
import json
from pathlib import Path

import numpy as np

from cv_pipeline import FEATURE_KEYS, Params
from dataset import featurize, load_items
from evaluate import print_report
from metrics import LABELS, summarise


def build_tree(features, labels, depth):
    counts = np.bincount(labels, minlength=len(LABELS))
    best = None
    if depth and np.count_nonzero(counts) > 1:
        for column in range(features.shape[1]):
            values = np.unique(features[:, column])
            for threshold in (values[1:] + values[:-1]) / 2:
                left = features[:, column] <= threshold
                if min(left.sum(), (~left).sum()) < 2:
                    continue
                a = np.bincount(labels[left], minlength=len(LABELS))
                b = counts - a
                score = (a @ a) / a.sum() + (b @ b) / b.sum()
                if best is None or score > best[0]:
                    best = score, column, threshold, left
    if best is None:
        return int(counts.argmax())
    _, column, threshold, left = best
    return (column, threshold, build_tree(features[left], labels[left], depth - 1),
            build_tree(features[~left], labels[~left], depth - 1))


def feature_vector(row):
    return [row["features"][key] for key in FEATURE_KEYS]


def fit_tree(rows):
    train = [row for row in rows if row["ok"] and row["features"]]
    features = np.array([feature_vector(row) for row in train])
    labels = np.array([LABELS.index(row["label"]) for row in train])
    return build_tree(features, labels, depth=3), len(train)


def classify(tree, row):
    if not row["ok"] or not row["features"]:
        return "not gradable"
    features = feature_vector(row)
    while isinstance(tree, tuple):
        column, threshold, left, right = tree
        tree = left if features[column] <= threshold else right
    return LABELS[tree]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsv-fallback", action="store_true",
                        help="Retry rejected GrabCut photos using a tree trained on HSV features.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    def load(split, mode="grabcut"):
        items, _ = load_items(str(root / "data" / split))
        return featurize(items, Params(seg_mode=mode), quiet=True)

    tree, training_count = fit_tree(load("train"))
    output = root / ("results_tree_hsv_fallback" if args.hsv_fallback else "results_shallow_tree")
    if args.hsv_fallback:
        hsv_tree, hsv_training_count = fit_tree(load("train", "hsv"))
    output.mkdir(exist_ok=True)
    for split in ("demo_easy", "test"):
        rows = load(split)
        photos = [{"file": row["rel"], "label": row["label"], "result": classify(tree, row)}
                  for row in rows]
        metadata = {"training_photos_used": training_count}
        if args.hsv_fallback:
            metadata["hsv_training_photos_used"] = hsv_training_count
            rejected = [photo for photo in photos if photo["result"] == "not gradable"]
            if rejected:
                hsv_rows = {row["rel"]: row for row in load(split, "hsv")}
                for photo in rejected:
                    photo["grabcut_result"] = photo["result"]
                    photo["result"] = classify(hsv_tree, hsv_rows[photo["file"]])
                    photo["fallback"] = "hsv"
        summary = summarise([row["label"] for row in photos], [row["result"] for row in photos])
        print_report(summary, split)
        (output / f"{split}.json").write_text(
            json.dumps({**metadata, "summary": summary, "photos": photos},
                       indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
