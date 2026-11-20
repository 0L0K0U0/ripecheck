"""Compare whole-image and manually boxed grading on the fixed demonstration set."""

import json
from pathlib import Path

import cv2

from cv_pipeline import Params, analyse, load_thresholds
from evaluate import print_report
from metrics import summarise


# Normalised x0, y0, x1, y1 coordinates drawn around the visible bunch.
BOXES = {
    "TestOverripe(3)": (0.00, 0.01, 0.98, 0.99),
    "TestOverripe(5)": (0.00, 0.05, 0.92, 0.96),
    "TestOverripe(7)": (0.15, 0.01, 0.82, 0.99),
    "TestOverripe(8)": (0.21, 0.17, 0.64, 0.87),
    "TestRipe(13)": (0.14, 0.10, 0.81, 0.87),
    "TestRipe(20)": (0.28, 0.05, 0.81, 0.88),
    "TestRipe(23)": (0.21, 0.06, 0.81, 0.85),
    "TestRipe(26)": (0.21, 0.04, 0.72, 0.87),
    "TestUnripe(13)": (0.08, 0.21, 0.94, 0.99),
    "TestUnripe(17)": (0.18, 0.00, 0.96, 1.00),
    "TestUnripe(24)": (0.22, 0.13, 0.90, 0.77),
    "TestUnripe(30)": (0.10, 0.07, 0.79, 0.86),
}


def main():
    root = Path(__file__).resolve().parent
    thresholds = load_thresholds(str(root / "thresholds.json"))
    sample = root / "data" / "demo_easy"
    rows = []
    for path in sorted(p for p in sample.glob("*/*") if p.stem in BOXES):
        photo = cv2.imread(str(path))
        if photo is None:
            raise ValueError(f"Cannot read {path}")
        box = BOXES[path.stem]
        grades = {}
        for name, region in (("whole", (0, 0, 1, 1)), ("boxed", box)):
            result = analyse(photo, region, Params(), thresholds, want_images=False)
            grades[name] = result.grade.lower()
        rows.append({"file": path.relative_to(sample).as_posix(),
                     "label": path.parent.name, "box": box, **grades})
    if len(rows) != len(BOXES):
        raise ValueError("Expected all 12 demonstration photos")
    labels = [r["label"] for r in rows]
    reports = {}
    for name in ("whole", "boxed"):
        reports[name] = summarise(labels, [r[name] for r in rows])
        print_report(reports[name], name)
    output = root / "results_demo_easy" / "box_comparison.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"summary": reports, "photos": rows}, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
