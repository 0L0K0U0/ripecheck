"""
test_pipeline.py - self-check. Run:  python test_pipeline.py

Checks that the code runs and behaves sensibly on the SYNTHETIC sample images
and through the web API. Passing says nothing about accuracy on real fruit.
"""
import io
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_pipeline import Params, Thresholds, analyse, fuse_views, grade_arrays, load_thresholds  # noqa: E402
from metrics import summarise  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load(name):
    img = cv2.imread(os.path.join(HERE, "sample_images", name))
    if img is None:
        sys.exit(f"Missing sample_images/{name}. Run: python make_samples.py")
    return img


BOX = (0.08, 0.08, 0.92, 0.92)
thr = load_thresholds()

# 1. the demo images get the grades they were drawn to show
expected = {"unripe_demo.jpg": "Unripe", "ripe_demo.jpg": "Ripe", "ripe_dim.jpg": "Ripe", "ripe_glare.jpg": "Ripe",
            "overripe_demo.jpg": "Overripe", "leaves_only.jpg": "Not gradable"}
for name, want in expected.items():
    r = analyse(load(name), BOX, thr=thr)
    check(f"{name} is {want}", r.grade == want, f"got {r.grade}, margin {r.margin:.2f}")

# 2. the same photo gives the same numbers twice (GrabCut and k-means are seeded)
a = analyse(load("ripe_demo.jpg"), BOX, thr=thr, want_images=False).features
b = analyse(load("ripe_demo.jpg"), BOX, thr=thr, want_images=False).features
check("analysis is repeatable", all(abs(a[k] - b[k]) < 1e-12 for k in a))

# 3. every segmentation mode runs
for mode in ("grabcut", "hsv", "kmeans"):
    r = analyse(load("overripe_demo.jpg"), BOX, Params(seg_mode=mode), thr, want_images=False)
    check(f"segmentation mode {mode} runs", r.features["area_ratio"] > 0.05, f"area {r.features['area_ratio']:.2f}")

# 4. a box that is too small is refused
try:
    analyse(load("ripe_demo.jpg"), (0.5, 0.5, 0.51, 0.51), thr=thr)
    check("tiny box is refused", False)
except ValueError:
    check("tiny box is refused", True)

# 5. rule logic
t = Thresholds(t_dark=0.4, t_orange=0.3, t_over=0.5, margin_min=0.1)
code, margin, _ = grade_arrays([0.6, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0],
                               [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.3, 0.0], t)
check("rules: unripe, ripe, overripe, no rule", list(code) == [0, 1, 2, -1], str(list(code)))
check("margins are non-negative", bool(np.all(margin >= 0)))

# 6. several views are averaged
f1 = analyse(load("ripe_demo.jpg"), BOX, thr=thr, want_images=False).features
f2 = analyse(load("ripe_glare.jpg"), BOX, thr=thr, want_images=False).features
fz = fuse_views([{"features": f1, "gates_ok": True}, {"features": f2, "gates_ok": True}], thr)
check("two ripe views fuse to ripe", fz["grade"] == "Ripe" and fz["used"] == 2, fz["grade"])
check("no usable view is not gradable", fuse_views([{"features": f1, "gates_ok": False}], thr)["grade"] == "Not gradable")

# 7. metrics
s = summarise(["ripe", "ripe", "unripe", "overripe"], ["ripe", "uncertain", "unripe", "ripe"])
check("metrics count uncertain as a miss", abs(s["accuracy_all"] - 0.5) < 1e-9 and abs(s["coverage"] - 0.75) < 1e-9)

# 8. web API
import app as web  # noqa: E402
c = web.app.test_client()
check("page loads", c.get("/").status_code == 200)
with open(os.path.join(HERE, "sample_images", "ripe_demo.jpg"), "rb") as fh:
    j = c.post("/api/analyse", data={"image": (fh, "ripe_demo.jpg"), "box": "0.08,0.08,0.92,0.92"},
               content_type="multipart/form-data").json
check("API grades a photo", j["ok"] and j["grade"] == "Ripe" and "overlay" in j["images"])
bad = c.post("/api/analyse", data={"image": (io.BytesIO(b"nope"), "x.jpg")}, content_type="multipart/form-data")
check("API rejects a non-image", bad.status_code == 400)
fz = c.post("/api/fuse", json={"views": [{"features": f1, "gates_ok": True}]}).json
check("API fuses views", fz["ok"] and fz["grade"] == "Ripe")
files = [(open(os.path.join(HERE, "sample_images", n), "rb"), n) for n in ("ripe_demo.jpg", "unripe_demo.jpg")]
bj = c.post("/api/batch", data={"images": files}, content_type="multipart/form-data").json
check("API batch scores labelled files", bj["ok"] and bj["summary"]["n"] == 2, str(bj["summary"]["accuracy_all"]))

print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) failed: {failures}")
sys.exit(1 if failures else 0)
