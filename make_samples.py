"""
make_samples.py - generate SYNTHETIC oil palm bunch images for RipeCheck.

These are drawn by code, not photographed. They exist so the app, the
calibration script and the evaluation script can be run and tested straight
away. They are NOT real data: results on them say nothing about accuracy on
real fruit, and they must not be used as evidence in the report.

Writes:
  sample_images/   a few demo images for the web app (class prefix in the name)
  sample_data/train/<class>/*.jpg  and  sample_data/test/<class>/*.jpg
  sample_data/train/labels.csv, sample_data/test/labels.csv   (file,label,bunch)

Run:  python make_samples.py
"""
import csv
import os

import cv2
import numpy as np

W, H = 640, 480
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("unripe", "ripe", "overripe")


def rgb(r, g, b):
    return np.array([b, g, r], np.float32)          # OpenCV order


def background(rng, with_trunk=True):
    img = np.zeros((H, W, 3), np.float32)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    top, bottom = rgb(190, 215, 232), rgb(120, 150, 110)
    img[:] = (top * (1 - t) + bottom * t)[:, None, :]
    if with_trunk:
        x = int(W * 0.5 + rng.normal(0, 12))
        cv2.rectangle(img, (x - 34, 0), (x + 34, H), tuple(float(v) for v in rgb(86, 62, 44)), -1)
    for _ in range(46):                              # fronds and leaves, mostly at the sides and top
        side = rng.random()
        cx = rng.uniform(0, W * 0.28) if side < .4 else rng.uniform(W * 0.72, W) if side < .8 else rng.uniform(0, W)
        cy = rng.uniform(0, H * 0.45) if side >= .8 else rng.uniform(0, H)
        col = rgb(rng.integers(26, 70), rng.integers(80, 140), rng.integers(30, 70))
        cv2.ellipse(img, (int(cx), int(cy)), (int(rng.integers(40, 110)), int(rng.integers(7, 16))),
                    float(rng.uniform(-70, 70)), 0, 360, tuple(float(v) for v in col), -1, cv2.LINE_AA)
    return img


def fruit(img, c, r, col):
    col = np.clip(col, 0, 255)
    cv2.circle(img, c, r, tuple(float(v) for v in col * 0.55), -1, cv2.LINE_AA)
    cv2.circle(img, c, r - 2, tuple(float(v) for v in col), -1, cv2.LINE_AA)
    hl = np.clip(col + 38, 0, 255)
    cv2.ellipse(img, (c[0] - r // 3, c[1] - r // 3), (max(1, r // 4), max(1, r // 5)), -30, 0, 360,
                tuple(float(v) for v in hl), -1, cv2.LINE_AA)


def bunch(img, rng, cls, mix):
    """Draw a bunch of fruitlets. `mix` sets how much of the off-colour or gap fraction to use."""
    cx, cy = W * 0.5 + rng.normal(0, 14), H * 0.5 + rng.normal(0, 10)
    rx, ry = 150 + rng.normal(0, 12), 168 + rng.normal(0, 12)
    pts = []
    while len(pts) < 230:
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y <= 1:
            pts.append((cx + x * rx, cy + y * ry))
    pts.sort(key=lambda q: q[1])
    for x, y in pts:
        r = int(rng.integers(14, 20))
        jit = rng.normal(0, 7, 3).astype(np.float32)
        u = rng.random()
        if cls == "unripe":
            col = rgb(40, 26, 62) + jit if u > mix else rgb(78, 52, 108) + jit
        elif cls == "ripe":
            col = (rgb(238, 118, 34) if u > mix else rgb(212, 72, 28)) + jit
        else:
            if u < mix * 0.55:                       # empty socket
                cv2.circle(img, (int(x), int(y)), 7, tuple(float(v) for v in rgb(140, 108, 68)), -1, cv2.LINE_AA)
                continue
            col = (rgb(168, 40, 30) if u > mix else rgb(228, 98, 30)) + jit
            if rng.random() < 0.18:
                col = rgb(100, 28, 22) + jit
        fruit(img, (int(x), int(y)), r, col)
    for a in np.linspace(0, 2 * np.pi, 34, endpoint=False):          # spikelets at the rim
        x0, y0 = cx + np.cos(a) * rx * 0.96, cy + np.sin(a) * ry * 0.96
        L = rng.uniform(14, 30)
        cv2.line(img, (int(x0), int(y0)), (int(x0 + np.cos(a) * L), int(y0 + np.sin(a) * L)),
                 tuple(float(v) for v in rgb(70, 52, 36)), 2, cv2.LINE_AA)
    if cls == "overripe":                                              # loose fruit on the ground
        for _ in range(int(rng.integers(8, 14))):
            fruit(img, (int(rng.uniform(W * 0.25, W * 0.75)), int(H * rng.uniform(0.9, 0.97))), 11,
                  rgb(214, 80, 28) + rng.normal(0, 8, 3).astype(np.float32))
    return img


def finish(img, rng, brightness=1.0, cast=(1, 1, 1), glare=0, blur=0.7):
    img = img * brightness * np.array(cast[::-1], np.float32)
    if glare:
        layer = np.zeros((H, W), np.float32)
        for _ in range(glare):
            cv2.ellipse(layer, (int(rng.uniform(W * .32, W * .68)), int(rng.uniform(H * .3, H * .7))),
                        (int(rng.integers(8, 20)), int(rng.integers(5, 12))), 0, 0, 360, 1.0, -1)
        a = np.clip(cv2.GaussianBlur(layer, (0, 0), 3) * 1.4, 0, 1)[..., None]
        img = img * (1 - a) + np.array([245, 248, 250], np.float32) * a
    img = cv2.GaussianBlur(img, (0, 0), blur) + rng.normal(0, 2.5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def make(cls, seed, mix, brightness=1.0, glare=0, cast=(1, 1, 1), blur=0.7):
    rng = np.random.default_rng(seed)
    return finish(bunch(background(rng), rng, cls, mix), rng, brightness, cast, glare, blur)


def save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 86])


def split(name, n_bunches, seed0):
    rows = {"train": [], "test": []}
    for ci, cls in enumerate(CLASSES):
        for b in range(n_bunches):
            rng = np.random.default_rng(seed0 + ci * 100 + b)
            mix = {"unripe": rng.uniform(0.0, 0.30), "ripe": rng.uniform(0.05, 0.40),
                   "overripe": rng.uniform(0.15, 0.55)}[cls]
            for v in (1, 2):
                br = rng.uniform(0.80, 1.18)
                cast = tuple(1 + rng.normal(0, 0.035, 3))
                img = make(cls, seed0 + ci * 100 + b * 7 + v, mix, br, int(rng.integers(0, 3)), cast)
                fn = f"b{b + 1:02d}_v{v}.jpg"
                save(os.path.join(HERE, "sample_data", name, cls, fn), img)
                rows[name].append((f"{cls}/{fn}", cls, f"{cls}_b{b + 1:02d}"))
    with open(os.path.join(HERE, "sample_data", name, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "bunch"])
        w.writerows(rows[name])


if __name__ == "__main__":
    split("train", 6, 1000)
    split("test", 4, 5000)
    demo = os.path.join(HERE, "sample_images")
    save(os.path.join(demo, "unripe_demo.jpg"), make("unripe", 9001, 0.12, 1.0))
    save(os.path.join(demo, "ripe_demo.jpg"), make("ripe", 9002, 0.15, 1.0))
    save(os.path.join(demo, "overripe_demo.jpg"), make("overripe", 9003, 0.40, 1.0))
    save(os.path.join(demo, "ripe_dim.jpg"), make("ripe", 9004, 0.15, 0.42, blur=1.4))
    save(os.path.join(demo, "ripe_glare.jpg"), make("ripe", 9005, 0.15, 1.05, glare=9))
    rng = np.random.default_rng(9006)
    save(os.path.join(demo, "leaves_only.jpg"), finish(background(rng, with_trunk=False), rng))
    print("sample images and sample_data written")
