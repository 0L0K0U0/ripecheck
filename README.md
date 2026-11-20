# RipeCheck

Grade an oil palm fresh fruit bunch (FFB) as **unripe, ripe or overripe** from one photo, using classical
computer vision (OpenCV, no trained model). A Flask web app with a hand-built front end.
Built for AIT204 Assignment 2, based on Udos & Lim (2026), *Seeing ripeness differently*, Bernama.

It is a decision aid, not a replacement for a harvester's judgement. Part of that judgement (loose fruit on the
ground, empty sockets) is only partly visible in one photo.

## Quick start

- **Windows with Anaconda:** double-click `start_windows.bat`.
- **macOS:** double-click `start_mac_linux.command` (if nothing happens, run `chmod +x start_mac_linux.command` once).
- **By hand** (any system, Python 3.10 or newer):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first launcher run creates a conda environment called `ripecheck` and installs the packages (a few minutes).
The app opens at http://127.0.0.1:5057. Set `PORT=8000` to use another port, `NO_BROWSER=1` to stop it opening a
browser. To stop it, press Ctrl+C.

## Using the app

1. **Grade a bunch.** Pick a sample, take a photo, or choose a file. Drag a box around the whole bunch (arrow keys
   move the box, Shift+arrows resize it). The grade, the reasons and the photo checks appear on the slip.
   The tabs under the photo show the outline, the colour map, the edges, a before/after slider and every
   processing step.
2. **Views of a bunch.** Add two or three photos of the same bunch. Their features are averaged before grading.
3. **Block planner.** Save a bunch to a block (A1, B2...). Blocks are ordered by a simple illustrative rule. Saved in
   the browser only; download a CSV.
4. **Test a set.** Upload many photos. File names starting `unripe_`, `ripe_` or `overripe_` are scored.

## How the grade is made

| Step | Technique | What it does |
|---|---|---|
| 1 | Bilateral filter | Smooths noise and keeps edges |
| 2 | CLAHE on LAB lightness | Local contrast boost, used for the edge map |
| 3 | HSV and LAB colour spaces | Colour classes and redness |
| 4 | GrabCut from your box (or HSV rules, or k-means) | Cuts the bunch out; foliage removed by hue |
| 5 | Opening, closing, largest component | Cleans the cut-out |
| 6 | Canny edges | Edge density inside the bunch |
| 7 | Contours | Holes (empty sockets, gaps) and solidity; features; rules |

Features (measured inside the bunch, away from its outline, ignoring glare and deep shadow):
`dark_ratio`, `orange_ratio`, `red_ratio`, `hole_ratio`, `edge_density`.

Rules, checked in this order:
1. dark share >= `t_dark` -> **unripe**
2. over-ripe score >= `t_over` -> **overripe** (score = weighted deep red, gaps and rough surface, each scaled to 0-1)
3. orange share >= `t_orange` -> **ripe**
4. otherwise -> **uncertain**

The **margin** is how far the deciding value is past its threshold. Below `margin_min` the result is **uncertain**.
Before grading, **photo checks** (bunch size, evenly lit pixels, sharpness, exposure, foliage, fruit-like colour)
can return **not gradable** with a reason and a fix, instead of guessing.

Two design choices to know about:
- **Colour is measured before CLAHE by default.** CLAHE equalises each tile, so it changes absolute brightness,
  which the dark rule relies on. Tick *Measure colour after the contrast boost* to compare.
- **Gray-world white balance is off by default.** It assumes the scene averages to grey, which an orange bunch
  filling the frame breaks. Test it in your ablation before turning it on.

## Using your own data (this is what makes the results meaningful)

**The sample images are synthetic.** They are drawn by `make_samples.py`. The default `thresholds.json` was fitted to
them so the demo behaves sensibly. Neither says anything about real fruit. The app shows this under the result.

1. Get real photos. Your own, or a public dataset. One candidate is Omar et al. (2024), *Outdoor oil palm fruit
   ripeness dataset*, Data in Brief, https://doi.org/10.1016/j.dib.2024.110667 (images also on Zenodo, about 1.2 GB).
   **Check the licence on the dataset page and cite it.** Do not commit the whole dataset to GitHub.
2. Arrange folders, one per class (names must match). Keep the classes you are scoring: `unripe`, `ripe`, `overripe`.

```
data/train/unripe/*.jpg   data/train/ripe/*.jpg   data/train/overripe/*.jpg
data/test/unripe/*.jpg    data/test/ripe/*.jpg    data/test/overripe/*.jpg
```

3. **Split by bunch, not by photo.** If one bunch has several photos, all of them go to train or all to test.
   Otherwise the same bunch appears on both sides and the scores are inflated. Tell the scripts which photos share
   a bunch with `data/train/labels.csv` and `data/test/labels.csv`:

```
file,label,bunch
ripe/img_001.jpg,ripe,tree12_bunch3
ripe/img_002.jpg,ripe,tree12_bunch3
```

4. Calibrate on the **train** folder only (writes `thresholds.json`):

```bash
python calibrate.py data/train
```

   It grid-searches the thresholds for the best macro F1 and reports grouped cross-validation (whole bunches held out).
5. Freeze the thresholds. Evaluate **once** on the test folder:

```bash
python evaluate.py data/test
python evaluate.py data/test --fuse-views                    # one grade per bunch
python evaluate.py data/test --ablation --train data/train   # what each technique contributes
python evaluate.py data/test --tree data/train               # optional: compare with a small decision tree (needs scikit-learn)
```

   Reports per-class precision, recall and F1, accuracy, a confusion matrix, and **coverage**. Uncertain and
   not-gradable photos count as misses, so abstaining never improves the headline numbers. A margin sweep shows
   accuracy against coverage. Results go to `results/`.

Do not tune on the test folder. If you change something after seeing test results, say so in the report.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server (the only web code in Python) |
| `cv_pipeline.py` | All computer vision and the grading rules |
| `templates/index.html`, `static/style.css`, `static/app.js` | The front end |
| `calibrate.py`, `evaluate.py`, `dataset.py`, `metrics.py` | Calibration and evaluation |
| `make_samples.py` | Generates the synthetic sample images and `sample_data/` |
| `test_pipeline.py` | Self-check: `python test_pipeline.py` |
| `thresholds.json` | Current rules (demo values until you calibrate) |
| `sample_images/`, `sample_data/` | Synthetic demo photos |
| `start_windows.bat`, `start_mac_linux.command` | One-click launchers |

## Hosting (optional)

The app is a standard Flask app (`app:app`). It can run on any Python host, for example with
`pip install gunicorn` then `gunicorn app:app`. Free hosts have limits, so also keep the run instructions above in
your report. Set `HOST=0.0.0.0` only if you deliberately want it reachable from other machines.

## Known limits

- Light, shadow and the phone camera change colour. Thresholds are calibrated for one set of conditions.
- Ripe versus overripe is hard from colour alone. Expect middling accuracy and report what you measure.
- GrabCut can include background that looks like fruit, and the k-means mode is experimental.
- One photo shows one side of the bunch. Use several views.
- It grades colour and texture only. It does not count loose fruit on the ground.
- The block planner order is an illustration, not an agronomic rule.

## Acknowledgements (complete before submitting)

- OpenCV, NumPy and Flask are open-source libraries.
- Sample images are synthetic, generated by `make_samples.py`.
- Generative AI assistance: *state which tool you used and for what.*
- Real photos or datasets you add: *list source, licence and citation for each.*

## Reference

Udos, W., & Lim, K. S. (2026, July 22). Seeing ripeness differently. *Bernama*. https://bernama.com/en/thoughts/news.php?id=2581455
