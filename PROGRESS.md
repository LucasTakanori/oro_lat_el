# Progress Log — Tongue-tip Tracking for LROM

Ongoing log of work done. Append one section per working step.
Latest on top.

---

## Step 3 — Inner-lip polygon ROI (option a): beard fixed, lips remain

**Date**: 2026-04-20
**Goal**: replace rectangular ROI with MediaPipe inner-lip ring polygon to geometrically exclude beard/outer-lip pixels.

### Changes

- `scoring.py`:
  - Added `INNER_LIP_RING` (20-pt MediaPipe inner-lip closed ring).
  - Added `_inner_lip_polygon`, `_polygon_mask`, `_extend_polygon_horizontally`.
  - `_mouth_mask_and_bbox` now rasterizes the polygon when `lm` is provided (falls back to rect bbox otherwise).
  - Lat tasks: polygon extended ±0.30·IC horizontally for tongue protrusion past commissure.
  - Elev task: polygon extended upward by 0.35·MH for tongue reaching toward palate.
- `detect_tongue_tip`, `extract_frame_features`, `load_predictor` signatures now accept `lm=None`.
- `build_dataset.py` / `score_clip` pass per-frame `lm` through.
- `debug_polygon.py` added to render ROI + tip overlay on peak frames.

### Result: beard excluded, but lips are still pink

Debug images (`/tmp/dbg_poly_latR_*_s50.png`, `/tmp/dbg_poly_latL_*_s25.png`, `/tmp/dbg_poly_latL_*_s75.png`) show the polygon correctly outlines the mouth opening with lateral extension past commissures. Zero beard/stubble in ROI now. Subject `01` no longer NaN (CR threshold robust).

But the color mask inside the polygon picks up **both** lip tissue (pink) and tongue (also pink/red), so the tip-extremum percentile just lands on the inner lip corner, regardless of actual tongue position. The 20-point inner-lip ring isn't a fine-grained enough boundary either — there are gaps the mask leaks through.

### LOSO (22 clips, 5 subjects) — got WORSE

| Method | MAE | disc-acc | q-kappa | latR MAE | latL MAE | elev MAE |
|--------|-----|----------|---------|----------|----------|----------|
| baseline (mean) | 15.8 | 0.77 | 0.00 | 14.5 | 18.7 | 14.5 |
| **geo (raw) — Step 1 (rect ROI)** | 16.5 | 0.59 | −0.23 | **7.4** | 14.3 | 29.1 |
| **geo (raw) — Step 3 (polygon ROI)** | 27.7 | 0.45 | −0.14 | 14.3 | 33.2 | 37.4 |
| geo + LOSO isotonic (Step 3) | 17.6 | 0.77 | 0.00 | 16.7 | 21.9 | 14.1 |

latR got slightly worse (7.4 → 14.3) because the rectangular ROI happened to accidentally bias tip-x toward true tongue pixels on several clips; polygon removes that bias but replaces it with inner-lip-corner extremum.

### Interpretation

Polygon fixes one problem (beard) and exposes another more fundamental one: **tongue ≠ distinguishable from lip tissue by color alone**. Both occupy the same saturated-red region of HSV/YCrCb.

Any color-only scorer will plateau here. No further threshold tweak will help.

### Decision

Proceed with option (c) — SAM2 auto-labeling + YOLOv8-pose tip keypoint. Rectangular ROI + YCrCb was good enough to prototype the pipeline; polygon refined the geometry. Next is the mask: SAM2 with a single positive click prompt per clip (on the peak frame of tongue extension) produces a pixel-precise tongue mask, from which the tip keypoint can be extracted geometrically or regressed.

### Artifacts

- `/tmp/dbg_poly_*.png` — polygon+mask+tip overlays
- `poc/out/dataset.csv` — polygon-ROI scores (still not trustworthy; kept for comparison)
- `poc/out/predictions.csv` — LOSO predictions across all methods

---

## Step 2 — Mirror fix + YCrCb pivot, segmentation still fails on beard

**Date**: 2026-04-20
**Goal**: fix the two bugs from Step 1's label-noise finding — inverted mirror semantics + HSV bleeding onto skin/beard.

### Changes

| File | Change |
|------|--------|
| `poc/build_dataset.py` | `mirrored = False` hardcoded — `cameraFlipped` in meta is UI selfie-mirror only, stored video is raw camera (subject-right = image-left) |
| `poc/scoring.py` | `_hsv_tongue_mask` switched from HSV hue band to **YCrCb Cr ≥ 150** AND **red_excess (2R−G−B) ≥ 35** AND morphological open 3×3 |
| `poc/scoring.py` | Lat horizontal ROI pad tightened `0.60·IC → 0.30·IC` |
| `poc/scoring.py` | Lat aggregation: HSV conf-only gate (`tip.conf > 0.15`), skipping visibility classifier (biased toward elevation-style round blobs) |

### Result: still broken on Lucas

Debug visualizations (`/tmp/dbg_latR_*_s50.png`, `/tmp/dbg_latL_*_s25.png`):
- Yellow "tongue" mask still covers pink lips + beard stubble on both sides of mouth
- Red-dot tip marker lands on beard, not tongue
- Actual tongue visible as **dark-brown/maroon** region inside mouth — darker than surrounding illuminated skin+lips, inverting the usual "tongue = saturated red" assumption
- One clip had blob=131525 px — obviously not tongue, filling entire face region

### Cross-subject instability

- Tightening `CR_MIN` to catch only saturated tongue on Lucas breaks subject `01` (returns all-NaN lat scores — threshold too aggressive for their lighting/skin tone).
- No single color threshold works across subjects.

### Interpretation

Pure color-based tongue segmentation is **fundamentally insufficient** for this dataset:
- Beard stubble has higher Cr than expected (facial hair shadow + skin undertone)
- Tongue in mouth shadow has LOWER V than lip tissue
- Lip tissue and tongue tissue occupy overlapping regions in HSV, YCrCb, and red-excess space

More HSV tweaks will overfit to one subject and break others.

### Decision point for next step

Three options, pending user pick:
- **(a) Inner-lip polygon ROI** via MediaPipe inner-lip ring (13, 14, 78, 95, 88, 178, 87, 82, 81, 80, 191, 308, 415, 310, 311, 312, 317, 402, 318, 324). Precisely excludes beard/outer-lip. Still relies on color inside polygon — may not rescue Lucas if tongue is darker than lips.
- **(b) Dark-region detection inside the polygon.** Tongue in shadow is darker than surrounding pink lip tissue — flip assumption, look for medium-V regions.
- **(c) Jump to Phase 2** (SAM2 auto-label → YOLOv8-pose). Dataset is small enough that manual SAM click-prompts on ~22 clips × peak-frame is tractable and gives a proper tongue mask.

### Artifacts

- `/tmp/dbg_latR_*_s50.png`, `/tmp/dbg_latL_*_s25.png` — annotated debug frames showing mask bleed
- `poc/out/dataset.csv` — re-built with mirror fix and YCrCb mask (scores still unreliable)

---

## Step 1 — Phase 1 implementation (geometric scorer + visibility classifier)

**Date**: 2026-04-20
**Goal**: implement Phase 1 of `plan_tongue_tracking.md`: continuous geometric scoring for lateralization and elevation, visibility classifier, LOSO evaluation vs. existing baselines.

### Files added / changed

| Path | Change | Purpose |
|------|--------|---------|
| `poc/scoring.py` | **new** | Continuous formulas, HSV tip + confidence, resting-frame calibration, task-adaptive ROI, clip aggregation |
| `poc/visibility_clf.py` | **new** | Logistic-regression P(tongue tip visible) on HSV + landmark features; pseudo-labeled from rest frames + HSV confidence; ONNX-free |
| `poc/build_dataset.py` | modified | Loads visibility classifier, calls `scoring.score_clip()`, writes new columns to `dataset.csv` |
| `poc/train.py` | modified | Adds geometric-scorer eval (raw + LOSO isotonic), quadratic-weighted kappa, predictions plot across methods |

### Formulas (current)

- **Lateralization** — signed displacement from mouth center toward task-side commissure, normalized by `IC/2`:
  `score = clip( sign × (tip_x − MC_x) / (IC/2), 0, 1) × 100`
- **Elevation** — two-branch gated by visibility:
  - `v < 0.5` + mouth open → `100` (palate contact, tip hidden)
  - `v < 0.5` + mouth closed → `None` (reject — unassessable)
  - `v ≥ 0.5` → linear on vertical tip position, **capped at 50** (visible tip cannot score 100)
- **Clip aggregation** — lat: 80th-percentile over valid frames; elev: mean over valid frames.
- **Tip detection** — HSV blob inside inner-lip ROI (y ∈ [UL + 0.1·MH, LL − 0.1·MH]), tip is 10th/90th percentile of blob pixels in the task direction (robust to edge outliers, not absolute extremum).
- **Resting-frame calibration** — first 5 frames define the reference `CL, CR, MC, IC` (canceling head-pose drift and facial asymmetry bias).

### Visibility classifier

- Features: `blob_area_frac, sat_mean, hue_concentration, mouth_open, blob_y_norm, blob_below_upper` (6 dims).
- Pseudo-labels: rest frames + zero-blob frames + mouth-closed frames → negative; high-area + high-sat + high-hue-concentration → positive; rest ambiguous.
- Model: `StandardScaler → LogisticRegression(C=1.0)`.
- **LOSO AUC = 0.921** on 687 labeled frames (132 pos / 555 neg) from 1273 total.

### LOSO evaluation (22 clips, 5 subjects)

| Method | MAE | disc-acc | q-kappa | latR MAE | latL MAE | elev MAE |
|--------|-----|----------|---------|----------|----------|----------|
| baseline (mean) | 15.8 | 0.77 | 0.00 | 14.5 | 18.7 | 14.5 |
| **geo (raw)** | 16.5 | 0.59 | −0.23 | **7.4** | 14.3 | 29.1 |
| geo + LOSO isotonic | 17.8 | 0.77 | 0.00 | 16.7 | 21.2 | 15.5 |
| Ridge | 19.5 | 0.59 | −0.23 | 21.5 | 26.3 | 10.6 |
| RandomForest | 18.0 | 0.73 | −0.08 | 15.6 | 25.3 | 13.3 |

Plot: `poc/out/predictions.png`.

### Label-noise finding (correction on prior read)

Earlier I stated the Lucas `latR_s50` clip showed the tongue "genuinely past commissure," indicating label noise. **That was wrong.**

Looking at the actual debug image (`/home/lucas_takanori/oro_lat_el/image.png`):
- The tongue is **inside the mouth, at the image-left commissure** (subject's right side) — correctly placed for the `latR` task.
- The red dot I drew as "tip" lands in **beard stubble past the image-right commissure** — a false HSV detection of skin/facial-hair redness, not tongue tissue.
- So the clinical label `s50` is probably fair; the geometric scorer was fooled by skin-redness bleed outside the actual mouth opening.

This reveals two distinct bugs, not label noise:

1. **Inverted mirror semantics.** The `cameraFlipped=true` flag in the landmarks JSON means the display was mirrored for the user during capture, but the *saved* video stream is the raw (non-mirrored) camera feed. In that feed, subject-right = image-left. My code assumed the opposite, so the "tip in target direction" picker looked in the wrong side of the image.
2. **HSV hue range too permissive.** The band `H ∈ [330°, 360°] ∪ [0°, 20°]` + `S ≥ 0.25` picks up skin tones and beard stubble in addition to tongue. The inner-lip vertical constraint helps but doesn't fully exclude redness bleed at the mouth corners / lip shadow.

### Known limitations (to drive the next step)

- Dataset is tiny and imbalanced: 17/22 = s100. Kappa is uninformative on that distribution.
- HSV mask bleeds onto skin / beard around the mouth. Needs better filter or a segmentation-based tongue mask (SAM2).
- Visibility classifier (AUC 0.921) is trained on a binary pseudo-labeling heuristic; its definition of "visible" doesn't separate "tongue body visible + tip hidden at palate" from "tip visible, some elevation."
- Mirror semantics need fixing before any scoring is trusted — all current lat scores are measured on the wrong side of the image.

### Artifacts

- `poc/out/dataset.csv` — manifest with new score columns
- `poc/out/predictions.csv` — per-clip predictions from all 4 methods
- `poc/out/visibility_clf.pkl`, `visibility_features.csv`
- `poc/out/predictions.png`, `dataset_overview.png`, `crops_gallery.png`

### Next

1. **Fix mirror semantics** — verify by re-reading `index.html` capture logic, flip the condition, re-run pipeline.
2. **Tighten HSV** — add `2R−G−B > threshold` requirement (red excess over skin) and/or morphological opening to drop small skin-colored pixels at mask edge.
3. Re-evaluate LOSO metrics with both fixes in place.
