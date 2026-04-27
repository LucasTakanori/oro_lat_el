# OroSense — Tongue ROM Automatic Scoring

Automated scoring of tongue range-of-motion (ROM) from webcam clips for stroke / oral-cancer patients using the Lazarus (2014) scale (0–100).

Three tasks: **lateralization right** (`latR`), **lateralization left** (`latL`), **elevation** (`elev`).

The pipeline: record → annotate → build dataset → train → validate.

---

## Repository layout

```
index.html                 # Browser data-recording app
poc/
  annotator.py             # Annotation server (backend)
  annotator.html           # Annotation UI (frontend)
  build_tip_dataset.py     # Build YOLO-pose dataset from annotations
  train_tip.py             # LOSO YOLOv8n-pose training
  render_val_video.py      # Render predicted vs GT tip on validation clips
  scoring.py               # Geometric scoring formulas + MediaPipe helpers
  explore_preprocessing.ipynb  # Preprocessing comparison notebook
data/
  <subject>/
    <stem>.webm            # Raw clip
    <stem>_meta.json       # Score + capture metadata
    <stem>_landmarks.json  # Per-frame MediaPipe face-mesh landmarks
poc/out/
  annotations/             # Per-clip annotation JSONs (tips + no_tongue)
dataset/                   # Built YOLO dataset (clahe pipeline)
dataset_clahe_unsharp/     # Built YOLO dataset (clahe + unsharp pipeline)
dataset_gamma_clahe_unsharp/  # Built YOLO dataset (gamma + clahe + unsharp)
runs/tip/
  loso_<subject>/          # Trained model weights + metrics per LOSO fold
poc/out/val_videos/        # Rendered validation videos
```

---

## 1 — Data recording

Open `index.html` directly in Chrome/Edge (no server needed).

1. Enter participant name.
2. Follow on-screen task instructions (latR, latL, elev).
3. Hold the requested tongue pose — a 2-second hold bar fills, then the clip saves automatically.
4. After playback, set the Lazarus score for what was actually performed.
5. Clips, landmarks and metadata are saved to `data/<subject>/`.

**Score rubric (Lazarus 2014):**

| Task | 100 | 50 | 25 | 0 |
|------|-----|----|----|---|
| Lat | Tongue touches commissure | < 50% reduction | > 50% reduction | No movement |
| Elev | Tip contacts alveolar ridge | Visible but no contact | — | No elevation |

---

## 2 — Annotation

The annotator marks tongue-tip pixel coordinates and no-tongue frames on every clip.

**Start the server:**
```bash
python -m poc.annotator
# opens http://localhost:8765
```

**Controls:**

| Action | Input |
|--------|-------|
| Place tip + advance frame | Left-click |
| Drag tip across frames | Left-click + drag |
| Mark NO TONGUE + advance | Right-click |
| Drag NO TONGUE across frames | Right-click + drag |
| Toggle NO TONGUE on frame | `N` |
| Range mode (bulk annotate) | Switch in UI; `[` / `]` set start/end |

Tips are saved densely — the server **linearly interpolates** between anchor clicks on save. Interpolation stops at any NO TONGUE frame.

**Output:** `poc/out/annotations/<subject>_<stem>.json`
```json
{
  "subject": "Albert",
  "stem": "latR_2026-04-23T23-51-10-466Z_s100",
  "task": "latR",
  "lazarus_score": 100,
  "tips": { "0": [x, y], "1": [x, y], ... },
  "no_tongue": { "12": true, "13": true, ... }
}
```

---

## 3 — Build dataset

Converts annotation JSONs to a YOLO-pose dataset (3 keypoints: left commissure, right commissure, tongue tip).

**Preprocessing pipelines:**

| Pipeline | What it does |
|----------|-------------|
| `clahe` | CLAHE contrast enhancement (default) |
| `clahe_unsharp` | CLAHE then unsharp mask |
| `gamma_clahe_unsharp` | Gamma (γ=0.5) → CLAHE → unsharp mask |

```bash
# default (clahe) — writes to dataset/
python -m poc.build_tip_dataset

# write to dataset_clahe_unsharp/
python -m poc.build_tip_dataset --pipeline clahe_unsharp

# write to dataset_gamma_clahe_unsharp/
python -m poc.build_tip_dataset --pipeline gamma_clahe_unsharp
```

One LOSO YAML is created per subject inside the dataset directory (e.g. `tip_loso_Miquel.yaml`). Labels are shared across pipelines; only images differ.

**Bounding box strategy:**
- Derived from MediaPipe inner-lip polygon (no manual bbox annotation needed).
- Lat tasks: extended ±0.6·IC horizontally to capture tongue protrusion past commissure.
- Elev task: extended upward by 0.5·IC to capture tongue reaching toward palate.
- Minimum height enforced at 0.2·IC so barely-open-mouth frames are not degenerate.

---

## 4 — Train

LOSO (Leave-One-Subject-Out) training with YOLOv8n-pose. One subject held out as validation per fold.

```bash
# all LOSO folds (default dataset/)
python -m poc.train_tip

# single fold
python -m poc.train_tip --val-subject Miquel

# custom pipeline dataset + custom run name
python -m poc.train_tip \
  --val-subject Miquel \
  --run-name loso_Miquel_clahe_unsharp_v1 \
  --dataset-dir /path/to/dataset_clahe_unsharp \
  --epochs 20
```

Results saved to `runs/tip/<run-name>/weights/best.pt`.

**Model:** YOLOv8n-pose, `kpt_shape: [3, 3]`, left-right flip disabled (laterality task). Trained with `batch=16, imgsz=640, epochs=20`.

**Trained runs available:**

| Run name | Val subject | Pipeline |
|----------|-------------|----------|
| `loso_Miquel_clahe_3kpt_v1` | Miquel | clahe |
| `loso_Miquel_clahe_unsharp_v1` | Miquel | clahe_unsharp |
| `loso_Miquel_gamma_clahe_unsharp_v1` | Miquel | gamma_clahe_unsharp |

---

## 5 — Validate (render video)

Renders an MP4 with predicted tip (red) vs ground-truth tip (green) overlaid on each frame. Also draws the two commissure points (orange) and shows keypoint visibility confidence.

```bash
# all clips for a subject using a trained fold
python -m poc.render_val_video --subject Miquel --fold loso_Miquel_clahe_unsharp_v1

# single clip
python -m poc.render_val_video \
  --subject Miquel \
  --clip latR_2026-04-23T23-51-10-466Z_s100 \
  --fold loso_Miquel_clahe_unsharp_v1

# save to a different subfolder name
python -m poc.render_val_video \
  --subject Miquel \
  --fold loso_Miquel_clahe_unsharp_v1 \
  --out-name my_comparison
```

Output videos: `poc/out/val_videos/<fold>/<subject>_<stem>.mp4`

**Overlay legend:**
- Green filled circle = ground-truth tip
- Red filled circle = predicted tip
- Orange dots = predicted commissures
- Yellow line = prediction error when GT exists
- `vis:X.XX` top-right = model's keypoint visibility confidence (red if < 0.5, green if ≥ 0.5)
- `NO TONGUE` label when GT says no tongue

---

## 6 — Preprocessing exploration

Interactive notebook to compare preprocessing options visually on annotated frames.

```bash
cd poc
jupyter notebook explore_preprocessing.ipynb
```

Sections:
- **Sanity check** — annotated tip dot on full frame (verifies frame/annotation alignment)
- **Option B** — MediaPipe inner-lip polygon crop
- **Option C** — CLAHE on crop
- **Combined B+C** — current pipeline baseline
- **Option D** — Brightness: gamma correction, linear lift, gamma→CLAHE
- **Option E** — Sharpness: unsharp mask at different strengths, CLAHE→Unsharp
- **Full comparison** — all 5 candidates side by side

---

## Dependencies

```bash
pip install ultralytics opencv-python-headless numpy pyyaml
# for the notebook:
pip install jupyter matplotlib
```

Python 3.10+. GPU optional but recommended for training (CUDA auto-detected).

---

## Dataset stats (as of 2026-04-25)

- **31 subjects**, ~11,683 annotated frames across all clips
- Tasks: latR, latL, elev
- Lazarus scores: 0, 25, 50, 100
- Annotation format: per-frame tip (x, y) + no-tongue flags + clip-level Lazarus score
