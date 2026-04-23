# Tongue-tip tracker — annotation guide + post-annotation plan

Priority: train per-frame model that outputs tongue tip `(x, y)` from a single image.

---

## 1. Annotate clips (current step)

Tool: `.venv/bin/python -m poc.annotator` → http://127.0.0.1:8765

### What to label

For every clip:
- **Tongue tip `(x, y)`** on every frame where the tongue is visible.
- **`NO TONGUE`** on every frame where the tongue is not visible (mouth closed, no protrusion, occluded).
- **Clip grade 0–100** (continuous LROM score) — slider in right panel.

### How to label efficiently

| Action | Input |
|---|---|
| Place tip on current frame + auto-advance | left-click |
| Hold-drag tip across frames | left-click + drag |
| Mark current frame as NO TONGUE + auto-advance | right-click |
| Hold-drag NO TONGUE across frames | right-click + drag |
| Toggle NO TONGUE on current frame | `N` |
| Apply tip / NO TONGUE to a range of frames | switch to `range`, `[` / `]` to mark, then click |

### Range mode + interpolation

- In **range mode**, clicking once writes the same `(x, y)` to every frame in the range. The server then **linearly interpolates between consecutive tip anchors at save time**, so the saved JSON contains a dense per-frame tip when there is no NO TONGUE frame in the gap.
- Workflow: place anchor tips on key frames (peak, midpoints, rest); the gap fills automatically on save.
- Interpolation **stops at NO TONGUE frames** — it never bridges across a frame the user marked invisible.
- Range mode is also useful for marking long NO TONGUE spans (right-click in range mode).

### Output format

`poc/out/annotations/<subject>_<stem>.json`:
```json
{
  "subject": "01",
  "stem": "latR_2026-04-14T22-45-45-045Z_s50",
  "task": "latR",
  "lazarus_score": 50,
  "grade": 45,
  "tips": { "0": [x, y], "1": [x, y], ... },     // dense after save (interpolated)
  "no_tongue": { "12": true, "13": true, ... }   // explicit invisibility
}
```

### Coverage targets

- Every clip annotated end-to-end: every frame is either in `tips` (visible) or `no_tongue` (invisible).
- ≥ 3 subjects fully annotated before training (LOSO needs hold-out subject).
- All three tasks (`latR`, `latL`, `elev`) per subject.

---

## 2. Build training dataset (next step, future)

Script to write: `poc/build_tip_dataset.py`.

Per annotation file, per frame:
1. **Decode frame** from `data/<subject>/<stem>.webm` at index `f`.
2. **Skip** frames not in `tips` and not in `no_tongue` (unannotated → no ground truth).
3. **Bbox** auto-derived from MediaPipe inner-lip polygon, extended ±0.6·IC laterally for `lat*` tasks (already in `poc/scoring.py`: `_extend_polygon_horizontally`, `_inner_lip_polygon`).
4. Write image `dataset/images/<split>/<subject>_<stem>_<f>.jpg`.
5. Write label `dataset/labels/<split>/<subject>_<stem>_<f>.txt`:
   - YOLO-pose format (single class, single keypoint):
     ```
     0  cx cy w h  tip_x tip_y vis
     ```
   - All coords normalized to image W/H.
   - `vis = 2` if frame in `tips`, `vis = 0` (and tip_x = tip_y = 0) if in `no_tongue`.
6. **Split**: LOSO — one subject reserved for val each fold.

---

## 3. Train tracker (next step, future)

`YOLOv8n-pose` or `YOLOv8s-pose` via ultralytics:
- `kpt_shape: [1, 3]` (one keypoint, three values: x, y, vis)
- `nc: 1`, `names: [tongue]`
- Pretrained weights: `yolov8n-pose.pt`
- Augmentations: keep default (HSV, flip-lr off — LROM tasks are laterality-specific).
- Train per LOSO fold: ~20 epochs on RTX should converge.

Eval per fold:
- Median per-frame xy error (pixels, normalized to inter-commissure distance).
- Visibility classification F1 (predicted bbox conf × kpt vis vs `no_tongue`).
- Downstream: feed predicted tip into existing `score_clip` (`poc/scoring.py`) → per-clip LROM score → MAE vs `lazarus_score` and quadratic-weighted κ.

---

## 4. Decisions deferred until after annotation

- Whether to also train an explicit visibility head (currently absorbed into kpt vis).
- Whether to add per-frame bbox manual annotation (currently auto-derived from MediaPipe — risk if MediaPipe drops face on extreme protrusion).
- Whether to fine-tune SAM2 instead of training YOLOv8-pose (not planned; SAM2 already works fine as a prompt-driven segmenter — the annotator tip is the prompt).
