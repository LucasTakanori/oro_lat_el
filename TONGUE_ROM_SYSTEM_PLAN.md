# OroSense — Tongue ROM Full System Plan

> Automated clinical scoring of tongue range-of-motion (ROM) using the Lazarus (2014)
> scale. Four tasks: protrusion, elevation, lateralization right, lateralization left.
> Target: continuous 0–100 score per task, generated from a structured iPhone recording
> session and processed partially on-device and partially offline.

---

## Table of Contents

1. [Clinical tasks and scoring rubric](#1-clinical-tasks-and-scoring-rubric)
2. [System architecture overview](#2-system-architecture-overview)
3. [Component specifications](#3-component-specifications)
4. [Per-task prediction pipeline](#4-per-task-prediction-pipeline)
5. [Online vs offline split](#5-online-vs-offline-split)
6. [Firestore data model](#6-firestore-data-model)
7. [Offline processing server](#7-offline-processing-server)
8. [iPhone app integration](#8-iphone-app-integration)
9. [Metrics and evaluation](#9-metrics-and-evaluation)
10. [How to run each component](#10-how-to-run-each-component)
11. [Implementation roadmap](#11-implementation-roadmap)

---

## 1. Clinical tasks and scoring rubric

### 1.1 Tasks

| ID | Task | Instruction to patient | Camera |
|----|------|----------------------|--------|
| `prot` | Tongue protrusion | "Stick your tongue out as far as possible" | Front TrueDepth |
| `elev` | Tongue elevation | "Open your mouth and lift your tongue tip to the roof of your mouth" | Front |
| `latR` | Lateralization right | "Touch the right corner of your mouth with your tongue tip" | Front |
| `latL` | Lateralization left | "Touch the left corner of your mouth with your tongue tip" | Front |

All tasks recorded from the front-facing iPhone camera (same position). No repositioning between tasks.

### 1.2 Lazarus (2014) scoring rubric

**Lateralization R / L:**

| Score | Label | Clinical criterion |
|-------|-------|--------------------|
| 100 | Normal | Tongue tip touches the commissure |
| 50 | Mild–moderate | < 50% reduction in movement toward commissure |
| 25 | Severe | > 50% reduction in movement toward commissure |
| 0 | Total | No tongue movement |

**Elevation:**

| Score | Label | Clinical criterion |
|-------|-------|--------------------|
| 100 | Normal | Tongue tip contacts upper alveolar ridge |
| 50 | Moderate | Visible elevation but no ridge contact |
| 0 | Severe | No visible elevation |

**Protrusion (proposed, Lazarus-compatible):**

| Score | Label | Clinical criterion |
|-------|-------|--------------------|
| 100 | Normal | Tongue tip extends clearly past the lip line |
| 50 | Moderate | Tongue tip reaches lip line but does not protrude |
| 0 | Severe | No anterior tongue movement |

### 1.3 Continuous scoring

All four ordinal levels are expanded to continuous 0–100 predictions. The model outputs a real-valued score; the ordinal category can be recovered by thresholding (0–12→0, 12–37→25, 37–75→50, 75–100→100 for lat; adapt per task).

---

## 2. System architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        iPhone App (Swift)                        │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  ARKit       │  │  Apple       │  │  CoreML              │  │
│  │  TrueDepth   │  │  Vision /    │  │  YOLOv8n-pose        │  │
│  │  depth map   │  │  MediaPipe   │  │  (clahe_unsharp_v2)  │  │
│  │  per frame   │  │  landmarks   │  │  tip + commissures   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └──────────────────┴──────────────────────┘             │
│                            │  per-frame data                     │
│                   ┌────────▼────────┐                            │
│                   │  Online scorer  │  raw geometric score        │
│                   │  (Swift)        │  live UI feedback           │
│                   └────────┬────────┘                            │
│                            │                                     │
│                   ┌────────▼────────┐                            │
│                   │  Firebase SDK   │  upload video + metadata    │
│                   └────────┬────────┘                            │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTPS / Firestore SDK
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Firebase / Google Cloud                      │
│                                                                   │
│  Firestore DB          Firebase Storage        Cloud Functions   │
│  patients/             videos/*.webm           onVideoUploaded   │
│  sessions/             landmarks/*.json        → trigger offline │
│  recordings/           depthMaps/*.bin         processing        │
│  scores/                                                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ trigger / poll
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Offline Processing Server (Python)             │
│                                                                   │
│  1. Download video + landmarks + depth from Storage              │
│  2. CLAHE + unsharp preprocessing per frame                      │
│  3. YOLOv8n-pose inference (clahe_unsharp_v2 weights)            │
│  4. Resting calibration (IC, MC, depth_ref)                      │
│  5. Per-task clip aggregation → continuous score 0–100           │
│  6. Quality checks                                               │
│  7. Write final scores → Firestore scores/                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component specifications

### 3.1 YOLO model

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n-pose |
| Keypoints | 3: left commissure (CL), right commissure (CR), tongue tip (T) |
| `kpt_shape` | `[3, 3]` (x, y, visibility) |
| Input size | 640×640 |
| Training pipeline | CLAHE (clipLimit=3, tile=4×4) → unsharp mask (strength=1.5, k=5) |
| Weights (best) | `runs/tip/loso_Miquel_clahe_unsharp_v2/weights/best.pt` |
| Full LOSO weights | `runs/tip/loso_<subject>/weights/best.pt` (one per fold) |
| Inference speed | ~5ms on iPhone 15 Pro via CoreML (estimated) |
| On-device format | CoreML `.mlpackage` (export from best.pt) |

Keypoint index mapping:
- `kp[0]` = CL (MediaPipe landmark 61, right commissure from patient perspective)
- `kp[1]` = CR (MediaPipe landmark 291, left commissure)
- `kp[2]` = tongue tip T

Tip is considered visible when predicted visibility confidence `kp_conf[2] ≥ 0.5`.

### 3.2 Preprocessing pipeline (offline only)

```
raw frame (BGR)
  → CLAHE on LAB L-channel (clipLimit=3.0, tileGridSize=4×4)
  → unsharp mask (strength=1.5, kernel=5×5 Gaussian)
  → YOLOv8n-pose input
```

Online (on iPhone) the model receives raw frames — preprocessing is skipped for speed. The online score is a rough estimate; the offline processed score is the clinical output.

### 3.3 TrueDepth (protrusion task only)

- ARKit `ARFaceTrackingConfiguration` gives per-pixel depth via `ARDepthData`
- Depth in metres at pixel `(px, py)`: `depthMap[py, px]`
- Reference depth `depth_ref` = depth at upper lip landmark pixel (MediaPipe UL = landmark 13)
- Tip depth `depth_tip` = depth at YOLO-predicted tip pixel
- Protrusion in mm: `prot_mm = (depth_ref − depth_tip) × 1000`
- Positive = tongue closer to camera than lip = protruding

### 3.4 Landmark reference geometry

Computed from MediaPipe face-mesh landmarks each frame:

```
CL  = landmark 61   (right commissure, patient POV)
CR  = landmark 291  (left commissure)
UL  = landmark 13   (upper lip midpoint)
LL  = landmark 14   (lower lip midpoint)
MC  = (CL + CR) / 2  (mouth centre)
IC  = |CR − CL|      (inter-commissure distance, pixels)
MH  = |LL − UL|      (mouth opening height, pixels)
```

IC is used as the normalisation denominator for lateral displacement scores.
IC_mm ≈ 50 mm (population average) → pixel-to-mm scale = 50 / IC_px.

---

## 4. Per-task prediction pipeline

### 4.1 Protrusion (`prot`)

**Goal:** How far past the lip line does the tongue extend?

```
Input : per-frame {tip_xy, tip_vis_conf, depth_tip, depth_ref, IC_px}

Per frame:
  if tip_vis_conf < 0.5 → skip frame
  prot_mm    = (depth_ref − depth_tip) × 1000
  IC_mm      = 50.0   (or calibrated from known reference)
  prot_norm  = prot_mm / (0.5 × IC_mm)   → normalised [0, 1+]

Clip aggregation:
  score_raw  = percentile(prot_norm_valid_frames, 90) × 100
  score      = clip(score_raw, 0, 100)
```

Lazarus threshold recovery: score ≥ 75 → 100, 25–75 → 50, < 25 → 0.

**Resting calibration:** first 5 frames (tongue at rest, mouth closed) → `depth_rest`. Protrusion computed relative to this baseline, not absolute depth.

### 4.2 Elevation (`elev`)

**Goal:** How high does the tongue tip rise? Does it contact the alveolar ridge?

```
Input : per-frame {tip_xy, tip_vis_conf, CL_xy, CR_xy, UL_y, LL_y, MH}

Per frame:
  if tip_vis_conf < 0.5 and MH > 0.05 × IC:
    → palate_contact = True   (tip hidden with open mouth = contact)
    → frame_score = 100
  elif tip_vis_conf ≥ 0.5:
    elev_norm = (LL_y − tip_y) / max(MH, 1)   (0 at lower lip, 1 at upper)
    frame_score = clip(elev_norm, 0, 1) × 50   (visible tip caps at 50)
  else:
    → skip (mouth closed, no signal)

Clip aggregation:
  if any frame_score == 100 → score = 100
  else → score = mean(valid_frame_scores)
```

### 4.3 Lateralization R (`latR`) and L (`latL`)

**Goal:** How far toward the target commissure does the tip move?

```
Input : per-frame {tip_xy, tip_vis_conf, CL_xy, CR_xy, MC_xy, IC}

sign = +1 for latR (tip moves toward CL), −1 for latL (tip moves toward CR)

Per frame:
  if tip_vis_conf < 0.5 → skip
  disp       = sign × (tip_x − MC_x)
  disp_norm  = disp / (IC / 2)              → 0 at centre, 1 at commissure
  frame_score = clip(disp_norm, 0, 1) × 100

Clip aggregation:
  score = percentile(valid_frame_scores, 80)
```

### 4.4 Resting calibration (all tasks)

First 3–5 frames of each clip with tongue at rest:
- Compute `IC_rest`, `MC_rest`, `depth_ref_rest`
- Use these as per-clip reference values (cancels head-pose drift and subject variability)
- If resting frames cannot be detected (face not stable), fall back to first frame

---

## 5. Online vs offline split

### 5.1 Online — computed on iPhone during/after recording

| Computation | Purpose | Latency |
|-------------|---------|---------|
| ARKit face tracking + TrueDepth depth | Raw depth map per frame | Hardware, ~30fps |
| Apple Vision / MediaPipe landmarks | CL, CR, UL, LL per frame | ~3ms/frame |
| CoreML YOLOv8n-pose (raw frames, no preprocessing) | Tip xy + visibility per frame | ~5ms/frame |
| Per-frame raw score (rough) | Live feedback bar in UI | <1ms |
| Quality gates (face centred, enough visible frames, head stable) | Reject bad recordings | <1ms |
| Store per-frame metadata JSON (landmarks + raw tip + depth) | Upload to Firestore | async |
| Preliminary clip score (raw, unprocessed) | Shown to clinician immediately | <100ms |

The **online score** is shown right after recording ends. It is computed from raw (non-preprocessed) YOLO predictions on device. It gives the clinician an immediate indication but is not the clinical ground truth.

### 5.2 Offline — computed on server after upload

| Computation | Purpose | Latency |
|-------------|---------|---------|
| Download video + metadata from Firebase Storage | — | network |
| CLAHE + unsharp mask per frame | Better tip localisation in dark/low-contrast frames | ~10ms/frame CPU |
| YOLOv8n-pose inference (Python, preprocessed frames) | Precise tip + commissure predictions | ~5ms/frame GPU |
| Resting frame detection and calibration | Per-clip IC, MC, depth_ref | per clip |
| Clip aggregation (percentile, mean, palate-contact) | Final LROM score per task | per clip |
| Tip visibility cleanup (remove isolated single-frame detections) | Reduce false positives | per clip |
| Cross-task consistency check | Flag implausible results (e.g. lat score > prot score for severe patient) | per session |
| Longitudinal delta (if prior session exists) | Track improvement/decline | per patient |
| Write final scores → Firestore `scores/` | Clinician dashboard | — |

The **offline score** is the clinical output. The app polls Firestore for `status == "scored"` and updates the display when available (typically within 1–2 minutes of upload).

### 5.3 Decision table

| Signal | Online | Offline | Reason |
|--------|--------|---------|--------|
| Raw tip xy per frame | ✓ | ✓ | Online: UI feedback; Offline: aggregation |
| Preprocessed tip xy | ✗ | ✓ | CLAHE adds ~10ms/frame, too slow for 30fps |
| Depth at tip pixel | ✓ | reuse uploaded | TrueDepth only on device |
| Commissure positions | ✓ | ✓ | Same model output |
| Clip-level score | rough | final | Full clip needed for percentile |
| Palate contact flag | ✓ | ✓ | Simple visibility + MH gate |
| Quality rejection | ✓ | ✓ | Online rejects during recording; offline validates |

---

## 6. Firestore data model

### 6.1 Collections

```
patients/{patientId}
  name: string
  dob: timestamp
  clinician: string
  createdAt: timestamp

sessions/{sessionId}
  patientId: ref → patients/{patientId}
  clinicianId: string
  recordedAt: timestamp
  deviceModel: string          # "iPhone 15 Pro"
  appVersion: string
  status: "recording" | "uploaded" | "processing" | "scored" | "error"
  tasks: ["prot", "elev", "latR", "latL"]

recordings/{recordingId}
  sessionId: ref → sessions/{sessionId}
  patientId: string
  task: "prot" | "elev" | "latR" | "latL"
  lazarusScore: int | null     # clinician-set during recording (ground truth)
  videoPath: string            # Firebase Storage path: videos/{recordingId}.mp4
  landmarksPath: string        # Storage: landmarks/{recordingId}.json
  depthPath: string | null     # Storage: depth/{recordingId}.bin  (prot only)
  frameCount: int
  fps: float
  resolutionW: int
  resolutionH: int
  uploadedAt: timestamp
  onlineScore: float | null    # set by app immediately after recording
  onlineScoreAt: timestamp | null

scores/{scoreId}
  recordingId: ref → recordings/{recordingId}
  sessionId: string
  patientId: string
  task: "prot" | "elev" | "latR" | "latL"
  scoreRaw: float              # continuous 0–100 (offline model output)
  scoreOrdinal: int            # 0 / 25 / 50 / 100 (thresholded)
  pctValidFrames: float        # fraction of frames with visible tip
  qualityFlags: [string]       # e.g. ["low_visible_frames", "head_unstable"]
  modelVersion: string         # e.g. "clahe_unsharp_v2"
  computedAt: timestamp
  pipeline: "offline_v1"
```

### 6.2 Firebase Storage layout

```
videos/
  {recordingId}.mp4
landmarks/
  {recordingId}.json           # per-frame MediaPipe landmarks
depth/
  {recordingId}.bin            # float32 depth map per frame (prot only)
exports/
  {sessionId}_report.pdf       # generated after scoring
```

### 6.3 Firestore security rules (outline)

```
patients: read/write if clinician owns patient or is admin
sessions: read/write if clinician owns session
recordings: write on create (app), read if owner clinician
scores: write only from service account (offline server), read if owner
```

### 6.4 Cloud Function trigger

```javascript
// functions/index.js
exports.onRecordingUploaded = functions.firestore
  .document("recordings/{recordingId}")
  .onUpdate((change, context) => {
    const after = change.after.data();
    // trigger when all tasks of a session are uploaded
    if (after.status === "uploaded") {
      // publish to Pub/Sub → offline processing server
      return pubsub.topic("process-recording")
        .publish(Buffer.from(JSON.stringify({
          recordingId: context.params.recordingId,
          ...after
        })));
    }
  });
```

---

## 7. Offline processing server

### 7.1 Script: `poc/score_recording.py`

```
Usage:
  python -m poc.score_recording --recording-id <id>   # single
  python -m poc.score_recording --session-id <id>     # full session
  python -m poc.score_recording --poll                # daemon: poll Firestore queue
```

### 7.2 Processing steps

```python
def score_recording(recording_id: str) -> dict:
    # 1. Fetch metadata from Firestore
    rec = firestore.get("recordings", recording_id)

    # 2. Download assets from Firebase Storage
    video_path     = download(rec["videoPath"])
    landmarks_path = download(rec["landmarksPath"])
    depth_path     = download(rec["depthPath"]) if rec["task"] == "prot" else None

    # 3. Decode video frames sequentially (webm/mp4)
    frames = decode_frames(video_path)                     # list of BGR np arrays

    # 4. Load landmarks
    lm_frames = json.load(landmarks_path)["landmarks"]    # per-frame MediaPipe lm

    # 5. Preprocess frames
    proc_frames = [pipeline_clahe_unsharp(f) for f in frames]

    # 6. Run YOLO on preprocessed frames
    model   = YOLO("runs/tip/loso_{val}/weights/best.pt")
    preds   = [model(f, verbose=False) for f in proc_frames]

    # 7. Extract per-frame predictions
    tip_preds = extract_tip_predictions(preds)  # list of {xy, vis_conf} or None

    # 8. Resting calibration from first 5 frames
    ref = calibrate_resting(lm_frames[:5])      # IC, MC, depth_ref

    # 9. Load depth map (prot only)
    depth_frames = load_depth(depth_path) if depth_path else None

    # 10. Compute clip-level score
    score = compute_score(rec["task"], tip_preds, lm_frames, ref, depth_frames)

    # 11. Quality checks
    flags = quality_check(tip_preds, lm_frames)

    # 12. Write to Firestore
    firestore.set("scores", new_id(), {
        "recordingId": recording_id,
        "scoreRaw": score.value,
        "scoreOrdinal": score.ordinal,
        "pctValidFrames": score.pct_valid,
        "qualityFlags": flags,
        "modelVersion": "clahe_unsharp_v2",
        "computedAt": now(),
    })

    # 13. Update recording status
    firestore.update("recordings", recording_id, {"status": "scored"})
```

### 7.3 Quality flags

| Flag | Condition | Action |
|------|-----------|--------|
| `low_visible_frames` | < 20% frames with visible tip | warn clinician |
| `head_unstable` | IC variance > 15% of mean IC | warn |
| `no_tongue_detected` | 0 frames with visible tip | score = 0, flag |
| `short_clip` | < 30 frames | warn |
| `depth_missing` | prot task but no depth map | fall back to 2D-only estimate |
| `landmark_dropout` | > 10% frames with no face detected | warn |

---

## 8. iPhone app integration

### 8.1 Recording session flow

```
Session start
  └─ [rest capture] 3s eyes-open face still → calibration frames
      └─ [task 1: prot]  "Stick your tongue out"
          └─ 2s hold → clip saved
      └─ [task 2: elev]  "Lift tongue to roof of mouth"
          └─ 2s hold → clip saved
      └─ [task 3: latR]  "Touch right corner"
          └─ 2s hold → clip saved
      └─ [task 4: latL]  "Touch left corner"
          └─ 2s hold → clip saved
  └─ Upload all 4 clips + metadata → Firestore
  └─ Show preliminary scores (online)
  └─ Poll for final scores (offline, ~1–2 min)
```

### 8.2 Per-task recording screen

```
┌─────────────────────────────────────────────┐
│          [Live camera feed]                  │
│                                              │
│  ● CL commissure dot (orange)                │
│  ● CR commissure dot (orange)                │
│  ● Tongue tip dot (red, only when detected)  │
│  ● Live score bar: ████░░░░ 62/100           │
│                                              │
│  "Move your tongue further right →"          │
│           [Hold bar: ████░░ 1.3s / 2.0s]    │
└─────────────────────────────────────────────┘
```

Live feedback text rules:
- Protrusion: "Good! Keep going" / "Try to extend further"
- Elevation: "Tongue detected — lift higher" / "Touch the roof"
- Lat R/L: "Move further right/left" / "Tongue at corner ✓"

### 8.3 CoreML model export

```bash
# Export best.pt → CoreML for iPhone
python - <<'EOF'
from ultralytics import YOLO
model = YOLO("runs/tip/loso_Miquel_clahe_unsharp_v2/weights/best.pt")
model.export(format="coreml", imgsz=640, nms=True)
# outputs: loso_Miquel_clahe_unsharp_v2/weights/best.mlpackage
EOF
```

Add `best.mlpackage` to the Xcode project. Call it via:
```swift
let model = try! TongueROMModel(configuration: MLModelConfiguration())
let input = TongueROMModelInput(image: pixelBuffer)
let output = try! model.prediction(input: input)
// output.keypoints: [CL_x, CL_y, CL_conf, CR_x, CR_y, CR_conf, T_x, T_y, T_conf]
```

### 8.4 TrueDepth depth extraction (protrusion)

```swift
// ARKit session
let config = ARFaceTrackingConfiguration()
config.frameSemantics = [.sceneDepth]
arSession.run(config)

// Per-frame in ARSessionDelegate
func session(_ session: ARSession, didUpdate frame: ARFrame) {
    guard let depthData = frame.sceneDepth else { return }
    let depthMap = depthData.depthMap   // CVPixelBuffer, Float32, metres
    
    // Sample depth at YOLO-predicted tip pixel
    let tipDepth = sampleDepth(depthMap, at: tipPixel)
    let refDepth = sampleDepth(depthMap, at: ulLandmarkPixel)
    let protMM   = (refDepth - tipDepth) * 1000.0
}
```

Depth map serialization for upload: convert `CVPixelBuffer` → `[Float32]` → write as `.bin` file → upload to Firebase Storage.

### 8.5 Data upload

```swift
func uploadRecording(_ rec: RecordingData) async throws {
    // 1. Upload video
    let videoRef = storage.reference().child("videos/\(rec.id).mp4")
    try await videoRef.putFileAsync(from: rec.videoURL)

    // 2. Upload landmarks JSON
    let lmRef = storage.reference().child("landmarks/\(rec.id).json")
    try await lmRef.putDataAsync(rec.landmarksJSON)

    // 3. Upload depth (prot only)
    if rec.task == "prot", let depth = rec.depthData {
        let depthRef = storage.reference().child("depth/\(rec.id).bin")
        try await depthRef.putDataAsync(depth)
    }

    // 4. Write Firestore document
    try await db.collection("recordings").document(rec.id).setData([
        "sessionId":     rec.sessionId,
        "patientId":     rec.patientId,
        "task":          rec.task,
        "lazarusScore":  rec.clinicianScore,
        "videoPath":     "videos/\(rec.id).mp4",
        "landmarksPath": "landmarks/\(rec.id).json",
        "depthPath":     rec.task == "prot" ? "depth/\(rec.id).bin" : nil,
        "frameCount":    rec.frameCount,
        "fps":           rec.fps,
        "onlineScore":   rec.onlineScore,
        "status":        "uploaded",
        "uploadedAt":    FieldValue.serverTimestamp(),
    ])
}
```

---

## 9. Metrics and evaluation

### 9.1 Per-task model metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Tip MAE (px) | Mean absolute error of predicted tip vs annotated tip | < 10px |
| Tip MAE (norm) | MAE normalised by IC | < 0.08 |
| Tip vis F1 | F1 score of predicted tip visibility vs ground truth | > 0.90 |
| Pose mAP50-95 | YOLO standard keypoint metric (all 3 kpts) | > 0.70 |
| LROM MAE | Mean absolute error of final 0–100 score vs clinician label | < 12 |
| LROM q-kappa | Quadratic-weighted kappa vs ordinal Lazarus labels | > 0.70 |

### 9.2 LOSO cross-validation

One subject held out per fold. Final reported metrics are mean ± std across all 30 folds.

```bash
# After full LOSO training:
python -m poc.evaluate_loso   # reads runs/tip/loso_*/  → prints table
```

### 9.3 Calibration metrics (protrusion)

| Metric | Description |
|--------|-------------|
| Depth MAE | Error in estimated protrusion mm vs ground truth (phantom / ruler) |
| IC scale error | Error in pixel→mm conversion using IC=50mm assumption |
| Resting drift | Depth_ref variance across rest frames (should be < 1mm) |

### 9.4 Quality thresholds for clinical use

| Threshold | Value | Action on failure |
|-----------|-------|-------------------|
| Min visible frames | 20% of clip | Ask to re-record |
| Max head translation | IC drift < 15% | Warn, still score |
| Min clip duration | 30 frames (~1s at 30fps) | Reject |
| Min face detection | 90% frames | Warn |
| Tip confidence gate | kp_conf[2] ≥ 0.5 | Skip frame |

---

## 10. How to run each component

### 10.1 Data collection (current browser app)

```bash
# Open index.html in Chrome — no server needed
open index.html
# Follow on-screen instructions
# Clips saved to data/<subject>/
```

### 10.2 Annotation tool

```bash
python -m poc.annotator
# open http://localhost:8765
# Controls: left-click = tip, right-click = no-tongue, drag = paint across frames
```

### 10.3 Build dataset (with preprocessing pipeline)

```bash
# CLAHE + unsharp (best pipeline)
python -m poc.build_tip_dataset --pipeline clahe_unsharp
# outputs: dataset_clahe_unsharp/images/, labels/, tip_loso_*.yaml

# All pipelines:
python -m poc.build_tip_dataset --pipeline clahe
python -m poc.build_tip_dataset --pipeline gamma_clahe_unsharp
```

### 10.4 Train — single LOSO fold

```bash
python -m poc.train_tip \
  --val-subject Miquel \
  --run-name loso_Miquel_clahe_unsharp_v2 \
  --dataset-dir dataset_clahe_unsharp \
  --epochs 20
# weights → runs/tip/loso_Miquel_clahe_unsharp_v2/weights/best.pt
```

### 10.5 Train — full LOSO (all subjects)

```bash
python -m poc.train_tip \
  --dataset-dir dataset_clahe_unsharp \
  --epochs 20
# trains 30 sequential folds, one per subject
# each fold → runs/tip/loso_<subject>/weights/best.pt
```

### 10.6 Render validation videos

```bash
# All clips for one subject with one model
python -m poc.render_val_video --subject Miquel --fold loso_Miquel_clahe_unsharp_v2

# Single clip
python -m poc.render_val_video \
  --subject Miquel \
  --clip latR_2026-04-23T23-51-10-466Z_s100 \
  --fold loso_Miquel_clahe_unsharp_v2

# Output: poc/out/val_videos/loso_Miquel_clahe_unsharp_v2/*.mp4
```

### 10.7 Export CoreML model for iPhone

```bash
python - <<'EOF'
from ultralytics import YOLO
model = YOLO("runs/tip/loso_Miquel_clahe_unsharp_v2/weights/best.pt")
model.export(format="coreml", imgsz=640, nms=True)
EOF
# output: runs/tip/loso_Miquel_clahe_unsharp_v2/weights/best.mlpackage
```

### 10.8 Offline score a recording (future)

```bash
# Single recording
python -m poc.score_recording --recording-id <firestore-id>

# Full session
python -m poc.score_recording --session-id <firestore-id>

# Daemon mode (poll queue)
python -m poc.score_recording --poll
```

---

## 11. Implementation roadmap

### Phase 1 — Scoring pipeline (Python, now) ✓ partial

| Task | Status | Script |
|------|--------|--------|
| YOLOv8n-pose training (clahe_unsharp) | ✓ done | `train_tip.py` |
| Full LOSO training (30 folds) | 🔄 running | `train_tip.py` |
| Lateralization scoring formula | ✓ in `scoring.py` | — |
| Elevation scoring formula | ✓ in `scoring.py` | — |
| Protrusion scoring formula | ✗ not yet | add to `scoring.py` |
| Resting calibration | ✗ not yet | add to `scoring.py` |
| Clip aggregator (Python) | ✗ not yet | `poc/score_recording.py` |
| LOSO evaluation script | ✗ not yet | `poc/evaluate_loso.py` |

### Phase 2 — Firebase backend

| Task | Status |
|------|--------|
| Firebase project setup | ✗ |
| Firestore collections + rules | ✗ |
| Firebase Storage buckets | ✗ |
| Cloud Function trigger (onUpload → queue) | ✗ |
| Offline server: download + process + write | ✗ |

### Phase 3 — iPhone app

| Task | Status |
|------|--------|
| CoreML export of best.pt | ✗ |
| ARKit TrueDepth session | ✗ |
| Per-frame YOLO inference (CoreML) | ✗ |
| Live score bar UI | ✗ |
| 2s hold + clip save | partial (existing app) |
| Firebase SDK upload | ✗ |
| Final score polling + display | ✗ |

### Phase 4 — Validation and clinical use

| Task | Status |
|------|--------|
| Protrusion depth calibration study | ✗ |
| LOSO evaluation table (all 30 folds) | ✗ (pending training) |
| Clinician blind comparison study | ✗ |
| IRB / ethics clearance for patient use | ✗ |
