# Tongue ROM Data Collection Demo — Implementation Spec

> **Context:** This is a browser-based data collection tool for a research project on
> tongue range of motion (ROM) in oral cancer patients (Lazarus et al., 2014 scale).
> It captures face mesh landmarks + short video clips of healthy subjects performing
> four tongue tasks to build a labeled dataset. Scoring (0 / 25 / 50 / 100) is applied
> offline by clinicians; this tool only captures data at peak pose.

---

## 1. Deliverable

A **single self-contained HTML file** (`tongue_rom_demo.html`) that runs in any modern
browser (Chrome 113+, Edge 113+) with no build step or server. Works on desktop and
mobile (iOS Safari 16.4+, Chrome Android). All dependencies loaded from CDN.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Face landmarks | `@mediapipe/tasks-vision` (latest) | 478-point 3D face mesh, WebGPU delegate |
| Tongue detection | Canvas HSV pixel heuristic on mouth ROI | No extra model needed for MVP |
| Video capture | `MediaRecorder` API | Built-in, outputs WebM |
| Styling | Plain CSS with CSS variables | Zero dependencies |
| Storage | Auto-download via anchor click | No server, no auth |

**No framework** (no React, no Vue). Plain HTML + vanilla JS + CSS.

---

## 3. MediaPipe Setup

```javascript
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const filesetResolver = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);

const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    delegate: "GPU"          // falls back to CPU automatically
  },
  runningMode: "VIDEO",
  numFaces: 1,
  minFaceDetectionConfidence: 0.5,
  minFacePresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
  outputFaceBlendshapes: false,
  outputFacialTransformationMatrixes: false
});
```

Call `faceLandmarker.detectForVideo(videoEl, Date.now())` on every animation frame.
The result gives `result.faceLandmarks[0]` — an array of 478 objects `{x, y, z}` where
x and y are **normalized** (0–1 relative to image width/height) and z is depth in the
same scale as x (camera-relative, smaller = closer).

---

## 4. Key Landmark Indices

These are the only landmarks needed for task scoring:

```javascript
const LM = {
  // Mouth corners (commissures)
  R_COMMISSURE: 61,
  L_COMMISSURE: 291,

  // Lip midpoints
  UPPER_LIP_CENTER: 13,   // inner upper lip
  LOWER_LIP_CENTER: 14,   // inner lower lip

  // Upper lip outer rim (used as alveolar ridge proxy for elevation)
  UPPER_LIP_TOP: 0,       // philtrum base / uppermost point of lip

  // Outer lip boundary corners (for mouth ROI bounding box)
  LIP_LEFT_OUTER: 78,
  LIP_RIGHT_OUTER: 308,

  // Nose tip (stable vertical reference)
  NOSE_TIP: 1,

  // Chin (stable vertical reference below)
  CHIN: 152
};
```

---

## 5. Tongue Tip Detection

MediaPipe does not output tongue landmarks. Use a **pixel-based heuristic** on the
mouth ROI defined by the face mesh.

### 5.1 Extract mouth ROI

```javascript
function getMouthROI(landmarks, canvas) {
  const W = canvas.width, H = canvas.height;
  // Use commissure + lip center landmarks to define bounding box
  const xs = [61, 291, 13, 14, 78, 308].map(i => landmarks[i].x * W);
  const ys = [61, 291, 13, 14, 78, 308].map(i => landmarks[i].y * H);
  const pad = 20; // px
  return {
    x: Math.max(0, Math.min(...xs) - pad),
    y: Math.max(0, Math.min(...ys) - pad),
    w: Math.min(W, Math.max(...xs) + pad) - Math.max(0, Math.min(...xs) - pad),
    h: Math.min(H, Math.max(...ys) + pad) - Math.max(0, Math.min(...ys) - pad)
  };
}
```

### 5.2 Tongue color mask (HSV)

Convert ROI pixels from RGB to HSV. Tongue tissue has a characteristic pinkish-red hue.

```javascript
// In HSV space (H: 0-360, S: 0-1, V: 0-1)
const TONGUE_MASK = {
  H_MIN: 330,  H_MAX: 360,  // red-pink hue (wraps; also 0-20)
  H_MIN2: 0,   H_MAX2: 20,
  S_MIN: 0.25,
  V_MIN: 0.35,
  V_MAX: 0.90
};
```

**Implementation steps:**
1. Use `ctx.getImageData(roi.x, roi.y, roi.w, roi.h)` on the canvas where the video
   frame is drawn.
2. For each pixel: convert RGBA → HSV.
3. Build a binary mask of tongue-colored pixels.
4. Morphologically erode (3×3 box, 1 pass) to remove noise.
5. Find the **largest connected blob** of masked pixels — this is the tongue region.
6. The tongue tip is the **extreme point** in the task direction:
   - Protrusion: lowest Y point (farthest from nose tip in image = most protruded)
   - Lateralization left: leftmost X point
   - Lateralization right: rightmost X point
   - Elevation: highest Y point (closest to upper lip)

> **Fallback:** If no tongue-colored blob is found (e.g. mouth closed, dark skin tone
> confounding HSV), return `null` and display a soft warning: "Make sure your tongue
> is visible and well-lit."

---

## 6. Task Definitions

Four tasks run sequentially. Each has:
- An instruction screen with text + SVG illustration
- A live capture screen
- A task score (0–1 float, updated every frame)
- A hold threshold and completion handler

### 6.1 Protrusion

**Clinical definition:** Tongue protrudes ≥15 mm past upper-lip margin = score 100.

**Detection:**
```javascript
function scoreProtrusion(landmarks, tongueTipY, canvas) {
  const H = canvas.height;
  const upperLipY = landmarks[LM.UPPER_LIP_TOP].y * H;
  const lowerLipY = landmarks[LM.LOWER_LIP_CENTER].y * H;
  const chinY     = landmarks[LM.CHIN].y * H;

  if (tongueTipY === null) return 0;

  // Tongue tip going below lower lip = protrusion in frontal view
  // Normalize: 0 at lower lip, 1 when tip is one chin-to-lip distance below lower lip
  const range = (chinY - lowerLipY) * 0.5;  // generous normalization
  const raw = (tongueTipY - lowerLipY) / range;
  return Math.max(0, Math.min(1, raw));
}
```

**Hold threshold:** `score > 0.55` for 2 continuous seconds.

### 6.2 Lateralization (Left and Right)

**Clinical definition:** Tongue touches commissure = score 100. <50% reduction = 50.

Run as two separate tasks: `LAT_LEFT` and `LAT_RIGHT`.

```javascript
function scoreLateralization(landmarks, tongueTipX, side, canvas) {
  const W = canvas.width;
  const rCommX = landmarks[LM.R_COMMISSURE].x * W;
  const lCommX = landmarks[LM.L_COMMISSURE].x * W;
  const mouthCenterX = (rCommX + lCommX) / 2;

  if (tongueTipX === null) return 0;

  if (side === 'LEFT') {
    // Left = smaller X in image (camera mirrored: LEFT in patient = right in image)
    // NOTE: mirror logic depends on whether video is flipped. See Section 8.
    const target = lCommX;  // adjust per mirror setting
    const raw = (mouthCenterX - tongueTipX) / (mouthCenterX - target);
    return Math.max(0, Math.min(1, raw));
  } else {
    const target = rCommX;
    const raw = (tongueTipX - mouthCenterX) / (target - mouthCenterX);
    return Math.max(0, Math.min(1, raw));
  }
}
```

**Hold threshold:** `score > 0.80` for 2 continuous seconds.

### 6.3 Elevation

**Clinical definition:** Tongue tip contacts upper alveolar ridge = score 100.
Elevation visible but no contact = score 50. No movement = 0.

```javascript
function scoreElevation(landmarks, tongueTipY, canvas) {
  const H = canvas.height;
  const upperRidgeY = landmarks[LM.UPPER_LIP_CENTER].y * H;  // proxy for alveolar ridge
  const lowerLipY   = landmarks[LM.LOWER_LIP_CENTER].y * H;

  if (tongueTipY === null) return 0;

  // Normalize: 0 when tip at lower lip, 1 when tip at/above upper lip center
  const range = lowerLipY - upperRidgeY;
  const raw = (lowerLipY - tongueTipY) / range;
  return Math.max(0, Math.min(1, raw));
}
```

**Hold threshold:** `score > 0.75` for 2 continuous seconds.

---

## 7. Hold Detection Logic

```javascript
class HoldDetector {
  constructor({ threshold = 0.75, holdMs = 2000, dropThreshold = 0.55 }) {
    this.threshold = threshold;
    this.holdMs = holdMs;
    this.dropThreshold = dropThreshold;
    this.startTime = null;
    this.state = 'IDLE';  // IDLE | HOLDING | COMPLETE
  }

  update(score, nowMs) {
    if (this.state === 'COMPLETE') return { state: 'COMPLETE', progress: 1 };

    if (score >= this.threshold) {
      if (this.startTime === null) this.startTime = nowMs;
      const elapsed = nowMs - this.startTime;
      const progress = Math.min(1, elapsed / this.holdMs);
      if (elapsed >= this.holdMs) {
        this.state = 'COMPLETE';
        return { state: 'COMPLETE', progress: 1 };
      }
      this.state = 'HOLDING';
      return { state: 'HOLDING', progress };
    } else {
      if (score < this.dropThreshold) {
        this.startTime = null;
        this.state = 'IDLE';
      }
      return { state: this.state, progress: this.startTime
        ? Math.min(1, (nowMs - this.startTime) / this.holdMs)
        : 0 };
    }
  }

  reset() {
    this.startTime = null;
    this.state = 'IDLE';
  }
}
```

When `state === 'HOLDING'`, show the progress bar UI (see Section 9.3).
When `state === 'COMPLETE'`, trigger data capture (see Section 8).

---

## 8. Data Capture

### 8.1 Landmark buffer

Accumulate landmarks in a rolling buffer on every frame during live capture:

```javascript
const landmarkBuffer = [];  // max 300 entries (~5s at 60fps)

// Each entry:
{
  t: performance.now(),                  // ms since page load
  lm: result.faceLandmarks[0]           // array of 478 {x,y,z}
}
```

When `HoldDetector` fires `COMPLETE`, extract the **last 120 frames** (~2s):

```javascript
const captureFrames = landmarkBuffer.slice(-120);
```

### 8.2 Video capture

Use `MediaRecorder` with a 5-second rolling buffer strategy:

```javascript
// Start recording as soon as live capture screen opens
const chunks = [];
const recorder = new MediaRecorder(stream, {
  mimeType: 'video/webm;codecs=vp9',
  videoBitsPerSecond: 2_500_000
});
recorder.ondataavailable = e => {
  if (e.data.size > 0) chunks.push(e.data);
  // Keep only last ~5s of chunks
  while (chunks.reduce((a, c) => a + c.size, 0) > 15_000_000) chunks.shift();
};
recorder.start(500);  // 500ms timeslice

// On capture complete: stop, assemble Blob
recorder.stop();
recorder.onstop = () => {
  const videoBlob = new Blob(chunks, { type: 'video/webm' });
  saveData(videoBlob, landmarkCapture);
};
```

### 8.3 Save to disk

Auto-download both files using hidden anchor elements:

```javascript
function saveData(videoBlob, frames) {
  const ts = new Date().toISOString().slice(0, 10);
  const base = `${subjectId}_${currentTask}_${ts}`;

  // Landmarks JSON
  const json = JSON.stringify({
    subject: subjectId,
    task: currentTask,
    capturedAt: new Date().toISOString(),
    sampleRateHz: 60,
    frameCount: frames.length,
    landmarks: frames
  }, null, 0);  // compact, no pretty-print (can be large)

  const jsonBlob = new Blob([json], { type: 'application/json' });
  downloadBlob(jsonBlob, `${base}_landmarks.json`);

  // Video
  downloadBlob(videoBlob, `${base}_clip.webm`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
```

### 8.4 Mirror / laterality note

Display the video **mirrored** (CSS `transform: scaleX(-1)`) so it feels natural like a
mirror. **Store landmark X coordinates as-is** (un-mirrored) from MediaPipe. Document
this in the JSON as `"mirrored": false` so downstream processing knows. For lateralization
scoring, compensate: if video is displayed mirrored, "left" for the subject appears as
"right" in the raw MediaPipe coordinates. The scoring function must flip X accordingly.
Add a metadata field `"cameraFlipped": true` to the JSON.

---

## 9. UI / UX

### 9.1 Screen sequence

```
[Welcome] → [Task 1 Instructions] → [Task 1 Live] → [Task 1 Saved]
         → [Task 2 Instructions] → [Task 2 Live] → [Task 2 Saved]
         → [Task 3 Instructions] → [Task 3 Live] → [Task 3 Saved]
         → [Task 4 Instructions] → [Task 4 Live] → [Task 4 Saved]
         → [All Done / Download Summary]
```

Task order: `PROTRUSION` → `LAT_RIGHT` → `LAT_LEFT` → `ELEVATION`

### 9.2 Welcome screen

- Title: "Tongue ROM Dataset Collection"
- Text input: "Your name or ID" (required)
- "Start" button → begins task sequence
- Brief note: "This tool will guide you through 4 short tongue movement tasks.
  Each task takes about 30 seconds."

### 9.3 Instructions screen (per task)

Show before each live capture. Contains:
- Task name (e.g. "Task 2 of 4 — Lateralization Right")
- A short written instruction (see below)
- An SVG illustration of the task (simple line drawing of a mouth/tongue)
- "I'm ready" button → opens live capture

**Instruction text per task:**

| Task | Instruction |
|---|---|
| Protrusion | "Open your mouth and stick your tongue out as far as possible, straight forward. Hold the maximum position." |
| Lat Right | "Open your mouth and move your tongue as far as possible toward the right corner of your mouth. Hold the maximum position." |
| Lat Left | "Open your mouth and move your tongue as far as possible toward the left corner of your mouth. Hold the maximum position." |
| Elevation | "Open your mouth and lift your tongue tip up to touch the roof of your mouth (just behind your upper teeth). Hold the maximum position." |

### 9.4 Live capture screen

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  Task 2 of 4 — Lateralization Right         [Cancel]│
├─────────────────────────────────────────────────────┤
│                                                     │
│   [Camera feed — mirrored — overlaid with:         ]│
│   - Green face mesh (thin lines)                    │
│   - Red dot at detected tongue tip                  │
│   - Semi-transparent mouth ROI box                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Status bar (appears when HOLDING):                 │
│  "Hold this position..."  [████████░░░░] 1.4s / 2s │
├─────────────────────────────────────────────────────┤
│  Soft guidance:                                     │
│  "Move your tongue further to the right"  (idle)   │
│  "Great — keep holding!"                 (holding)  │
└─────────────────────────────────────────────────────┘
```

**Canvas layout:**
- Use two stacked `<canvas>` elements (z-index layering):
  1. `videoCanvas` — draws the mirrored video frame
  2. `overlayCanvas` — draws face mesh + tongue tip + ROI box
- Both 100% width, max-height 60vh, aspect ratio maintained.

**Face mesh drawing:**
Use `DrawingUtils` from MediaPipe Tasks Vision:
```javascript
const drawingUtils = new DrawingUtils(overlayCtx);
drawingUtils.drawConnectors(
  landmarks,
  FaceLandmarker.FACE_LANDMARKS_TESSELATION,
  { color: "rgba(0,255,0,0.15)", lineWidth: 0.5 }
);
drawingUtils.drawConnectors(
  landmarks,
  FaceLandmarker.FACE_LANDMARKS_LIPS,
  { color: "rgba(0,255,120,0.6)", lineWidth: 1.5 }
);
```

**Tongue tip indicator:**
Draw a 6px filled red circle at the tongue tip pixel coordinates on the overlay canvas.
If no tongue detected: draw a pulsing orange question-mark glyph at mouth center.

**Progress bar:**
- Only visible when `HoldDetector.state === 'HOLDING'`
- CSS transition for smooth fill animation
- Color: green (#22c55e), transitions to a brighter flash on completion
- Show elapsed seconds: "1.4s / 2.0s"

**Guidance text logic:**
```javascript
function getGuidanceText(task, score, state) {
  if (state === 'HOLDING') return "Great — keep holding!";
  if (score < 0.3) {
    const msgs = {
      PROTRUSION: "Stick your tongue out further",
      LAT_RIGHT:  "Move your tongue further to the right",
      LAT_LEFT:   "Move your tongue further to the left",
      ELEVATION:  "Lift your tongue tip higher"
    };
    return msgs[task];
  }
  if (score < 0.7) return "Almost there — push a bit further";
  return "Excellent — hold that position!";
}
```

### 9.5 Saved screen

- Large green checkmark
- "✓ Task 2 saved — files downloading..."
- Show filename: `{subjectId}_latR_2026-04-13_landmarks.json`
- "Next task" button → goes to Instructions for task 3

### 9.6 All Done screen

- "All 4 tasks complete!"
- List of downloaded files
- "Start over with new participant" button (reloads page)
- Optional: "Download summary CSV" with one row per task:
  `subject, task, timestamp, frame_count, peak_score`

---

## 10. Performance Requirements

- Target 30fps minimum on mobile (reduce to 15fps detect interval on low-end).
- Detection should run on every animation frame:
  ```javascript
  function onFrame(nowMs) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const result = faceLandmarker.detectForVideo(video, nowMs);
    if (result.faceLandmarks.length > 0) processLandmarks(result.faceLandmarks[0]);
    requestAnimationFrame(onFrame);
  }
  ```
- Do **not** run `getImageData` on every frame — throttle tongue pixel detection to
  every 3rd frame (every ~50ms) to reduce jank. Landmark processing runs every frame.
- Use `offscreen = document.createElement('canvas')` at reduced resolution (320×240)
  for `getImageData` pixel analysis — do not analyze full-resolution canvas.

---

## 11. Error Handling

| Condition | Behavior |
|---|---|
| Camera permission denied | Show instructions: "Please allow camera access and reload." |
| No face detected | Overlay message: "Position your face in the center of the frame." |
| No tongue detected | Soft warning below video: "Tongue not visible — ensure good lighting." |
| WebGPU unavailable | MediaPipe falls back to CPU silently (no user-visible change). |
| Download blocked (mobile) | Show a "Save files" button that triggers download on tap. |
| MediaRecorder unsupported | Try `video/webm` → `video/mp4` → skip video, landmarks-only. |

---

## 12. File Structure

Single file, organized in this order:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- meta, title, viewport -->
  <!-- CSS: variables, layout, screens, progress bar, canvas overlay -->
</head>
<body>
  <!-- Screen: welcome -->
  <!-- Screen: instructions -->
  <!-- Screen: live-capture -->
  <!-- Screen: saved -->
  <!-- Screen: all-done -->
  <!-- Hidden: video element (not displayed, used as MediaPipe source) -->
  <!-- Hidden: download anchor -->
</body>
<!-- Script: imports (MediaPipe ESM) -->
<!-- Script: main app logic -->
</html>
```

### CSS screen switching
```css
.screen { display: none; }
.screen.active { display: flex; flex-direction: column; }
```
```javascript
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}
```

---

## 13. Task Metadata Reference

```javascript
const TASKS = [
  {
    id: 'prot',
    name: 'Protrusion',
    direction: 'forward',
    holdThreshold: 0.55,
    dropThreshold: 0.35,
    scoreFunction: scoreProtrusion
  },
  {
    id: 'latR',
    name: 'Lateralization — Right',
    direction: 'right',
    holdThreshold: 0.80,
    dropThreshold: 0.55,
    scoreFunction: (lm, tip, canvas) => scoreLateralization(lm, tip, 'RIGHT', canvas)
  },
  {
    id: 'latL',
    name: 'Lateralization — Left',
    direction: 'left',
    holdThreshold: 0.80,
    dropThreshold: 0.55,
    scoreFunction: (lm, tip, canvas) => scoreLateralization(lm, tip, 'LEFT', canvas)
  },
  {
    id: 'elev',
    name: 'Elevation',
    direction: 'up',
    holdThreshold: 0.75,
    dropThreshold: 0.50,
    scoreFunction: scoreElevation
  }
];
```

---

## 14. Known Limitations (document in the UI footer)

- Tongue detection relies on skin-color heuristics and may fail on very dark or very
  light skin tones, or under poor lighting. In these cases, data is still captured
  (landmarks always work) but the tongue tip indicator may be absent.
- Protrusion scoring is 2D only — no depth measurement. For research use, TrueDepth
  hardware (iPhone 12+ Face ID) should be used for final scoring.
- All scores in this tool are **guidance only** for the hold detector. Actual clinical
  scores (0/25/50/100) must be assigned by a clinician reviewing the stored video.
- Tested on Chrome 120+, Safari 17+. Firefox does not support WebGPU as of 2025 —
  MediaPipe will fall back to WASM/CPU (slower but functional).

---

## 15. Clinical Reference (Lazarus et al., 2014)

The scoring rubric this tool supports:

| Task | Score 100 | Score 50 | Score 25 | Score 0 |
|---|---|---|---|---|
| Protrusion | ≥15 mm past upper-lip margin | 1–15 mm past margin | Some movement, short of margin | No movement |
| Lateralization | Touches commissure | <50% reduction | >50% reduction | No movement |
| Elevation | Tip contacts upper alveolar ridge | Elevation, no contact | — | No visible elevation |

Total ROM = (Protrusion + Lat_Right + Lat_Left + Elevation) / 4

---

*Spec version: 1.0 — April 2026*
*Sanchez Research Lab, UIC ECE*