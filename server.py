import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Tongue ROM Data Collection")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_HTML = ROOT / "index.html"

ALLOWED_TASKS = {"latR", "latL", "elev"}
ALLOWED_SCORES = {
    "latR": {0, 25, 50, 100},
    "latL": {0, 25, 50, 100},
    "elev": {0, 50, 100},
}
NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{1,64}$")
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_LANDMARKS_BYTES = 20 * 1024 * 1024


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/upload")
async def upload(
    name: str = Form(...),
    task: str = Form(...),
    clinical_score: int = Form(...),
    peak_auto_score: float = Form(...),
    captured_at: str = Form(...),
    video: UploadFile = File(...),
    landmarks: UploadFile = File(...),
):
    name = name.strip()
    if not NAME_RE.match(name):
        raise HTTPException(400, "Invalid name (1-64 chars, letters/digits/space/._-)")
    if task not in ALLOWED_TASKS:
        raise HTTPException(400, f"Invalid task '{task}'")
    if clinical_score not in ALLOWED_SCORES[task]:
        raise HTTPException(
            400,
            f"Score {clinical_score} not allowed for task '{task}'; allowed: {sorted(ALLOWED_SCORES[task])}",
        )

    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(400, "Empty video upload")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(413, f"Video exceeds {MAX_VIDEO_BYTES} bytes")

    landmarks_bytes = await landmarks.read()
    if not landmarks_bytes:
        raise HTTPException(400, "Empty landmarks upload")
    if len(landmarks_bytes) > MAX_LANDMARKS_BYTES:
        raise HTTPException(413, f"Landmarks exceeds {MAX_LANDMARKS_BYTES} bytes")
    try:
        landmarks_doc = json.loads(landmarks_bytes)
    except json.JSONDecodeError:
        raise HTTPException(400, "landmarks file is not valid JSON")

    ts_safe = re.sub(r"[^0-9A-Za-z]", "-", captured_at)[:32]
    safe_dir = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    subject_dir = DATA_DIR / safe_dir
    subject_dir.mkdir(parents=True, exist_ok=True)

    base = f"{task}_{ts_safe}_s{clinical_score}"
    video_path = subject_dir / f"{base}.webm"
    landmarks_path = subject_dir / f"{base}_landmarks.json"
    meta_path = subject_dir / f"{base}_meta.json"

    video_path.write_bytes(video_bytes)
    landmarks_path.write_bytes(landmarks_bytes)
    meta_path.write_text(
        json.dumps(
            {
                "name": name,
                "task": task,
                "clinical_score": clinical_score,
                "peak_auto_score": peak_auto_score,
                "captured_at": captured_at,
                "frame_count": landmarks_doc.get("frameCount"),
                "mirrored": landmarks_doc.get("mirrored", True),
                "video_bytes": len(video_bytes),
            },
            indent=2,
        )
    )

    return JSONResponse(
        {"ok": True, "path": str(video_path.relative_to(ROOT))}
    )
