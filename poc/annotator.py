"""Localhost annotation tool for LROM clips.

Run:  .venv/bin/python -m poc.annotator
Then open http://localhost:8765 in a browser.

Features:
  - Lists all clips under data/<subject>/*_meta.json.
  - Scrub frames with slider / arrow keys / play button.
  - Click video = place tongue-tip at (x, y) on current frame.
  - Single-frame annotate OR range annotate (mark start + end, one click
    applies to every frame in range).
  - Continuous grade 0-100 slider per clip.
  - Saves per-clip JSON to poc/out/annotations/<subject>_<stem>.json.

Annotation schema:
    {
      "subject": "Lucas",
      "stem": "latR_2026-04-14T22-45-45-045Z_s50",
      "task": "latR",
      "lazarus_score": 50,         // from filename, reference only
      "grade": 45,                 // continuous 0-100, user-set
      "tips": {
        "20": [x, y],              // pixel coords in video space
        "21": [x, y],
        ...
      }
    }
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "poc" / "out" / "annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = Path(__file__).parent / "annotator.html"

META_RE = re.compile(r"^(latR|latL|elev)_(.+)_s(\d+)_meta\.json$")

# Cache decoded frames per clip (lazy, in-memory)
_FRAME_CACHE: dict[tuple[str, str], list] = {}
_META_CACHE: dict[tuple[str, str], dict] = {}


def _list_clips() -> list[dict]:
    out = []
    for subj_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for meta in sorted(subj_dir.glob("*_meta.json")):
            m = META_RE.match(meta.name)
            if not m:
                continue
            task, ts, score_str = m.group(1), m.group(2), m.group(3)
            stem = meta.name.replace("_meta.json", "")
            ann_path = _annotation_path(subj_dir.name, stem)
            n_tips = 0
            n_none = 0
            if ann_path.exists():
                try:
                    a = json.loads(ann_path.read_text())
                    n_tips = len(a.get("tips") or {})
                    n_none = len(a.get("no_tongue") or {})
                except Exception:
                    pass
            out.append({
                "subject": subj_dir.name,
                "stem": stem,
                "task": task,
                "lazarus_score": int(score_str),
                "n_tips": n_tips,
                "n_none": n_none,
            })
    return out


def _load_frames(subject: str, stem: str) -> list:
    key = (subject, stem)
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    vid = DATA_DIR / subject / f"{stem}.webm"
    cap = cv2.VideoCapture(str(vid))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    _FRAME_CACHE[key] = frames
    return frames


def _clip_meta(subject: str, stem: str) -> dict:
    key = (subject, stem)
    if key in _META_CACHE:
        return _META_CACHE[key]
    frames = _load_frames(subject, stem)
    if not frames:
        return {"n_frames": 0, "H": 0, "W": 0}
    H, W = frames[0].shape[:2]
    m = {"n_frames": len(frames), "H": int(H), "W": int(W)}
    _META_CACHE[key] = m
    return m


def _annotation_path(subject: str, stem: str) -> Path:
    return OUT_DIR / f"{subject}_{stem}.json"


def _default_annotation(subject: str, stem: str) -> dict:
    m = META_RE.match(f"{stem}_meta.json")
    if m is None:
        return {"subject": subject, "stem": stem, "task": "", "lazarus_score": 0,
                "grade": 0, "tips": {}, "no_tongue": {}}
    task, _ts, score = m.group(1), m.group(2), m.group(3)
    return {
        "subject": subject,
        "stem": stem,
        "task": task,
        "lazarus_score": int(score),
        "grade": int(score),
        "tips": {},
        "no_tongue": {},
    }


def _load_annotation(subject: str, stem: str) -> dict:
    p = _annotation_path(subject, stem)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return _default_annotation(subject, stem)


def _interpolate_tips(tips: dict, no_tongue: dict) -> dict:
    """Fill frames between consecutive anchor tips via linear interp.
    Skip any gap that contains a no_tongue frame."""
    if not tips:
        return tips
    keys = sorted(int(k) for k in tips.keys())
    none_set = {int(k) for k in (no_tongue or {}).keys()}
    out = {str(k): list(tips[str(k)]) for k in keys}
    for a, b in zip(keys, keys[1:]):
        if b - a <= 1:
            continue
        if any(f in none_set for f in range(a + 1, b)):
            continue
        xa, ya = tips[str(a)]
        xb, yb = tips[str(b)]
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            out[str(f)] = [xa + t * (xb - xa), ya + t * (yb - ya)]
    return out


def _save_annotation(obj: dict) -> None:
    obj["tips"] = _interpolate_tips(obj.get("tips") or {}, obj.get("no_tongue") or {})
    p = _annotation_path(obj["subject"], obj["stem"])
    p.write_text(json.dumps(obj, indent=2))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, mime: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            if HTML_PATH.exists():
                self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_bytes(b"annotator.html missing", "text/plain", 500)
            return

        if u.path == "/api/clips":
            self._send_json(_list_clips())
            return

        qs = parse_qs(u.query)
        subject = qs.get("subject", [""])[0]
        stem = qs.get("stem", [""])[0]

        if u.path == "/api/meta":
            self._send_json(_clip_meta(subject, stem))
            return

        if u.path == "/api/annotation":
            self._send_json(_load_annotation(subject, stem))
            return

        if u.path == "/api/frame":
            idx = int(qs.get("idx", ["0"])[0])
            frames = _load_frames(subject, stem)
            if not frames or idx < 0 or idx >= len(frames):
                self._send_bytes(b"bad idx", "text/plain", 404)
                return
            ok, jpg = cv2.imencode(".jpg", frames[idx],
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self._send_bytes(b"encode error", "text/plain", 500)
                return
            self._send_bytes(jpg.tobytes(), "image/jpeg")
            return

        self._send_bytes(b"not found", "text/plain", 404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        if u.path == "/api/save":
            try:
                obj = json.loads(raw)
                _save_annotation(obj)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            return
        self._send_bytes(b"not found", "text/plain", 404)


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"Serving on http://{host}:{port}")
    print(f"Annotations → {OUT_DIR}")
    srv = ThreadingHTTPServer((host, port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
