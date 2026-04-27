"""Upload trained models to Hugging Face Hub (private repo).

Creates or updates a private HF repo with:
  - best model weights at root (best.pt)
  - all full LOSO fold weights under loso_folds/<subject>/best.pt
  - model card (README.md) auto-generated from training metadata

Run:
    python -m poc.upload_model_hf                             # upload best model only
    python -m poc.upload_model_hf --run loso_Miquel_clahe_unsharp_v2
    python -m poc.upload_model_hf --all-loso                  # upload all 30 LOSO folds
    python -m poc.upload_model_hf --all-loso --run loso_Miquel_clahe_unsharp_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "tip"

DEFAULT_RUN  = "loso_Miquel_clahe_unsharp_v2"
DEFAULT_REPO = "orosense-tongue-tip"   # will be created under your HF username


def _read_best_metrics(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(
        r.get("metrics/pose_mAP50-95(B)", 0) or 0))
    return {k.strip(): v.strip() for k, v in best.items()}


def _make_model_card(run_name: str, metrics: dict, repo_id: str) -> str:
    pose_map50    = metrics.get("metrics/pose_mAP50(B)",    "N/A")
    pose_map5095  = metrics.get("metrics/pose_mAP50-95(B)", "N/A")
    box_map50     = metrics.get("metrics/mAP50(B)",         "N/A")

    parts = run_name.split("_")
    val_subject = parts[1] if len(parts) > 1 else "unknown"
    pipeline    = "_".join(parts[2:]) if len(parts) > 2 else "clahe"

    return f"""---
license: mit
tags:
  - yolov8
  - pose-estimation
  - tongue-tracking
  - oral-cancer
  - clinical
  - ultralytics
pipeline_tag: object-detection
---

# OroSense — Tongue Tip Tracker

YOLOv8n-pose model for tongue-tip keypoint detection in tongue range-of-motion (ROM)
clinical assessment. Trained for the OroSense project as part of automatic Lazarus
(2014) LROM scoring.

## Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n-pose |
| Keypoints | 3: left commissure, right commissure, tongue tip |
| Input size | 640×640 |
| Preprocessing | {pipeline.replace("_", " → ")} |
| Val subject (LOSO) | {val_subject} |
| Run name | `{run_name}` |

## Keypoint layout

| Index | Keypoint | MediaPipe landmark |
|-------|----------|--------------------|
| 0 | Left commissure (CL) | 61 |
| 1 | Right commissure (CR) | 291 |
| 2 | Tongue tip (T) | — (annotated) |

Visibility: `vis=2` tongue present, `vis=0` no tongue / mouth closed.

## Val metrics (LOSO fold — val={val_subject})

| Metric | Value |
|--------|-------|
| Pose mAP50 | {pose_map50} |
| Pose mAP50-95 | {pose_map5095} |
| Box mAP50 | {box_map50} |

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("frame.jpg", conf=0.25)

for r in results:
    kp      = r.keypoints.xy[0]    # (3, 2)
    kp_conf = r.keypoints.conf[0]  # (3,)
    tip_xy  = kp[2]                # tongue tip pixel coords
    tip_vis = float(kp_conf[2]) >= 0.5
```

## Preprocessing (match training distribution)

```python
import cv2
import numpy as np

def preprocess(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    bgr_clahe = cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(bgr_clahe, (5, 5), 0)
    return cv2.addWeighted(bgr_clahe, 2.5, blurred, -1.5, 0)
```

## Training

Trained with LOSO cross-validation on {val_subject} held out as val set.
Full pipeline: `poc/build_tip_dataset.py` + `poc/train_tip.py`.
See [OroSense repository](https://github.com/LucasTakanori/orosense_lateralization)
for full training code and annotation tools.
"""


def _upload_single(api, run_name: str, repo_id: str,
                   path_in_repo_prefix: str = "") -> None:
    """Upload weights + artefacts for one run into repo at optional subfolder prefix."""
    run_dir = RUNS_DIR / run_name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"  SKIP {run_name} — no weights found")
        return

    prefix = path_in_repo_prefix.rstrip("/") + "/" if path_in_repo_prefix else ""

    print(f"  [{run_name}] uploading best.pt → {prefix}best.pt")
    api.upload_file(
        path_or_fileobj=str(weights),
        path_in_repo=f"{prefix}best.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add weights: {run_name}",
    )

    for fname in ("results.csv", "args.yaml"):
        fpath = run_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"{prefix}{fname}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {fname}: {run_name}",
            )

    for img in ("val_batch0_pred.jpg", "results.png", "confusion_matrix_normalized.png"):
        fpath = run_dir / img
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"{prefix}{img}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {img}: {run_name}",
            )


def _full_loso_runs() -> list[str]:
    """Return the 30 clean LOSO fold run names (no experimental suffixes)."""
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # keep only loso_<subject> — skip anything with extra _vN / _clahe / _gamma / -2
        if not name.startswith("loso_"):
            continue
        suffix = name[len("loso_"):]
        # skip experimental runs
        if any(x in suffix for x in ["_clahe", "_gamma", "_3kpt", "_v1", "_v2", "-2"]):
            continue
        if (d / "weights" / "best.pt").exists():
            runs.append(name)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",       default=DEFAULT_RUN,  help="Best/single run to upload at repo root")
    ap.add_argument("--repo",      default=DEFAULT_REPO, help="HF repo name (without username)")
    ap.add_argument("--all-loso",  action="store_true",  help="Also upload all 30 LOSO folds under loso_folds/")
    ap.add_argument("--export-coreml", action="store_true", help="Export CoreML before upload")
    args = ap.parse_args()

    from huggingface_hub import HfApi, login

    login()   # prompts for token if not cached; use `hf auth login` in terminal first
    api     = HfApi()
    user    = api.whoami()["name"]
    repo_id = f"{user}/{args.repo}"

    # ── create private repo (idempotent) ──────────────────────────────────────
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"Repo: https://huggingface.co/{repo_id}  (private)\n")

    # ── best model at repo root ───────────────────────────────────────────────
    run_dir = RUNS_DIR / args.run
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights}")

    # optional CoreML export
    coreml_path = run_dir / "weights" / "best.mlpackage"
    if args.export_coreml and not coreml_path.exists():
        print("Exporting CoreML...")
        from ultralytics import YOLO
        YOLO(str(weights)).export(format="coreml", imgsz=640, nms=True)

    print(f"=== Best model: {args.run} → root ===")
    _upload_single(api, args.run, repo_id, path_in_repo_prefix="")

    if coreml_path.exists():
        print(f"  Uploading best.mlpackage ...")
        api.upload_folder(
            folder_path=str(coreml_path),
            path_in_repo="best.mlpackage",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add CoreML export",
        )

    # ── model card ────────────────────────────────────────────────────────────
    metrics   = _read_best_metrics(run_dir)
    card_text = _make_model_card(args.run, metrics, repo_id)
    card_path = Path("/tmp/hf_model_card_README.md")
    card_path.write_text(card_text)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )

    # ── all LOSO folds ────────────────────────────────────────────────────────
    if args.all_loso:
        loso_runs = _full_loso_runs()
        print(f"\n=== Uploading {len(loso_runs)} LOSO folds → loso_folds/ ===")
        for run_name in loso_runs:
            subject = run_name[len("loso_"):]
            _upload_single(api, run_name, repo_id,
                           path_in_repo_prefix=f"loso_folds/{subject}")

    print(f"\nDone → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
