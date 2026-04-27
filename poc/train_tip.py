"""LOSO YOLOv8n-pose training for tongue-tip tracker.

Trains one fold per LOSO subject using dataset YAMLs produced by build_tip_dataset.py.
Each fold: one subject = val, rest = train.

Run:
    python -m poc.train_tip                        # all LOSO folds
    python -m poc.train_tip --val-subject 01       # single fold
    python -m poc.train_tip --val-subject 01 --epochs 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "tip"


def train_fold(yaml_path: Path, run_name: str, epochs: int, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        # Disable left-right flip — laterality-specific task
        fliplr=0.0,
        # Keep other augmentations at ultralytics defaults
        device="cuda" if _cuda_available() else "cpu",
        verbose=True,
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _find_yamls(val_subject: str | None, dataset_dir: Path) -> list[tuple[str, Path]]:
    yamls = sorted(dataset_dir.glob("tip_loso_*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No LOSO YAML files in {dataset_dir}. "
            "Run: python -m poc.build_tip_dataset"
        )
    if val_subject:
        yamls = [y for y in yamls if f"tip_loso_{val_subject}.yaml" == y.name]
        if not yamls:
            raise FileNotFoundError(f"No YAML for val_subject={val_subject!r}")
    return [(y.stem.replace("tip_loso_", ""), y) for y in yamls]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-subject",  default=None, help="Run only this LOSO fold")
    ap.add_argument("--run-name",     default=None, help="Custom run name (default: loso_<subject>)")
    ap.add_argument("--dataset-dir",  default=None, help="Path to dataset dir (default: <root>/dataset)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else ROOT / "dataset"
    folds = _find_yamls(args.val_subject, dataset_dir)
    print(f"Training {len(folds)} LOSO fold(s): {[s for s, _ in folds]}")

    for val_subj, yaml_path in folds:
        run_name = args.run_name or f"loso_{val_subj}"
        print(f"\n{'='*60}")
        print(f"FOLD: val={val_subj}  run={run_name}  yaml={yaml_path.name}")
        print(f"{'='*60}")
        train_fold(yaml_path, run_name, args.epochs, args.imgsz)
        print(f"FOLD {val_subj} done → {RUNS_DIR}/loso_{val_subj}")


if __name__ == "__main__":
    main()
