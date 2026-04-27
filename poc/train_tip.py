"""LOSO YOLOv8n-pose training for tongue-tip tracker.

Trains one fold per LOSO subject using dataset YAMLs produced by build_tip_dataset.py.
Each fold: one subject = val, rest = train.
Metrics are logged to Weights & Biases (wandb) under project "orosense-tip".

Run:
    python -m poc.train_tip                        # all LOSO folds
    python -m poc.train_tip --val-subject Miquel   # single fold
    python -m poc.train_tip --val-subject Miquel --epochs 30
    python -m poc.train_tip --no-wandb             # disable wandb logging
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "tip"

WANDB_PROJECT = "orosense-tip"


def train_fold(yaml_path: Path, run_name: str, epochs: int, imgsz: int,
               pipeline: str, use_wandb: bool) -> None:
    from ultralytics import YOLO
    import yaml as _yaml

    # derive val subject from run_name for wandb tags
    val_subject = run_name.replace("loso_", "").split("_")[0]

    if use_wandb:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            tags=["loso", pipeline, f"val={val_subject}"],
            config={
                "val_subject": val_subject,
                "pipeline":    pipeline,
                "epochs":      epochs,
                "imgsz":       imgsz,
                "batch":       16,
                "model":       "yolov8n-pose",
                "fliplr":      0.0,
                "dataset":     yaml_path.name,
            },
            reinit=True,
        )

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        fliplr=0.0,
        device="cuda" if _cuda_available() else "cpu",
        verbose=True,
    )

    if use_wandb:
        # log best val metrics as summary
        results_csv = RUNS_DIR / run_name / "results.csv"
        if results_csv.exists():
            import csv
            rows = list(csv.DictReader(results_csv.read_text().splitlines()))
            if rows:
                best = max(rows, key=lambda r: float(r.get(
                    "metrics/pose_mAP50-95(B)", 0) or 0))
                wandb.summary.update({k.strip(): _safe_float(v)
                                      for k, v in best.items()})
        wandb.finish()


def _safe_float(v: str) -> float | str:
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _pipeline_from_dataset_dir(dataset_dir: Path) -> str:
    name = dataset_dir.name
    if name.startswith("dataset_"):
        return name[len("dataset_"):]
    return "clahe"


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
    ap.add_argument("--epochs",       type=int, default=20)
    ap.add_argument("--imgsz",        type=int, default=640)
    ap.add_argument("--no-wandb",     action="store_true", help="Disable wandb logging")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else ROOT / "dataset"
    pipeline    = _pipeline_from_dataset_dir(dataset_dir)
    use_wandb   = not args.no_wandb

    if use_wandb:
        try:
            import wandb
            wandb.login()
        except Exception as e:
            print(f"wandb login failed ({e}), continuing without wandb")
            use_wandb = False

    folds = _find_yamls(args.val_subject, dataset_dir)
    print(f"Training {len(folds)} LOSO fold(s): {[s for s, _ in folds]}")
    print(f"Pipeline : {pipeline}")
    print(f"wandb    : {'enabled → project=' + WANDB_PROJECT if use_wandb else 'disabled'}")

    for val_subj, yaml_path in folds:
        run_name = args.run_name or f"loso_{val_subj}"
        print(f"\n{'='*60}")
        print(f"FOLD: val={val_subj}  run={run_name}  yaml={yaml_path.name}")
        print(f"{'='*60}")
        train_fold(yaml_path, run_name, args.epochs, args.imgsz, pipeline, use_wandb)
        print(f"FOLD {val_subj} done → {RUNS_DIR}/{run_name}")


if __name__ == "__main__":
    main()
