"""POC training: Leave-One-Subject-Out CV on the tiny dataset.

Reads poc/out/dataset.csv, fits a Ridge and RandomForest regressor
per LOSO fold, writes predictions and diagnostic plots.

Run:  .venv/bin/python -m poc.train
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "poc" / "out"

LANDMARK_FEATS = [
    "mouth_aspect", "mouth_open_norm", "inter_comm_norm",
    "nose_to_mouth_y", "rC_z", "lC_z", "lip_depth_diff",
]
IMAGE_FEATS = [
    "img_mean_H", "img_mean_S", "img_mean_V",
    "img_tongue_frac", "img_redness",
]
TASK_FEATS = ["task_latR", "task_latL", "task_elev"]
FEAT_COLS = LANDMARK_FEATS + IMAGE_FEATS + TASK_FEATS

ALLOWED_BINS = {
    "latR": np.array([0, 25, 50, 100]),
    "latL": np.array([0, 25, 50, 100]),
    "elev": np.array([0, 50, 100]),
}


def add_task_onehot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for t in ("latR", "latL", "elev"):
        df[f"task_{t}"] = (df["task"] == t).astype(float)
    return df


def discretize(y_pred: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    out = np.empty_like(y_pred)
    for i, (p, t) in enumerate(zip(y_pred, tasks)):
        bins = ALLOWED_BINS[t]
        out[i] = bins[np.argmin(np.abs(bins - p))]
    return out


def loso_cv(df: pd.DataFrame, name: str, factory):
    logo = LeaveOneGroupOut()
    X = df[FEAT_COLS].values
    y = df["score"].values.astype(float)
    groups = df["subject"].values
    preds = np.full_like(y, np.nan, dtype=float)
    for tr_idx, te_idx in logo.split(X, y, groups):
        if len(np.unique(y[tr_idx])) < 2:
            preds[te_idx] = y[tr_idx].mean()
            continue
        model = factory()
        model.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] = model.predict(X[te_idx])
    preds = np.clip(preds, 0, 100)
    mae = float(np.mean(np.abs(preds - y)))
    disc = discretize(preds, df["task"].values)
    acc = float((disc == y).mean())
    print(f"\n[{name}]  LOSO MAE = {mae:5.2f}   discrete-accuracy = {acc:.2f}")
    for t in ("latR", "latL", "elev"):
        mask = df["task"].values == t
        if mask.any():
            t_mae = float(np.mean(np.abs(preds[mask] - y[mask])))
            print(f"    {t:5s}  n={mask.sum():2d}   mae={t_mae:5.2f}")
    return preds, mae, acc


def plot_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    task_dist = df.groupby(["task", "score"]).size().unstack(fill_value=0)
    task_dist.plot(kind="bar", ax=axes[0], edgecolor="k")
    axes[0].set_title("Samples per task × clinical score")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="score")

    subj_dist = df.groupby(["subject", "task"]).size().unstack(fill_value=0)
    subj_dist.plot(kind="bar", stacked=True, ax=axes[1], edgecolor="k")
    axes[1].set_title("Samples per subject × task")
    axes[1].set_ylabel("count")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = OUT_DIR / "dataset_overview.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"→ {path}")


def plot_predictions(df: pd.DataFrame, preds: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, len(preds), figsize=(5.5 * len(preds), 5))
    if len(preds) == 1:
        axes = [axes]
    colors = {"latR": "C0", "latL": "C1", "elev": "C2"}
    for ax, (name, p) in zip(axes, preds.items()):
        for t in ("latR", "latL", "elev"):
            mask = df["task"].values == t
            if not mask.any():
                continue
            ax.scatter(
                df["score"].values[mask] + np.random.uniform(-1, 1, mask.sum()),
                p[mask],
                c=colors[t], s=80, alpha=0.8, edgecolors="k", label=t,
            )
        ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="ideal")
        ax.set_xlim(-8, 108)
        ax.set_ylim(-8, 108)
        ax.set_xlabel("clinical label")
        ax.set_ylabel("predicted")
        mae = float(np.mean(np.abs(p - df["score"].values)))
        ax.set_title(f"{name}   MAE={mae:.1f}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "predictions.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"→ {path}")


def plot_crop_gallery(df: pd.DataFrame):
    if df.empty or "crop_path" not in df.columns:
        return
    n = len(df)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_2d(axes)
    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        try:
            img = mpimg.imread(str(ROOT / row["crop_path"]))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(
            f"{row['subject']}\n{row['task']} = {int(row['score'])}",
            fontsize=9,
        )
        ax.axis("off")
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")
    fig.suptitle("Buccal ROI crops (peak frame)", y=1.0)
    fig.tight_layout()
    path = OUT_DIR / "crops_gallery.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"→ {path}")


def main():
    csv = OUT_DIR / "dataset.csv"
    if not csv.exists():
        raise SystemExit(f"Missing {csv} — run build_dataset first")
    df = pd.read_csv(csv)
    df = add_task_onehot(df)
    print(f"Loaded {len(df)} samples · {df['subject'].nunique()} subjects")
    print("\nSamples per task × score:")
    print(df.groupby(["task", "score"]).size().unstack(fill_value=0))

    y = df["score"].values.astype(float)
    base = np.full_like(y, y.mean())
    print(f"\n[baseline: constant mean] MAE = {np.mean(np.abs(base - y)):.2f}")

    ridge_pred, _, _ = loso_cv(
        df,
        "Ridge (landmark + img feats)",
        lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    )
    rf_pred, _, _ = loso_cv(
        df,
        "RandomForest (landmark + img feats)",
        lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=0, n_jobs=-1
        ),
    )

    out = df[["subject", "task", "score"]].copy()
    out["pred_ridge"] = ridge_pred
    out["pred_rf"] = rf_pred
    out.to_csv(OUT_DIR / "predictions.csv", index=False)
    print(f"\n→ {OUT_DIR / 'predictions.csv'}")

    plot_overview(df)
    plot_predictions(df, {"Ridge": ridge_pred, "RandomForest": rf_pred})
    plot_crop_gallery(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
