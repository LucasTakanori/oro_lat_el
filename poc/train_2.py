"""Multimodal training pipeline (image crop + landmarks) with ResNet-50 backbones.

Trains three variants:
1) 4-class softmax classifier
2) CORAL ordinal head
3) CORN ordinal head

Inputs:
- crops from poc/out/crops (resized to 224x224x3)
- per-frame landmarks read from *_landmarks.json (paired through dataset.csv rows)
- labels from poc/out/dataset.csv ("score")

Artifacts are written to:
  poc/out/train2/<run_name>/<model_name>/
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.models import ResNet50_Weights

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "poc" / "out" / "dataset.csv"
OVERLAYS_DIR = ROOT / "poc" / "overlays"
ARTIFACTS_ROOT = ROOT / "poc" / "out" / "train2"

TASK_TO_ID = {"latR": 0, "latL": 1, "elev": 2}
LM_IDXS = [61, 291, 13, 14, 1, 152]  # 6 points x (x,y) => 12 scalars
LAT_LM_SLICE = slice(0, 8)  # commissures + upper/lower lip centers (4 points x 2D)
N_CLASSES = 4
TASK_SCORE_ANCHORS = {
    0: np.array([0.0, 25.0, 50.0, 100.0], dtype=np.float32),  # latR
    1: np.array([0.0, 25.0, 50.0, 100.0], dtype=np.float32),  # latL
    2: np.array([0.0, 50.0, 100.0], dtype=np.float32),  # elev
}


@dataclass
class TrainConfig:
    """Hyperparameters and data checks used by the full training run."""
    seed: int = 42
    n_seeds: int = 4
    seed_stride: int = 97
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_folds: int = 5
    patience: int = 20
    min_delta: float = 0.15
    num_workers: int = 0
    use_pretrained_backbone: bool = True
    require_overlay_file: bool = True
    freeze_backbone_stages: int = 4
    save_all_epochs: bool = False
    use_weighted_sampler: bool = True
    loss_distance_penalty: float = 0.15

    # Early stopping improves only when val_mae drops by at least min_delta.


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score_to_class(score: float, task: str) -> int:
    """Map a clinical score to a task-aware ordinal class id.

    Lateralization uses anchors [0, 25, 50, 100].
    Elevation uses anchors [0, 50, 100] (stored as [0, 50, 100, 100]).
    """
    anchors = TASK_SCORE_ANCHORS[TASK_TO_ID[task]]
    return int(np.argmin(np.abs(anchors - score)))


def read_json(path: Path) -> dict[str, Any]:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text())


def normalize_landmark_vec(lm: list[dict[str, float]]) -> np.ndarray:
    """Create a 12D normalized landmark vector from 6 key points.

    Each point contributes (x, y) normalized by mouth center and inter-commissure
    distance to reduce scale and translation effects.
    """
    def p(i: int) -> np.ndarray:
        return np.array([lm[i]["x"], lm[i]["y"]], dtype=np.float32)

    r_c = p(61)
    l_c = p(291)
    center = (r_c + l_c) / 2.0
    scale = np.linalg.norm(r_c - l_c) + 1e-6

    feat = []
    for idx in LM_IDXS:
        xy = p(idx)
        xy_norm = (xy - center) / scale
        feat.extend([float(xy_norm[0]), float(xy_norm[1])])
    return np.array(feat, dtype=np.float32)


def make_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """Create subject-wise K folds as (train_subjects, val_subjects)."""
    uniq = sorted(set(subjects.tolist()))
    if len(uniq) < 2:
        raise ValueError("Need at least 2 unique subjects for fold split.")
    k = max(2, min(n_folds, len(uniq)))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    buckets = [[] for _ in range(k)]
    for i, subj in enumerate(uniq):
        buckets[i % k].append(subj)

    folds: list[tuple[set[str], set[str]]] = []
    for i in range(k):
        val_subj = set(buckets[i])
        train_subj = set(s for j, b in enumerate(buckets) if j != i for s in b)
        if val_subj and train_subj:
            folds.append((train_subj, val_subj))
    return folds


def build_samples(cfg: TrainConfig) -> tuple[list[dict[str, Any]], list[str]]:
    """Build multimodal training samples from dataset.csv and landmark files."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("dataset.csv is empty")

    overlay_subjects = {
        p.name for p in OVERLAYS_DIR.iterdir() if p.is_dir()
    } if OVERLAYS_DIR.exists() else set()
    if overlay_subjects:
        df = df[df["subject"].isin(overlay_subjects)].copy()

    lm_cache: dict[Path, list[dict[str, Any]]] = {}
    samples: list[dict[str, Any]] = []
    skips: list[str] = []
    for _, row in df.iterrows():
        subject = str(row["subject"])
        task = str(row["task"])
        score = float(row["score"])
        crop_abs = ROOT / str(row["crop_path"])
        video_rel = Path(str(row["video_path"]))
        video_abs = ROOT / video_rel
        overlay_abs = OVERLAYS_DIR / subject / f"{video_abs.stem}_overlay.mp4"
        lm_abs = video_abs.with_name(f"{video_abs.stem}_landmarks.json")

        if cfg.require_overlay_file and not overlay_abs.exists():
            skips.append(f"{subject}/{video_abs.stem}: missing overlay")
            continue
        if not crop_abs.exists() or not lm_abs.exists():
            skips.append(f"{subject}/{video_abs.stem}: missing crop/json")
            continue
        frame_idx = int(row.get("best_frame_idx", 0))

        try:
            if lm_abs not in lm_cache:
                lm_cache[lm_abs] = read_json(lm_abs).get("landmarks", [])
            frames = lm_cache[lm_abs]
            if not frames:
                skips.append(f"{subject}/{video_abs.stem}: empty landmarks")
                continue
            frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
            lm = frames[frame_idx]["lm"]
            lm_vec = normalize_landmark_vec(lm)
            if lm_vec.shape[0] != 12:
                skips.append(f"{subject}/{video_abs.stem}: bad landmark size")
                continue
        except Exception as exc:  # pragma: no cover - safety for malformed files
            skips.append(f"{subject}/{video_abs.stem}: landmark error ({exc})")
            continue

        samples.append(
            {
                "subject": subject,
                "task": task,
                "task_id": TASK_TO_ID[task],
                "score": score,
                "class_id": score_to_class(score, task),
                "sample_uid": f"{subject}|{video_abs.stem}|f{frame_idx}",
                "crop_path": crop_abs,
                "overlay_path": overlay_abs,
                "landmark_vec": lm_vec,
            }
        )
    return samples, skips


class MultiModalDataset(Dataset):
    """PyTorch dataset returning image, landmarks, task id, and labels."""
    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_transform: transforms.Compose,
        is_train: bool = False,
    ) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.is_train = is_train
        self.rand_crop = transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1))
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03
        )
        self.max_rot_deg = 6.0

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and transform one sample into tensors."""
        sample = self.samples[idx]
        image = Image.open(sample["crop_path"]).convert("RGB")
        lm_vec_np = np.array(sample["landmark_vec"], dtype=np.float32).copy()
        task_id_int = int(sample["task_id"])

        if self.is_train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, self.rand_crop.scale, self.rand_crop.ratio
            )
            image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)

            flipped = random.random() < 0.5
            if flipped:
                image = TF.hflip(image)
                lm_points = lm_vec_np.reshape(6, 2)
                lm_points[:, 0] = -lm_points[:, 0]  # mirror x in normalized space
                lm_points[[0, 1]] = lm_points[[1, 0]]  # swap R/L commissures
                lm_vec_np = lm_points.reshape(-1)
                if task_id_int == TASK_TO_ID["latR"]:
                    task_id_int = TASK_TO_ID["latL"]
                elif task_id_int == TASK_TO_ID["latL"]:
                    task_id_int = TASK_TO_ID["latR"]

            image = self.color_jitter(image)
            angle = random.uniform(-self.max_rot_deg, self.max_rot_deg)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)

        image = self.image_transform(image)
        lm_vec = torch.tensor(lm_vec_np, dtype=torch.float32)
        lat_lm_vec = torch.tensor(lm_vec_np[LAT_LM_SLICE], dtype=torch.float32)
        task_id = torch.tensor(task_id_int, dtype=torch.long)
        class_id = torch.tensor(sample["class_id"], dtype=torch.long)
        score = torch.tensor(sample["score"], dtype=torch.float32)
        return {
            "image": image,
            "lm_vec": lm_vec,
            "lat_lm_vec": lat_lm_vec,
            "task_id": task_id,
            "class_id": class_id,
            "score": score,
            "sample_uid": sample["sample_uid"],
        }


class MultiModalResNet(nn.Module):
    """ResNet50 image backbone fused with landmark and task embeddings."""
    def __init__(self, head_type: str = "softmax", pretrained: bool = True, freeze_stages: int = 0):
        """Initialize fusion model with selected prediction head."""
        super().__init__()
        self.head_type = head_type
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.lm_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )
        self.lat_lm_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
        )
        fused_dim = feat_dim + 32 + 16

        if head_type == "softmax":
            lat_dim = 4
            elev_dim = 3
        elif head_type in {"coral", "corn"}:
            lat_dim = 3
            elev_dim = 2
        else:
            raise ValueError(f"Unknown head_type={head_type}")
        self.lat_head = nn.Linear(fused_dim + 16, lat_dim)
        self.elev_head = nn.Linear(fused_dim, elev_dim)
        self._freeze_backbone_stages(freeze_stages)

    def _freeze_backbone_stages(self, freeze_stages: int) -> None:
        """Freeze early ResNet stages to reduce overfitting on small data.

        Stages:
        1 -> conv1 + bn1
        2 -> layer1
        3 -> layer2
        4 -> layer3
        5 -> layer4
        """
        if freeze_stages <= 0:
            return
        stages = [
            nn.ModuleList([self.backbone.conv1, self.backbone.bn1]),
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        for module in stages[: min(freeze_stages, len(stages))]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self, image: torch.Tensor, lm_vec: torch.Tensor, lat_lm_vec: torch.Tensor, task_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute logits from image, landmark vector, and task id."""
        img_feat = self.backbone(image)
        lm_feat = self.lm_mlp(lm_vec)
        lat_feat = self.lat_lm_mlp(lat_lm_vec)
        task_one_hot = torch.nn.functional.one_hot(task_id, num_classes=3).float()
        task_feat = self.task_mlp(task_one_hot)
        fused = torch.cat([img_feat, lm_feat, task_feat], dim=1)
        is_lat = (task_id != TASK_TO_ID["elev"])
        lat_logits = torch.zeros((0, self.lat_head.out_features), device=fused.device, dtype=fused.dtype)
        elev_logits = torch.zeros((0, self.elev_head.out_features), device=fused.device, dtype=fused.dtype)
        if is_lat.any():
            lat_fused = torch.cat([fused[is_lat], lat_feat[is_lat]], dim=1)
            lat_logits = self.lat_head(lat_fused)
        if (~is_lat).any():
            elev_logits = self.elev_head(fused[~is_lat])
        return {
            "is_lat": is_lat,
            "lat_logits": lat_logits,
            "elev_logits": elev_logits,
        }


def coral_levels_from_labels(class_ids: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert ordinal class labels into CORAL binary level targets."""
    levels = []
    for k in range(n_classes - 1):
        levels.append((class_ids > k).float())
    return torch.stack(levels, dim=1)


def coral_loss(logits: torch.Tensor, class_ids: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Binary cross-entropy CORAL loss over ordinal levels."""
    levels = coral_levels_from_labels(class_ids, n_classes=n_classes)
    return nn.functional.binary_cross_entropy_with_logits(logits, levels)


def corn_loss(logits: torch.Tensor, class_ids: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Conditional ordinal regression (CORN) loss."""
    total = 0.0
    used = 0
    for k in range(n_classes - 1):
        mask = class_ids >= k
        if mask.sum() == 0:
            continue
        target = (class_ids[mask] > k).float()
        total = total + nn.functional.binary_cross_entropy_with_logits(logits[mask, k], target)
        used += 1
    if used == 0:
        return logits.sum() * 0.0
    return total / used


def logits_to_class(logits: torch.Tensor, head_type: str) -> torch.Tensor:
    """Decode logits to discrete ordinal class ids."""
    if head_type == "softmax":
        return logits.argmax(dim=1)
    probs = torch.sigmoid(logits)
    # Threshold decoding for ordinal heads.
    return (probs > 0.5).sum(dim=1)


def _continuous_score_from_rank(rank: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Interpolate continuous score from a fractional rank."""
    n_classes = anchors.shape[1]
    lo = np.floor(rank).astype(int)
    lo = np.clip(lo, 0, n_classes - 1)
    hi = np.clip(lo + 1, 0, n_classes - 1)
    frac = rank - lo
    low_v = anchors[np.arange(len(rank)), lo]
    high_v = anchors[np.arange(len(rank)), hi]
    return (1.0 - frac) * low_v + frac * high_v


def decode_outputs(
    outputs: dict[str, torch.Tensor], task_ids: torch.Tensor, head_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Decode task-specific logits into class ids and continuous scores."""
    task_ids_np = task_ids.detach().cpu().numpy()
    pred_class = np.zeros(len(task_ids_np), dtype=np.int64)
    pred_score = np.zeros(len(task_ids_np), dtype=np.float32)

    is_lat = outputs["is_lat"]
    lat_idx = torch.where(is_lat)[0]
    elev_idx = torch.where(~is_lat)[0]

    if len(lat_idx) > 0:
        lat_logits = outputs["lat_logits"]
        lat_cls = logits_to_class(lat_logits, head_type).detach().cpu().numpy()
        pred_class[lat_idx.cpu().numpy()] = lat_cls
        lat_anchors = np.tile(TASK_SCORE_ANCHORS[TASK_TO_ID["latR"]], (len(lat_cls), 1))
        if head_type == "softmax":
            probs = torch.softmax(lat_logits, dim=1).detach().cpu().numpy()
            pred_score[lat_idx.cpu().numpy()] = np.sum(probs * lat_anchors, axis=1)
        else:
            rank = torch.sigmoid(lat_logits).detach().cpu().numpy().sum(axis=1)
            pred_score[lat_idx.cpu().numpy()] = _continuous_score_from_rank(rank, lat_anchors)

    if len(elev_idx) > 0:
        elev_logits = outputs["elev_logits"]
        elev_cls = logits_to_class(elev_logits, head_type).detach().cpu().numpy()
        pred_class[elev_idx.cpu().numpy()] = elev_cls
        elev_anchors = np.tile(TASK_SCORE_ANCHORS[TASK_TO_ID["elev"]], (len(elev_cls), 1))
        if head_type == "softmax":
            probs = torch.softmax(elev_logits, dim=1).detach().cpu().numpy()
            pred_score[elev_idx.cpu().numpy()] = np.sum(probs * elev_anchors, axis=1)
        else:
            rank = torch.sigmoid(elev_logits).detach().cpu().numpy().sum(axis=1)
            pred_score[elev_idx.cpu().numpy()] = _continuous_score_from_rank(rank, elev_anchors)

    return pred_class, pred_score


def build_class_weights(samples: list[dict[str, Any]]) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean=1."""
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for s in samples:
        counts[int(s["class_id"])] += 1.0
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def ordinal_distance_weights(pred_class: torch.Tensor, true_class: torch.Tensor, scale: float) -> torch.Tensor:
    """Sample-wise penalty weight based on class distance."""
    dist = (pred_class - true_class).abs().float()
    return 1.0 + scale * dist


def compute_loss(
    outputs: dict[str, torch.Tensor],
    class_id: torch.Tensor,
    task_id: torch.Tensor,
    head_type: str,
    class_weights: torch.Tensor,
    distance_penalty: float,
) -> torch.Tensor:
    """Compute weighted loss with additional distance penalty."""
    is_lat = (task_id != TASK_TO_ID["elev"])

    def _per_group(logits_g: torch.Tensor, y_g: torch.Tensor, n_classes: int) -> torch.Tensor:
        if y_g.numel() == 0:
            return logits_g.sum() * 0.0
        if head_type == "softmax":
            per = nn.functional.cross_entropy(
                logits_g, y_g, weight=class_weights[:n_classes], reduction="none"
            )
        elif head_type == "coral":
            levels = coral_levels_from_labels(y_g, n_classes=n_classes)
            per = nn.functional.binary_cross_entropy_with_logits(logits_g, levels, reduction="none").mean(dim=1)
            per = per * class_weights[y_g]
        else:
            probs = torch.sigmoid(logits_g)
            steps = []
            for k in range(n_classes - 1):
                target = (y_g > k).float()
                steps.append(nn.functional.binary_cross_entropy(probs[:, k], target, reduction="none"))
            per = torch.stack(steps, dim=1).mean(dim=1) * class_weights[y_g]
        pred = logits_to_class(logits_g.detach(), head_type).to(y_g.device)
        penalty = ordinal_distance_weights(pred, y_g, scale=distance_penalty)
        return (per * penalty).mean()

    lat_loss = _per_group(outputs["lat_logits"], class_id[is_lat], n_classes=4)
    elev_loss = _per_group(outputs["elev_logits"], class_id[~is_lat], n_classes=3)
    if is_lat.any() and (~is_lat).any():
        return 0.5 * (lat_loss + elev_loss)
    return lat_loss if is_lat.any() else elev_loss


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, head_type: str
) -> pd.DataFrame:
    """Collect per-sample validation predictions for analysis/plots."""
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        image = batch["image"].to(device)
        lm_vec = batch["lm_vec"].to(device)
        lat_lm_vec = batch["lat_lm_vec"].to(device)
        task_id = batch["task_id"].to(device)
        class_id = batch["class_id"].cpu().numpy()
        score = batch["score"].cpu().numpy()
        sample_uids = batch["sample_uid"]

        outputs = model(image, lm_vec, lat_lm_vec, task_id)
        task_ids_np = batch["task_id"].cpu().numpy()
        pred_class, pred_score = decode_outputs(outputs, batch["task_id"], head_type)
        for uid, y_cls, y_score, p_cls, p_score, t_id in zip(
            sample_uids, class_id, score, pred_class, pred_score, task_ids_np
        ):
            rows.append(
                {
                    "sample_uid": str(uid),
                    "task_id": int(t_id),
                    "true_class": int(y_cls),
                    "pred_class": int(p_cls),
                    "true_score": float(y_score),
                    "pred_score": float(p_score),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_type: str,
    class_weights: torch.Tensor,
    distance_penalty: float,
) -> dict[str, float]:
    """Evaluate loss, class accuracy, and continuous-score MAE on a loader."""
    model.eval()
    losses = []
    preds = []
    targets = []
    pred_scores = []
    true_scores = []
    for batch in loader:
        image = batch["image"].to(device)
        lm_vec = batch["lm_vec"].to(device)
        lat_lm_vec = batch["lat_lm_vec"].to(device)
        task_id = batch["task_id"].to(device)
        class_id = batch["class_id"].to(device)
        score = batch["score"].cpu().numpy()

        outputs = model(image, lm_vec, lat_lm_vec, task_id)
        loss = compute_loss(outputs, class_id, task_id, head_type, class_weights, distance_penalty)
        pred_class, pred_score = decode_outputs(outputs, batch["task_id"], head_type)

        losses.append(float(loss.item()))
        preds.extend(pred_class.tolist())
        targets.extend(class_id.cpu().numpy().tolist())
        pred_scores.extend(pred_score.tolist())
        true_scores.extend(score.tolist())

    preds_np = np.array(preds)
    targets_np = np.array(targets)
    pred_scores_np = np.array(pred_scores)
    true_scores_np = np.array(true_scores)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float((preds_np == targets_np).mean()) if len(targets_np) else 0.0,
        "mae": float(np.mean(np.abs(pred_scores_np - true_scores_np))) if len(true_scores_np) else 0.0,
    }


def train_one_model(
    model_name: str,
    head_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    """Train one model variant with early stopping and artifact export."""
    model = MultiModalResNet(
        head_type=head_type,
        pretrained=cfg.use_pretrained_backbone,
        freeze_stages=cfg.freeze_backbone_stages,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
    )

    best_val_mae = float("inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        train_pred_scores = []
        train_true_scores = []

        for batch in train_loader:
            image = batch["image"].to(device)
            lm_vec = batch["lm_vec"].to(device)
            lat_lm_vec = batch["lat_lm_vec"].to(device)
            task_id = batch["task_id"].to(device)
            class_id = batch["class_id"].to(device)
            score = batch["score"].cpu().numpy()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(image, lm_vec, lat_lm_vec, task_id)
            loss = compute_loss(
                outputs, class_id, task_id, head_type, class_weights, cfg.loss_distance_penalty
            )
            loss.backward()
            optimizer.step()

            pred_class, pred_score = decode_outputs(outputs, batch["task_id"], head_type)
            train_losses.append(float(loss.item()))
            train_preds.extend(pred_class.tolist())
            train_targets.extend(class_id.detach().cpu().numpy().tolist())
            train_pred_scores.extend(pred_score.tolist())
            train_true_scores.extend(score.tolist())

        train_preds_np = np.array(train_preds)
        train_targets_np = np.array(train_targets)
        train_pred_scores_np = np.array(train_pred_scores)
        train_true_scores_np = np.array(train_true_scores)
        train_metrics = {
            "loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "acc": float((train_preds_np == train_targets_np).mean()) if len(train_targets_np) else 0.0,
            "mae": float(np.mean(np.abs(train_pred_scores_np - train_true_scores_np))) if len(train_true_scores_np) else 0.0,
        }
        val_metrics = evaluate(
            model, val_loader, device, head_type, class_weights, cfg.loss_distance_penalty
        )
        scheduler.step(val_metrics["mae"])
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_mae": val_metrics["mae"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"train(loss={train_metrics['loss']:.4f},acc={train_metrics['acc']:.3f},mae={train_metrics['mae']:.2f}) "
            f"val(loss={val_metrics['loss']:.4f},acc={val_metrics['acc']:.3f},mae={val_metrics['mae']:.2f})"
        )

        improved = (best_val_mae - val_metrics["mae"]) >= cfg.min_delta
        if cfg.save_all_epochs:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "head_type": head_type,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_val_mae,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "config": asdict(cfg),
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
            )
        if improved:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_name": model_name,
                    "head_type": head_type,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "config": asdict(cfg),
                },
                out_dir / "best_model.pth",
            )
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    torch.save(
        {
            "model_name": model_name,
            "head_type": head_type,
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
            "class_weights": class_weights.detach().cpu().tolist(),
            "config": asdict(cfg),
        },
        out_dir / "last_model.pth",
    )
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    val_preds = collect_predictions(model, val_loader, device, head_type)
    val_preds.to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "model_name": model_name,
        "head_type": head_type,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "last_val_mae": float(history[-1]["val_mae"]) if history else float("nan"),
        "last_val_acc": float(history[-1]["val_acc"]) if history else float("nan"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run full experiment: data prep, 3 model trainings, and summaries."""
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_ts}_k{cfg.n_folds}_s{cfg.n_seeds}"
    run_root = ARTIFACTS_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"Artifacts: {run_root}")
    print(f"Device: {device}")

    samples, skips = build_samples(cfg)
    if skips:
        (run_root / "skipped_samples.txt").write_text("\n".join(skips))
    if not samples:
        raise SystemExit("No usable samples were found. Check overlays/crops/dataset alignment.")

    seed_values = [cfg.seed + i * cfg.seed_stride for i in range(cfg.n_seeds)]
    (run_root / "seeds.json").write_text(json.dumps(seed_values, indent=2))
    print(f"Using seeds: {seed_values}")

    models_to_run = [
        ("resnet50_softmax4", "softmax"),
        ("resnet50_coral", "coral"),
        ("resnet50_corn", "corn"),
    ]
    summaries = []
    for seed in seed_values:
        print(f"\n######## Seed {seed} ########")
        set_seed(seed)
        subjects = np.array([s["subject"] for s in samples], dtype=object)
        folds = make_subject_folds(subjects, cfg.n_folds, seed)
        print(f"Samples total={len(samples)} | subject-wise folds={len(folds)}")

        for fold_idx, (train_subj, val_subj) in enumerate(folds, start=1):
            print(f"\n=== Seed {seed} Fold {fold_idx}/{len(folds)} ===")
            print(f"Train subjects: {sorted(train_subj)}")
            print(f"Val subjects:   {sorted(val_subj)}")
            train_samples = [s for s in samples if s["subject"] in train_subj]
            val_samples = [s for s in samples if s["subject"] in val_subj]
            if not train_samples or not val_samples:
                print(f"Skipping fold {fold_idx}: empty train/val")
                continue

            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            val_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            train_ds = MultiModalDataset(train_samples, image_transform=train_transform, is_train=True)
            val_ds = MultiModalDataset(val_samples, image_transform=val_transform, is_train=False)

            class_weights = build_class_weights(train_samples).to(device)
            sampler = None
            shuffle = True
            if cfg.use_weighted_sampler:
                sample_w = [float(class_weights[int(s["class_id"])].cpu().item()) for s in train_samples]
                sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_w, dtype=torch.double),
                    num_samples=len(sample_w),
                    replacement=True,
                )
                shuffle = False

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            for model_name, head_type in models_to_run:
                out_dir = run_root / f"seed_{seed}" / f"fold_{fold_idx}" / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                summary = train_one_model(
                    model_name=model_name,
                    head_type=head_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=cfg,
                    device=device,
                    out_dir=out_dir,
                    class_weights=class_weights,
                )
                summary["seed"] = seed
                summary["fold"] = fold_idx
                summary["n_train"] = len(train_samples)
                summary["n_val"] = len(val_samples)
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_root / "model_comparison_by_seed_fold.csv", index=False)
    print("\nModel comparison by seed/fold:")
    if not summary_df.empty:
        print(summary_df.sort_values(["model_name", "seed", "fold"]).to_string(index=False))
        agg = (
            summary_df.groupby("model_name", as_index=False)
            .agg(
                runs=("fold", "count"),
                mean_best_val_mae=("best_val_mae", "mean"),
                std_best_val_mae=("best_val_mae", "std"),
                mean_last_val_acc=("last_val_acc", "mean"),
            )
            .sort_values("mean_best_val_mae")
        )
        agg.to_csv(run_root / "model_comparison_aggregate.csv", index=False)
        print("\nAggregate comparison:")
        print(agg.to_string(index=False))

        pred_files = list(run_root.glob("seed_*/fold_*/resnet50_*/val_predictions.csv"))
        if pred_files:
            pred_rows = []
            for pf in pred_files:
                dfp = pd.read_csv(pf)
                if dfp.empty:
                    continue
                seed_token = pf.parts[-4]  # seed_<value>
                fold_token = pf.parts[-3]  # fold_<idx>
                model_token = pf.parts[-2]
                dfp["seed"] = int(seed_token.split("_")[1])
                dfp["fold"] = int(fold_token.split("_")[1])
                dfp["model_name"] = model_token
                pred_rows.append(dfp)
            if pred_rows:
                pred_df = pd.concat(pred_rows, ignore_index=True)
                avg_pred = (
                    pred_df.groupby(
                        ["model_name", "fold", "sample_uid", "task_id", "true_class", "true_score"],
                        as_index=False,
                    )
                    .agg(pred_score_mean=("pred_score", "mean"))
                )
                avg_pred.to_csv(run_root / "avg_predictions_across_seeds.csv", index=False)
                print(f"\nSaved averaged predictions: {run_root / 'avg_predictions_across_seeds.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
