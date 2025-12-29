"""
Train EfficientNetV2-S on RAF-DB1 (image-only), evaluate, and export raw embeddings.

This mirrors the B2 script settings as much as possible, but uses EffNetV2-S:
- Uses existing NPZs only for (paths, labels):
    data/RAF-DB1/basic/train_blendshapes.npz
    data/RAF-DB1/basic/test_blendshapes.npz
- Outputs:
    models/effnetv2_s_rafdb1_solo.pth
    data/RAF-DB1/basic/train_effnetv2_s_solo_raw.npz  (X: [N,1280], y, paths)
    data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz   (X: [N,1280], y, paths)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


TRAIN_NPZ = Path("data/RAF-DB1/basic/train_blendshapes.npz")
TEST_NPZ = Path("data/RAF-DB1/basic/test_blendshapes.npz")

OUT_CKPT = Path("models/effnetv2_s_rafdb1_solo.pth")
FEAT_TRAIN_NPZ = Path("data/RAF-DB1/basic/train_effnetv2_s_solo_raw.npz")
FEAT_TEST_NPZ = Path("data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz")

NUM_CLASSES = 7
IMG_SIZE = 224  # EffNetV2-S默认 224；也可调大但会更慢

# Training hyperparams
EPOCHS = 25
BATCH_SIZE = 64
LR = 3e-4
WARMUP_EPOCHS = 3
WARMUP_LR = 1e-5
COSINE_T0 = 5
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-5
WEIGHT_DECAY_BACKBONE = 5e-5
WEIGHT_DECAY_HEAD = 1e-4
NUM_WORKERS = 0  # Windows
USE_AMP = True
LABEL_SMOOTHING = 0.15
USE_WEIGHTED_SAMPLER = True

# If checkpoint exists, skip training and only export features
SKIP_TRAIN_IF_CKPT_EXISTS = True


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _prefer_aligned(path: Path) -> Path:
    s = str(path)
    if "Image\\original\\" in s:
        cand = Path(s.replace("Image\\original\\", "Image\\aligned\\"))
        if cand.exists():
            return cand
    if "Image/original/" in s:
        cand = Path(s.replace("Image/original/", "Image/aligned/"))
        if cand.exists():
            return cand
    return path


def load_split(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    data = np.load(npz_path)
    paths = data["paths"].astype(str)
    labels = data["y"].astype(np.int64)
    return paths, labels


class RAFDBDataset(Dataset):
    def __init__(self, paths: np.ndarray, labels: np.ndarray, tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_path = _prefer_aligned(_resolve_path(self.paths[idx]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        x = self.tfm(img)
        y = int(self.labels[idx])
        return x, y


@dataclass
class EpochStats:
    loss: float
    acc: float


def run_epoch(model, loader, optimizer, device, scaler, train: bool) -> EpochStats:
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if train:
                optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.item()) * yb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == yb).sum().item())
        total += yb.size(0)

    return EpochStats(loss=total_loss / max(1, total), acc=total_correct / max(1, total))


@torch.no_grad()
def extract_embeddings(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats_list = []
    labels_list = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feat_map = model.features(xb)
        emb = torch.flatten(model.avgpool(feat_map), 1)  # (B, 1280)
        feats_list.append(emb.float().cpu().numpy())
        labels_list.append(yb.numpy())
    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)
    return feats, labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    train_paths, train_y = load_split(TRAIN_NPZ)
    test_paths, test_y = load_split(TEST_NPZ)

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    mean = getattr(weights, "meta", {}).get("mean", [0.485, 0.456, 0.406])
    std = getattr(weights, "meta", {}).get("std", [0.229, 0.224, 0.225])

    train_tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = RAFDBDataset(train_paths, train_y, train_tfm)
    test_ds = RAFDBDataset(test_paths, test_y, test_tfm)

    if USE_WEIGHTED_SAMPLER:
        cls, counts = np.unique(train_y, return_counts=True)
        w = np.zeros(len(train_y), dtype=np.float64)
        for c, cnt in zip(cls, counts):
            w[train_y == c] = 1.0 / float(cnt)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(train_y), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = models.efficientnet_v2_s(weights=weights)
    in_feats = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(),
        nn.Dropout(p=0.35),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=0.35),
        nn.Linear(512, NUM_CLASSES),
    )
    model.to(device)

    # Layer-wise weight decay
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            param_groups.append({"params": [param], "weight_decay": WEIGHT_DECAY_HEAD})
        else:
            param_groups.append({"params": [param], "weight_decay": WEIGHT_DECAY_BACKBONE})

    optimizer = torch.optim.AdamW(param_groups, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=COSINE_ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_acc = -1.0
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_TRAIN_IF_CKPT_EXISTS and OUT_CKPT.exists():
        print(f"Loading existing checkpoint: {OUT_CKPT}")
        model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
        model.eval()
    else:
        for epoch in range(1, EPOCHS + 1):
            if epoch <= WARMUP_EPOCHS:
                warm_lr = WARMUP_LR + (LR - WARMUP_LR) * (epoch / WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr
            t0 = time.time()
            tr = run_epoch(model, train_loader, optimizer, device, scaler, train=True)
            te = run_epoch(model, test_loader, optimizer, device, None, train=False)
            if epoch > WARMUP_EPOCHS:
                scheduler.step(epoch - WARMUP_EPOCHS)
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"lr {lr_now:.2e} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
                f"val loss {te.loss:.4f} acc {te.acc:.4f} | "
                f"{dt:.1f}s"
            )

            if te.acc > best_acc:
                best_acc = te.acc
                torch.save(model.state_dict(), OUT_CKPT)
                print(f"Saved best checkpoint: {OUT_CKPT} (acc={best_acc:.4f})")

        print(f"Done. Best val acc: {best_acc:.4f}")
        model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
        model.eval()

    # Export embeddings using deterministic transforms (no augmentation)
    export_train_ds = RAFDBDataset(train_paths, train_y, test_tfm)
    export_test_ds = RAFDBDataset(test_paths, test_y, test_tfm)
    export_train_loader = DataLoader(export_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    export_test_loader = DataLoader(export_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    train_feats, train_labels = extract_embeddings(model, export_train_loader, device)
    test_feats, test_labels = extract_embeddings(model, export_test_loader, device)
    print(f"Embeddings: train {train_feats.shape}, test {test_feats.shape}")

    FEAT_TRAIN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEAT_TRAIN_NPZ, X=train_feats, y=train_labels, paths=train_paths)
    np.savez_compressed(FEAT_TEST_NPZ, X=test_feats, y=test_labels, paths=test_paths)
    print(f"Saved: {FEAT_TRAIN_NPZ}")
    print(f"Saved: {FEAT_TEST_NPZ}")


if __name__ == "__main__":
    main()
