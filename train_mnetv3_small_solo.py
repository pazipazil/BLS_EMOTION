"""
Train MobileNetV3-Small on RAF-DB1 (image-only), evaluate, and export fixed-dim embeddings.

Assumptions:
- Use paths/labels from existing blendshape NPZs to keep split一致:
    data/RAF-DB1/basic/train_blendshapes.npz
    data/RAF-DB1/basic/test_blendshapes.npz
- Images位于 Image/aligned 下（若缺失回退到 Image/original）。

Outputs:
- models/mnetv3_small_rafdb1_solo.pth   (最佳验证权重)
- data/RAF-DB1/basic/train_mnetv3_solo_576.npz  (train 特征+标签)
- data/RAF-DB1/basic/test_mnetv3_solo_576.npz   (test 特征+标签)
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.decomposition import PCA


TRAIN_NPZ = Path("data/RAF-DB1/basic/train_blendshapes.npz")
TEST_NPZ = Path("data/RAF-DB1/basic/test_blendshapes.npz")
OUT_CKPT = Path("models/mnetv3_small_rafdb1_solo.pth")
FEAT_TRAIN_NPZ = Path("data/RAF-DB1/basic/train_mnetv3_solo_raw.npz")
FEAT_TEST_NPZ = Path("data/RAF-DB1/basic/test_mnetv3_solo_raw.npz")
NUM_CLASSES = 7
EMBED_DIM = None  # 不做 PCA，保留 576 维原始特征

# 训练超参（适用于 8GB 级别显存）
EPOCHS = 1
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4  # will apply layer-wise tweaks below
NUM_WORKERS = 0  # Windows 下用 0 防止多进程问题
USE_AMP = True

# 快速调试可截断样本
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None


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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_path = _prefer_aligned(_resolve_path(self.paths[idx]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        x = self.tfm(img)
        y = int(self.labels[idx])
        return x, y


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean().item())


@dataclass
class EpochStats:
    loss: float
    acc: float


def run_epoch(model, loader, optimizer, device, scaler, train: bool) -> EpochStats:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    weight = getattr(loader, "class_weights", None)
    if weight is not None:
        weight = weight.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

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


def extract_embeddings(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            feat_map = model.features(xb)  # (B, C, H, W)
            emb = torch.flatten(model.avgpool(feat_map), 1)  # (B, 576)
            feats_list.append(emb.cpu().numpy())
            labels_list.append(yb.numpy())
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    train_paths, train_y = load_split(TRAIN_NPZ)
    test_paths, test_y = load_split(TEST_NPZ)

    if MAX_TRAIN_SAMPLES:
        train_paths, train_y = train_paths[:MAX_TRAIN_SAMPLES], train_y[:MAX_TRAIN_SAMPLES]
    if MAX_TEST_SAMPLES:
        test_paths, test_y = test_paths[:MAX_TEST_SAMPLES], test_y[:MAX_TEST_SAMPLES]

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    mean = getattr(weights, "meta", {}).get("mean", [0.485, 0.456, 0.406])
    std = getattr(weights, "meta", {}).get("std", [0.229, 0.224, 0.225])

    train_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = RAFDBDataset(train_paths, train_y, train_tfm)
    test_ds = RAFDBDataset(test_paths, test_y, test_tfm)

    # 加权损失（按样本频率）
    unique, counts = np.unique(train_y, return_counts=True)
    class_weights = torch.tensor(len(train_y) / (len(unique) * counts), dtype=torch.float32)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )
    # 将 class_weights 挂到 loader 供 run_epoch 使用
    train_loader.class_weights = class_weights
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = models.mobilenet_v3_small(weights=weights)
    in_feats = model.classifier[0].in_features  # 576 after global pooling
    # 增强分类头：Dropout + 隐藏层 + Dropout + 输出
    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, NUM_CLASSES),
    )
    model.to(device)

    # 按层设置 weight decay：特征层较小，分类头较大
    decay_head = 1e-4
    decay_backbone = 5e-5
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            params.append({"params": [param], "weight_decay": decay_head})
        else:
            params.append({"params": [param], "weight_decay": decay_backbone})
    optimizer = torch.optim.AdamW(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    # 如已有训练好的权重，直接加载并跳过训练
    if OUT_CKPT.exists():
        print(f"Loading existing checkpoint: {OUT_CKPT}")
        model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
        model.eval()
    else:
        best_acc = -1.0
        OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, optimizer, device, scaler, train=True)
            te = run_epoch(model, test_loader, optimizer, device, None, train=False)
            dt = time.time() - t0
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
                f"val loss {te.loss:.4f} acc {te.acc:.4f} | "
                f"{dt:.1f}s"
            )
            scheduler.step()
            if te.acc > best_acc:
                best_acc = te.acc
                torch.save(model.state_dict(), OUT_CKPT)
                print(f"Saved best checkpoint: {OUT_CKPT} (acc={best_acc:.4f})")

        print(f"Done. Best val acc: {best_acc:.4f}")
        model.eval()
    train_feats, train_labels = extract_embeddings(model, train_loader, device)
    test_feats, test_labels = extract_embeddings(model, test_loader, device)
    print(f"Raw feats: train {train_feats.shape}, test {test_feats.shape}")

    # 如需 PCA，可设置 EMBED_DIM；默认保留原始 576 维
    if EMBED_DIM:
        pca = PCA(n_components=min(EMBED_DIM, train_feats.shape[1]), random_state=42)
        train_feats = pca.fit_transform(train_feats).astype(np.float32)
        test_feats = pca.transform(test_feats).astype(np.float32)
        print(
            f"PCA -> {train_feats.shape[1]} dims, var_sum={pca.explained_variance_ratio_.sum():.4f}"
        )
    else:
        pca = None
        train_feats = train_feats.astype(np.float32)
        test_feats = test_feats.astype(np.float32)
        print(f"Using raw features: train {train_feats.shape}, test {test_feats.shape}")

    FEAT_TRAIN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEAT_TRAIN_NPZ, X=train_feats, y=train_labels, paths=train_paths)
    np.savez_compressed(FEAT_TEST_NPZ, X=test_feats, y=test_labels, paths=test_paths)
    print(f"Saved features to {FEAT_TRAIN_NPZ} and {FEAT_TEST_NPZ}")


if __name__ == "__main__":
    main()
