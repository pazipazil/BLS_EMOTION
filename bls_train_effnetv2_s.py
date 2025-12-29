"""
Extract EfficientNetV2-S features on RAF-DB1 (image-only) using the fine-tuned
checkpoint, then train a BLS classifier on the extracted embeddings.

Inputs:
  - data/RAF-DB1/basic/train_blendshapes.npz  (paths, labels)
  - data/RAF-DB1/basic/test_blendshapes.npz   (paths, labels)
  - models/effnetv2_s_rafdb1_solo.pth         (fine-tuned backbone)

Outputs:
  - data/RAF-DB1/basic/train_effnetv2_s_solo_raw.npz (X: [N,1280], y, paths)
  - data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz
  - models/bls_effnetv2_s.pkl
  - models/bls_effnetv2_s_scaler.pkl
  - models/bls_effnetv2_s_pca.pkl  (if USE_PCA=True)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Add BroadLearning to path
BL_DIR = Path("Broad-Learning-System-master") / "BroadLearning"
if str(BL_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(BL_DIR.resolve()))

# Force CuPy backend; fail fast if unavailable
try:
    from bls_addinput_gpu import broadnet_enhmap_gpu as broadnet_impl  # type: ignore  # noqa: E402
    USING_BACKEND = "GPU(CuPy)"
except Exception as e:  # pragma: no cover
    raise ImportError(f"Failed to import bls_addinput_gpu (CuPy): {e}")

import cupy as cp  # type: ignore  # noqa: E402

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()


# Defaults
TRAIN_NPZ = Path("data/RAF-DB1/basic/train_blendshapes.npz")
TEST_NPZ = Path("data/RAF-DB1/basic/test_blendshapes.npz")
CKPT_PATH = Path("models/effnetv2_s_rafdb1_solo.pth")
FEAT_TRAIN_NPZ = Path("data/RAF-DB1/basic/train_effnetv2_s_solo_raw.npz")
FEAT_TEST_NPZ = Path("data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz")
MODEL_DIR = Path("models")
MODEL_NAME = "bls_effnetv2_s"

IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 0  # Windows
USE_AMP = True

# BLS params for 1280-D features
GPU_DTYPE = "float32"
BLS_CFG = dict(
    maptimes=16,
    enhencetimes=22,
    mapstep=12,
    enhencestep=14,
    reg=1e-3,
    traintimes=15,
    batchsize=8,
    acc=1,
    map_function="tanh",
    enhence_function="leakyrelu",
    map_whiten=True,
)

USE_PCA = True
PCA_VARIANCE = 0.95
RANDOM_STATE = 42


def load_split(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    data = np.load(npz_path)
    paths = data["paths"].astype(str)
    labels = data["y"].astype(np.int64)
    return paths, labels


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
        return x, y, str(img_path)


@dataclass
class FeatPack:
    X: np.ndarray
    y: np.ndarray
    paths: np.ndarray


def extract_features(model, loader, device) -> FeatPack:
    model.eval()
    feats, labels, all_paths = [], [], []
    with torch.no_grad():
        for xb, yb, pb in loader:
            xb = xb.to(device, non_blocking=True)
            feat_map = model.features(xb)
            emb = torch.flatten(model.avgpool(feat_map), 1)  # (B, 1280)
            feats.append(emb.float().cpu().numpy())
            labels.append(yb.numpy())
            all_paths.extend(pb)
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return FeatPack(X=X, y=y, paths=np.array(all_paths))


def load_feat_npz(path: Path) -> FeatPack:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return FeatPack(
        X=data["X"].astype(np.float32, copy=False),
        y=data["y"].astype(np.int64, copy=False),
        paths=data["paths"],
    )


class BLSEnhMapEstimator:
    def __init__(
        self,
        maptimes=12,
        enhencetimes=14,
        batchsize=8,
        acc=1,
        mapstep=8,
        enhencestep=10,
        reg=1e-3,
        map_function="tanh",
        enhence_function="leakyrelu",
        map_whiten=False,
        traintimes=12,
    ):
        self.maptimes = maptimes
        self.enhencetimes = enhencetimes
        self.batchsize = batchsize
        self.acc = acc
        self.mapstep = mapstep
        self.enhencestep = enhencestep
        self.reg = reg
        self.map_function = map_function
        self.enhence_function = enhence_function
        self.map_whiten = map_whiten
        self.traintimes = traintimes
        self._model = None

    def fit(self, X, y):
        self._model = broadnet_impl(
            maptimes=self.maptimes,
            enhencetimes=self.enhencetimes,
            traintimes=self.traintimes,
            map_function=self.map_function,
            enhence_function=self.enhence_function,
            batchsize=self.batchsize,
            acc=self.acc,
            mapstep=self.mapstep,
            enhencestep=self.enhencestep,
            reg=self.reg,
            map_whiten=self.map_whiten,
            use_internal_scaler=False,
            dtype=GPU_DTYPE,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)


def preprocess_features(
    X_train: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Optional[str], StandardScaler, Optional[PCA]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    pca = None
    if USE_PCA:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train).astype(np.float32)
        X_val = pca.transform(X_val).astype(np.float32)
        info = f"PCA: components={pca.n_components_}, var_sum={pca.explained_variance_ratio_.sum():.4f}"
        return X_train, X_val, info, scaler, pca
    return X_train, X_val, None, scaler, pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EffNetV2-S features and train BLS.")
    parser.add_argument("--train-npz", type=Path, default=TRAIN_NPZ, help="NPZ with train paths/labels")
    parser.add_argument("--test-npz", type=Path, default=TEST_NPZ, help="NPZ with test paths/labels")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH, help="Fine-tuned EfficientNetV2-S checkpoint")
    parser.add_argument("--feat-train-npz", type=Path, default=FEAT_TRAIN_NPZ, help="Where to save train features")
    parser.add_argument("--feat-test-npz", type=Path, default=FEAT_TEST_NPZ, help="Where to save test features")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Where to save BLS/scaler/PCA")
    parser.add_argument("--name", type=str, default=MODEL_NAME, help="Base name for saved BLS artifacts")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction even if NPZ exists")
    args = parser.parse_args()

    global USE_PCA
    if args.no_pca:
        USE_PCA = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    train_paths, train_y = load_split(args.train_npz)
    test_paths, test_y = load_split(args.test_npz)

    # Transforms (deterministic)
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    mean = getattr(weights, "meta", {}).get("mean", [0.485, 0.456, 0.406])
    std = getattr(weights, "meta", {}).get("std", [0.229, 0.224, 0.225])
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Prepare model
    model = models.efficientnet_v2_s(weights=None)
    in_feats = model.classifier[1].in_features  # 1280
    model.classifier = nn.Identity()  # not used; we extract features
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Extract features (reuse if cached)
    if not args.force_extract and args.feat_train_npz.exists() and args.feat_test_npz.exists():
        print(f"Loading cached features: {args.feat_train_npz} / {args.feat_test_npz}")
        train_feats = load_feat_npz(args.feat_train_npz)
        test_feats = load_feat_npz(args.feat_test_npz)
    else:
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            RAFDBDataset(train_paths, train_y, tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )
        test_loader = DataLoader(
            RAFDBDataset(test_paths, test_y, tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )
        t0 = time.time()
        train_feats = extract_features(model, train_loader, device)
        test_feats = extract_features(model, test_loader, device)
        dt = time.time() - t0
        print(f"Extracted train {train_feats.X.shape}, test {test_feats.X.shape} in {dt:.1f}s")
        args.feat_train_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feat_train_npz, X=train_feats.X, y=train_feats.y, paths=train_feats.paths)
        np.savez_compressed(args.feat_test_npz, X=test_feats.X, y=test_feats.y, paths=test_feats.paths)
        print(f"Saved features to {args.feat_train_npz} and {args.feat_test_npz}")

    X_train, y_train = train_feats.X, train_feats.y
    X_val, y_val = test_feats.X, test_feats.y

    X_train_p, X_val_p, info, scaler, pca = preprocess_features(X_train, X_val)
    if info:
        print(info)

    model_bls = BLSEnhMapEstimator(**BLS_CFG)
    model_bls.fit(X_train_p, y_train)
    preds = model_bls.predict(X_val_p)
    acc = float(np.mean(preds == y_val))
    print(f"BLS accuracy on pub-test (backend={USING_BACKEND}): {acc:.4f}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.model_dir / f"{args.name}_scaler.pkl")
    if pca is not None:
        joblib.dump(pca, args.model_dir / f"{args.name}_pca.pkl")
    joblib.dump(model_bls, args.model_dir / f"{args.name}.pkl")
    print(f"Saved BLS artifacts to {args.model_dir}")


if __name__ == "__main__":
    main()
