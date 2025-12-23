"""
BLS training on MobileNetV3-Small extracted features (image-only) for RAF-DB1.

Inputs:
    data/RAF-DB1/basic/train_mnetv3_solo_raw.npz  (X: [N,576], y, paths)
    data/RAF-DB1/basic/test_mnetv3_solo_raw.npz
Outputs:
    models/bls_mnetv3_scaler.pkl
    models/bls_mnetv3_pca.pkl   (if USE_PCA=True)
    models/bls_mnetv3.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
import argparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add BroadLearning to path
BL_DIR = Path("Broad-Learning-System-master") / "BroadLearning"
if str(BL_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(BL_DIR.resolve()))

# Force CuPy backend; fail fast if unavailable
try:
    from bls_addinput_gpu import broadnet_enhmap_gpu as broadnet_impl  # type: ignore  # noqa: E402
    USING_BACKEND = "GPU(CuPy)"
except Exception as e:  # pragma: no cover - require GPU path
    raise ImportError(f"Failed to import bls_addinput_gpu (CuPy): {e}")

import cupy as cp  # type: ignore

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

FEATURES_PATH = Path("data/RAF-DB1/basic/train_mnetv3_solo_raw.npz")
PUBTEST_FEATURES_PATH = Path("data/RAF-DB1/basic/test_mnetv3_solo_raw.npz")
MODEL_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_PCA = True
PCA_VARIANCE = 0.98
GPU_DTYPE = "float64"

# BLS params tuned for 576D features
BLS_CFG = dict(
    maptimes=22,
    enhencetimes=25,
    mapstep=13,
    enhencestep=17,
    reg=1e-3,
    traintimes=15,
    batchsize=16,
    acc=1,
    map_function="tanh",
    enhence_function="leakyrelu",
    map_whiten=False,
)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    return data["X"].astype(np.float32, copy=False), data["y"].astype(np.int64, copy=False), data["paths"]


class BLSEnhMapEstimator(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around broadnet_enhmap_gpu."""

    def __init__(
        self,
        maptimes=15,
        enhencetimes=18,
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
        kwargs = dict(
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
        self._model = broadnet_impl(**kwargs)
        self._model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)


def evaluate(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return np.mean(preds == y)


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


def train_and_eval(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    base_name: str,
) -> float:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train_p, X_val_p, pca_info, scaler, pca = preprocess_features(X_train, X_val)
    if pca_info:
        print(pca_info)

    model = BLSEnhMapEstimator(
        maptimes=BLS_CFG["maptimes"],
        enhencetimes=BLS_CFG["enhencetimes"],
        batchsize=BLS_CFG["batchsize"],
        acc=BLS_CFG["acc"],
        mapstep=BLS_CFG["mapstep"],
        enhencestep=BLS_CFG["enhencestep"],
        reg=BLS_CFG["reg"],
        map_function=BLS_CFG["map_function"],
        enhence_function=BLS_CFG["enhence_function"],
        map_whiten=BLS_CFG["map_whiten"],
        traintimes=BLS_CFG["traintimes"],
    )
    model.fit(X_train_p, y_train)
    val_acc = evaluate(model, X_val_p, y_val)
    print(f"MobileNetV3-feats-only accuracy (backend={USING_BACKEND}): {val_acc:.4f}")

    # Save artifacts
    try:
        joblib.dump(scaler, MODEL_DIR / f"{base_name}_scaler.pkl")
        if pca is not None:
            joblib.dump(pca, MODEL_DIR / f"{base_name}_pca.pkl")
        joblib.dump(model, MODEL_DIR / f"{base_name}.pkl")
    except Exception as e:
        print(f"Warning: failed to save model artifacts: {e}")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return val_acc


def main() -> None:
    global MODEL_DIR, FEATURES_PATH, PUBTEST_FEATURES_PATH

    parser = argparse.ArgumentParser(description="BLS on MobileNetV3 features (image-only).")
    parser.add_argument("--train-npz", type=Path, default=FEATURES_PATH, help="Path to train NPZ.")
    parser.add_argument("--test-npz", type=Path, default=PUBTEST_FEATURES_PATH, help="Path to pubtest NPZ (optional).")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Directory to save scaler/pca/bls.")
    parser.add_argument("--name", type=str, default="bls_mnetv3", help="Base name for saved model files.")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    FEATURES_PATH = args.train_npz
    PUBTEST_FEATURES_PATH = args.test_npz

    X, y, paths = load_npz(FEATURES_PATH)
    print(f"MobileNetV3 features train: X {X.shape}, y {y.shape}")

    if PUBTEST_FEATURES_PATH.exists():
        X_val, y_val, _ = load_npz(PUBTEST_FEATURES_PATH)
        X_train, y_train = X, y
        print(f"Using PubTest: X_val {X_val.shape}")
    else:
        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X, y, paths, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    train_and_eval(X_train, X_val, y_train, y_val, base_name=args.name)


if __name__ == "__main__":
    main()
