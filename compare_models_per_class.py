"""
Compare per-class accuracy between two models on the same 7-class emotion set:
  1) Mediapipe blendshape BLS (52D features)
  2) EfficientNetV2-S BLS (1280D features)

Inputs (defaults):
  - data/RAF-DB1/basic/test_blendshapes.npz          (X: [N,52], y)
  - models/bls_blendshape.pkl, bls_scaler.pkl, bls_pca.pkl
  - data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz (X: [N,1280], y)
  - models/bls_effnetv2_s.pkl, bls_effnetv2_s_scaler.pkl, bls_effnetv2_s_pca.pkl

Usage:
  python compare_models_per_class.py
  python compare_models_per_class.py --blend-npz ... --blend-model ... --eff-npz ... --eff-model ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make pickled BLS estimators loadable
try:
    from bls_train import BLSEnhMapEstimator as _BLSBlend  # type: ignore
except Exception:
    _BLSBlend = None

try:
    from bls_train_effnetv2_s import BLSEnhMapEstimator as _BLSEff  # type: ignore
except Exception:
    _BLSEff = None


CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data["X"].astype(np.float32, copy=False), data["y"].astype(np.int64, copy=False)


def load_artifacts(model_path: Path, scaler_path: Path, pca_path: Path, alias_cls) -> tuple[object, Optional[object], Optional[object]]:
    if alias_cls is not None:
        sys.modules["__main__"].BLSEnhMapEstimator = alias_cls  # ensure pickle can resolve
    model = joblib.load(model_path) if model_path.exists() else None
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    pca = joblib.load(pca_path) if pca_path.exists() else None
    if model is None:
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model, scaler, pca


def preprocess(X: np.ndarray, scaler, pca) -> np.ndarray:
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X.astype(np.float32, copy=False)


def _expected_input_dim(model) -> Optional[int]:
    # broadnet_enhmap_gpu stores mapping_generator.Wlist[0] with shape (input_dim, ?)
    try:
        return int(model._model.mapping_generator.Wlist[0].shape[0])  # type: ignore[attr-defined]
    except Exception:
        return None


def eval_model(X: np.ndarray, y: np.ndarray, model, scaler, pca) -> Tuple[float, np.ndarray]:
    Xp = preprocess(X, scaler, pca)
    exp_dim = _expected_input_dim(model)
    if exp_dim is not None and Xp.shape[1] != exp_dim:
        raise ValueError(
            f"Feature dim mismatch: model expects {exp_dim}, but got {Xp.shape[1]}. "
            f"Check that NPZ, scaler, PCA, and model correspond; if PCA was used during training, ensure the PCA file is provided."
        )
    preds = model.predict(Xp)
    overall = float(np.mean(preds == y))
    per_class = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for cid in range(len(CLASS_NAMES)):
        mask = y == cid
        if mask.any():
            per_class[cid] = float(np.mean(preds[mask] == y[mask]))
        else:
            per_class[cid] = np.nan
    return overall, per_class


def main():
    parser = argparse.ArgumentParser(description="Compare per-class accuracy of Mediapipe BLS vs EffNetV2-S BLS.")
    parser.add_argument("--blend-npz", type=Path, default=Path("data/RAF-DB1/basic/test_blendshapes.npz"))
    parser.add_argument("--blend-model", type=Path, default=Path("models/bls_blendshape.pkl"))
    parser.add_argument("--blend-scaler", type=Path, default=Path("models/bls_blendshape_scaler.pkl"))
    parser.add_argument("--blend-pca", type=Path, default=Path("models/bls_blendshape_pca.pkl"))
    parser.add_argument("--eff-npz", type=Path, default=Path("data/RAF-DB1/basic/test_effnetv2_s_solo_raw.npz"))
    parser.add_argument("--eff-model", type=Path, default=Path("models/bls_effnetv2_s.pkl"))
    parser.add_argument("--eff-scaler", type=Path, default=Path("models/bls_effnetv2_s_scaler.pkl"))
    parser.add_argument("--eff-pca", type=Path, default=Path("models/bls_effnetv2_s_pca.pkl"))
    parser.add_argument("--fig-out", type=Path, default=Path("compare_models_per_class.png"), help="Path to save bar chart")
    parser.add_argument("--no-fig", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    Xb, yb = load_npz(args.blend_npz)
    blend_model, blend_scaler, blend_pca = load_artifacts(args.blend_model, args.blend_scaler, args.blend_pca, _BLSBlend)

    Xe, ye = load_npz(args.eff_npz)
    eff_model, eff_scaler, eff_pca = load_artifacts(args.eff_model, args.eff_scaler, args.eff_pca, _BLSEff)

    if len(yb) != len(ye):
        print(f"Warning: sample sizes differ (blend {len(yb)} vs eff {len(ye)}); per-class comparison still shown.")

    acc_b, pc_b = eval_model(Xb, yb, blend_model, blend_scaler, blend_pca)
    acc_e, pc_e = eval_model(Xe, ye, eff_model, eff_scaler, eff_pca)

    print(f"Overall accuracy:")
    print(f"  Mediapipe BLS: {acc_b:.4f}")
    print(f"  EffNetV2-S BLS: {acc_e:.4f}")
    print("\nPer-class accuracy:")
    print(f"{'Class':<12} {'Blendshape':>12} {'EffNetV2-S':>12}")
    for name, pb, pe in zip(CLASS_NAMES, pc_b, pc_e):
        pb_s = f"{pb:.4f}" if not np.isnan(pb) else "n/a"
        pe_s = f"{pe:.4f}" if not np.isnan(pe) else "n/a"
        print(f"{name:<12} {pb_s:>12} {pe_s:>12}")

    # Identify hardest classes per model
    hardest_blend = np.nanargmin(pc_b)
    hardest_eff = np.nanargmin(pc_e)
    print("\nHardest classes (lower acc):")
    print(f"  Mediapipe BLS: {CLASS_NAMES[hardest_blend]} ({pc_b[hardest_blend]:.4f})")
    print(f"  EffNetV2-S BLS: {CLASS_NAMES[hardest_eff]} ({pc_e[hardest_eff]:.4f})")

    if not args.no_fig:
        # Replace nan with 0 for plotting
        pc_b_plot = np.nan_to_num(pc_b, nan=0.0)
        pc_e_plot = np.nan_to_num(pc_e, nan=0.0)
        x = np.arange(len(CLASS_NAMES))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.bar(x - width / 2, pc_b_plot, width, label="Mediapipe BLS")
        ax.bar(x + width / 2, pc_e_plot, width, label="EffNetV2-S BLS")
        # Overall accuracy bars on the right
        xo = len(CLASS_NAMES) + 1
        ax.bar(xo - width / 2, acc_b, width, color="#1f77b4", alpha=0.7)
        ax.bar(xo + width / 2, acc_e, width, color="#ff7f0e", alpha=0.7)
        ax.set_xticks(list(x) + [xo])
        ax.set_xticklabels(CLASS_NAMES + ["overall"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-class accuracy comparison")
        ax.legend()
        fig.tight_layout()
        args.fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_out, dpi=200)
        plt.close(fig)
        print(f"\nSaved comparison figure to: {args.fig_out}")


if __name__ == "__main__":
    main()
