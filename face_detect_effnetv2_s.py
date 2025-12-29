"""
Realtime face emotion recognition using EfficientNetV2-S features + BLS classifier.

Pipeline:
  webcam frame -> face detection (OpenCV DNN) -> crop face -> EfficientNetV2-S embedding (1280D)
  -> scaler/PCA -> BLS -> emotion label overlay

Prereqs:
  - models/deploy.prototxt
  - models/res10_300x300_ssd_iter_140000.caffemodel
  - models/effnetv2_s_rafdb1_solo.pth
  - models/bls_effnetv2_s.pkl
  - models/bls_effnetv2_s_scaler.pkl
  - models/bls_effnetv2_s_pca.pkl  (optional if PCA used in training)

Run:
  python face_detect_effnetv2_s.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import joblib
import numpy as np
import torch
from torchvision import models, transforms

# Make pickled BLS estimator loadable (saved from bls_train_effnetv2_s.py as __main__.BLSEnhMapEstimator)
try:
    import sys as _sys
    from bls_train_effnetv2_s import BLSEnhMapEstimator as _BLSCls  # type: ignore

    BLSEnhMapEstimator = _BLSCls  # type: ignore
    _sys.modules["__main__"].BLSEnhMapEstimator = _BLSCls  # type: ignore[attr-defined]
except Exception:
    BLSEnhMapEstimator = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Face detector (OpenCV DNN SSD)
FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# EfficientNetV2-S weights (RAF-DB1 finetuned)
EFFNET_WEIGHTS = MODEL_DIR / "effnetv2_s_rafdb1_solo.pth"

# BLS artifacts (EffNetV2-S features)
BLS_MODEL_PATH = MODEL_DIR / "bls_effnetv2_s.pkl"
SCALER_PATH = MODEL_DIR / "bls_effnetv2_s_scaler.pkl"
PCA_PATH = MODEL_DIR / "bls_effnetv2_s_pca.pkl"  # optional

CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]

CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
DETECT_INTERVAL = 4
SMOOTH_WINDOW = 4

# Mirror selfie view
MIRROR = True

_FACE_NET = None


def load_face_detector(proto_path: Path = FACE_PROTO, model_path: Path = FACE_MODEL) -> cv.dnn_Net:
    if not proto_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing face detector files in {proto_path.parent}")
    return cv.dnn.readNetFromCaffe(str(proto_path), str(model_path))


def get_face_net() -> cv.dnn_Net:
    global _FACE_NET
    if _FACE_NET is None:
        _FACE_NET = load_face_detector()
    return _FACE_NET


def detect_faces_dnn(
    frame: np.ndarray,
    net: cv.dnn_Net,
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> List[Tuple[List[int], float]]:
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    boxes: List[List[int]] = []
    confidences: List[float] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)

    picked: List[Tuple[List[int], float]] = []
    if boxes:
        rects = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]
        indices = cv.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                picked.append((boxes[idx], confidences[idx]))
    return picked


class BoundingBoxSmoother:
    def __init__(self, maxlen: int = SMOOTH_WINDOW):
        self.history: deque = deque(maxlen=maxlen)

    def reset(self) -> None:
        self.history.clear()

    def update(self, box: Sequence[int]) -> Tuple[int, int, int, int]:
        arr = np.array(box, dtype=np.float32)
        self.history.append(arr)
        mean_box = np.mean(self.history, axis=0)
        x1, y1, x2, y2 = mean_box.astype(int)
        return x1, y1, x2, y2


def _create_tracker():
    creator_chain = []
    if hasattr(cv, "legacy"):
        creator_chain += [
            getattr(cv.legacy, "TrackerCSRT_create", None),
            getattr(cv.legacy, "TrackerKCF_create", None),
        ]
    creator_chain += [
        getattr(cv, "TrackerCSRT_create", None),
        getattr(cv, "TrackerKCF_create", None),
    ]
    for ctor in creator_chain:
        if ctor:
            try:
                return ctor()
            except Exception:
                continue
    return None


class FaceTracker:
    def __init__(self) -> None:
        self.tracker = None

    def reset(self) -> None:
        self.tracker = None

    @staticmethod
    def _to_xywh(box: Sequence[int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _to_xyxy(box: Sequence[float]) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        return int(x), int(y), int(x + w), int(y + h)

    def init(self, frame: np.ndarray, box: Sequence[int]) -> None:
        tracker = _create_tracker()
        if tracker is None:
            self.tracker = None
            return
        tracker.init(frame, self._to_xywh(box))
        self.tracker = tracker

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.tracker is None:
            return None
        success, xywh = self.tracker.update(frame)
        if not success:
            self.reset()
            return None
        return self._to_xyxy(xywh)

    def active(self) -> bool:
        return self.tracker is not None


class FaceDetector:
    def __init__(self) -> None:
        self.net = get_face_net()
        self.tracker = FaceTracker()
        self.smoother = BoundingBoxSmoother()
        self.frame_idx = 0

    def process(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[float]]]:
        self.frame_idx += 1
        use_detection = (self.frame_idx % DETECT_INTERVAL == 1) or not self.tracker.active()

        final_box: Optional[Tuple[int, int, int, int]] = None
        confidence: Optional[float] = None

        if use_detection:
            detections = detect_faces_dnn(frame, self.net)
            if detections:
                box, confidence = max(detections, key=lambda item: item[1])
                final_box = tuple(box)
                self.tracker.init(frame, final_box)

        if final_box is None:
            tracked_box = self.tracker.update(frame)
            if tracked_box is not None:
                final_box = tracked_box
            else:
                self.smoother.reset()
                return []

        smoothed = self.smoother.update(final_box)
        return [(smoothed, confidence)]


class LabelSmoother:
    """Temporal smoothing for predicted labels to avoid jitter."""

    def __init__(self, maxlen: int = 5):
        self.history: deque = deque(maxlen=maxlen)

    def update(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        self.history.append(label)
        if not self.history:
            return None
        # simple majority vote
        vals, counts = np.unique(list(self.history), return_counts=True)
        return str(vals[int(np.argmax(counts))])


class EffNetV2SFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_v2_s(weights=weights)
        in_feats = self.model.classifier[1].in_features  # 1280
        # match training head for state_dict compatibility
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_feats, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.35),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.35),
            torch.nn.Linear(512, len(CLASS_NAMES)),
        )

    def load_weights(self, path: Path, device: torch.device) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing EfficientNetV2-S weights: {path}")
        state = torch.load(path, map_location=device)
        self.model.load_state_dict(state, strict=True)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.model.features(x)
        emb = torch.flatten(self.model.avgpool(feat_map), 1)  # (B, 1280)
        return emb


class EmotionPredictor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.scaler = self._safe_load(SCALER_PATH)
        self.pca = self._safe_load(PCA_PATH)
        self.bls = self._safe_load(BLS_MODEL_PATH)
        self.label_smoother = LabelSmoother(maxlen=5)

        self.extractor = EffNetV2SFeatureExtractor().to(device)
        self.extractor.load_weights(EFFNET_WEIGHTS, device)
        self.extractor.eval()

        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        mean = getattr(weights, "meta", {}).get("mean", [0.485, 0.456, 0.406])
        std = getattr(weights, "meta", {}).get("std", [0.229, 0.224, 0.225])
        self.tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        print(
            f"Loaded effnetv2_s={EFFNET_WEIGHTS.exists()} | "
            f"bls={'ok' if self.bls else 'None'} | "
            f"scaler={'ok' if self.scaler else 'None'} | "
            f"pca={'ok' if self.pca else 'None'}"
        )

    @staticmethod
    def _safe_load(path: Path):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: 加载 {path} 失败: {e}")
        return None

    def _extract_feature(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        # Square crop with margin + pad
        h, w = frame_bgr.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = int(max(bw, bh) * 1.2)  # +20% margin
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        sx1 = cx - side // 2
        sy1 = cy - side // 2
        sx2 = sx1 + side
        sy2 = sy1 + side

        pad_left = max(0, -sx1)
        pad_top = max(0, -sy1)
        pad_right = max(0, sx2 - w)
        pad_bottom = max(0, sy2 - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            frame_bgr = cv.copyMakeBorder(
                frame_bgr,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            sx1 += pad_left
            sx2 += pad_left
            sy1 += pad_top
            sy2 += pad_top

        crop = frame_bgr[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None

        rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        img = transforms.functional.to_pil_image(rgb)
        x = self.tfm(img).unsqueeze(0).to(self.device)
        emb = self.extractor.embed(x).float().cpu().numpy()  # (1, 1280)
        return emb

    def predict(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
        if self.bls is None:
            return None
        feat = self._extract_feature(frame_bgr, box)
        if feat is None:
            return None
        if self.scaler is not None:
            feat = self.scaler.transform(feat)
        if self.pca is not None:
            feat = self.pca.transform(feat)
        try:
            pred = int(self.bls.predict(feat)[0])
        except Exception as e:
            print("predict error:", e)
            return None
        if 0 <= pred < len(CLASS_NAMES):
            label = CLASS_NAMES[pred]
        else:
            label = str(pred)
        return self.label_smoother.update(label)


def draw_detections(frame: np.ndarray, detections):
    for (x1, y1, x2, y2), conf in detections:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{conf:.2f}" if conf is not None else "track"
        cv.putText(frame, label, (x1, y1 - 8 if y1 - 8 > 8 else y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)


def face_detect_demo(frame: np.ndarray, detector: FaceDetector, predictor: EmotionPredictor):
    detections = detector.process(frame)
    draw_detections(frame, detections)

    for (x1, y1, x2, y2), _ in detections:
        label = predictor.predict(frame, (x1, y1, x2, y2)) if predictor else None
        if label:
            cv.putText(frame, label, (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow("Face Detection + Emotion (EffNetV2-S+BLS)", frame)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    detector = FaceDetector()
    predictor = EmotionPredictor(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MIRROR:  # 镜像后再检测，保持实时预览为自拍视角
            frame = cv.flip(frame, 1)
        face_detect_demo(frame, detector, predictor)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
