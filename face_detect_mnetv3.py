"""
Realtime face emotion recognition using MobileNetV3-Small features + BLS classifier.

- Face detection: SSD (ResNet-10) from face.detect.py
- Feature: MobileNetV3-Small (ImageNet pretrain, finetuned on RAF-DB1)
- Classifier: BLS model trained on 576-D MobileNetV3 features (bls_train_mnetv3.py)
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

# Make pickled BLS estimator loadable
try:
    import sys as _sys
    from bls_train_mnetv3 import BLSEnhMapEstimator as _BLSCls  # type: ignore

    BLSEnhMapEstimator = _BLSCls  # type: ignore
    _sys.modules["__main__"].BLSEnhMapEstimator = _BLSCls  # type: ignore[attr-defined]
except Exception:
    BLSEnhMapEstimator = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Emotion artifacts
MNET_WEIGHTS = MODEL_DIR / "mnetv3_small_rafdb1_solo.pth"
BLS_MODEL_PATH = MODEL_DIR / "bls_mnetv3.pkl"
SCALER_PATH = MODEL_DIR / "bls_mnetv3_scaler.pkl"
PCA_PATH = MODEL_DIR / "bls_mnetv3_pca.pkl"  # optional
CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]

CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
DETECT_INTERVAL = 4
SMOOTH_WINDOW = 4

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
        # Try common trackers; fall back to None
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
        tracker = None
        for ctor in creator_chain:
            if ctor:
                try:
                    tracker = ctor()
                    break
                except Exception:
                    continue
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


class EmotionPredictor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.mnet = self._load_mnet().to(device).eval()
        self.scaler = self._safe_load(SCALER_PATH)
        self.pca = self._safe_load(PCA_PATH)
        self.bls = self._safe_load(BLS_MODEL_PATH)
        if self.bls is None:
            print("Warning: BLS 模型未找到，情绪预测将被跳过。")
        print(
            f"Loaded mnetv3, scaler={'ok' if self.scaler else 'None'}, pca={'ok' if self.pca else 'None'}, bls={'ok' if self.bls else 'None'}"
        )
        self.tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_mnet(self) -> torch.nn.Module:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[0].in_features
        model.classifier = torch.nn.Linear(in_feats, NUM_CLASSES := len(CLASS_NAMES))  # type: ignore
        if not MNET_WEIGHTS.exists():
            print(f"Warning: missing {MNET_WEIGHTS}, using ImageNet weights.")
        else:
            state = torch.load(MNET_WEIGHTS, map_location="cpu")
            model.load_state_dict(state, strict=False)
        return model

    @staticmethod
    def _safe_load(path: Path):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: 加载 {path} 失败: {e}")
        return None

    def _extract_feature(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self.bls is None:
            return None
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        img = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        pil = transforms.functional.to_pil_image(img)
        x = self.tfm(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_map = self.mnet.features(x)
            emb = torch.flatten(self.mnet.avgpool(feat_map), 1).cpu().numpy()
        return emb

    def predict(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
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
            return CLASS_NAMES[pred]
        return str(pred)


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

    cv.imshow("Face Detection + Emotion (MobileNetV3+BLS)", frame)


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
        frame = cv.flip(frame, 1)  # mirror
        face_detect_demo(frame, detector, predictor)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
