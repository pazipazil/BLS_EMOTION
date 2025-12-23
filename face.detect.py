from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import joblib

# Optional: MediaPipe Tasks for blendshape extraction
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
except Exception:
    mp = None
    mp_tasks = None
    mp_vision = None

# Make pickled BLS estimator (saved from bls_train.py as __main__.BLSEnhMapEstimator) loadable
try:
    import sys as _sys
    from bls_train import BLSEnhMapEstimator as _BLSCls  # type: ignore

    # expose under __main__ so joblib can resolve the pickled path
    BLSEnhMapEstimator = _BLSCls  # type: ignore
    _sys.modules["__main__"].BLSEnhMapEstimator = _BLSCls  # type: ignore[attr-defined]
except Exception:
    pass


# Pretrained DNN face detector (more accurate than Haar and handles profile faces better)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
# Optional keypoint filter: ensure each detection also contains an eye to reduce false positives
EYE_CASCADE = Path("D:/opencv/1/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")

# Emotion model artifacts (produced by bls_train.py with face_landmarker.task features)
SCALER_PATH = MODEL_DIR / "bls_blendshape_scaler.pkl"
PCA_PATH = MODEL_DIR / "bls_blendshape_pca.pkl"          # optional
BLS_MODEL_PATH = MODEL_DIR / "bls_blendshape.pkl"
CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]

CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
DETECT_INTERVAL = 4  # Run the DNN detector every N frames, use tracker otherwise
SMOOTH_WINDOW = 4    # Moving-average window for bounding-box smoothing
_FACE_NET = None
_FACEMESH = None
_LANDMARKER = None


def load_face_detector(proto_path: Path = FACE_PROTO, model_path: Path = FACE_MODEL) -> cv.dnn_Net:
    """Load the ResNet-SSD face detector weights."""
    if not proto_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Face detector model files missing. Download deploy.prototxt and "
            "res10_300x300_ssd_iter_140000.caffemodel into:\n"
            f"{proto_path.parent}"
        )
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
    """Return face boxes with confidence after thresholding and NMS."""
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


def _create_tracker():
    """Pick an available tracker implementation for the current OpenCV build."""
    creator_chain = [
        lambda: cv.legacy.TrackerCSRT_create(),  # type: ignore[attr-defined]
        lambda: cv.legacy.TrackerKCF_create(),   # type: ignore[attr-defined]
        lambda: cv.TrackerCSRT_create(),         # type: ignore[attr-defined]
        lambda: cv.TrackerKCF_create(),          # type: ignore[attr-defined]
    ]
    for creator in creator_chain:
        try:
            return creator()
        except AttributeError:
            continue
    return None


def get_landmarker():
    """Lazily create MediaPipe Tasks FaceLandmarker with blendshapes."""
    global _LANDMARKER
    if mp_tasks is None or mp_vision is None:
        return None
    if _LANDMARKER is None:
        # Use face_landmarker.task (verified to output 52 blendshapes)
        model_path = MODEL_DIR / "face_landmarker.task"
        if not model_path.exists():
            return None
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


class BoundingBoxSmoother:
    """Sliding-average smoothing on bounding boxes to reduce jitter."""

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
    """Simple wrapper around OpenCV trackers to bridge frames between detections."""

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
    """Combine DNN detection, eye filtering, tracking, and smoothing."""

    def __init__(self) -> None:
        self.net = get_face_net()
        self.eye_classifier = (
            cv.CascadeClassifier(str(EYE_CASCADE)) if EYE_CASCADE.exists() else None
        )
        self.tracker = FaceTracker()
        self.smoother = BoundingBoxSmoother()
        self.frame_idx = 0

    def _filter_with_eyes(
        self, gray: np.ndarray, detections: List[Tuple[List[int], float]]
    ) -> List[Tuple[List[int], float]]:
        """Filter detections by ensuring at least one eye is inside the ROI."""
        if self.eye_classifier is None:
            return detections
        filtered: List[Tuple[List[int], float]] = []
        for (x1, y1, x2, y2), conf in detections:
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            eyes = self.eye_classifier.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=2)
            if len(eyes) >= 1:
                filtered.append(([x1, y1, x2, y2], conf))
        return filtered

    def _pick_best(self, detections: List[Tuple[List[int], float]]) -> Optional[Tuple[List[int], float]]:
        if not detections:
            return None
        return max(detections, key=lambda item: item[1])

    def process(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[float]]]:
        self.frame_idx += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        use_detection = (self.frame_idx % DETECT_INTERVAL == 1) or not self.tracker.active()

        final_box: Optional[Tuple[int, int, int, int]] = None
        confidence: Optional[float] = None

        if use_detection:
            detections = detect_faces_dnn(frame, self.net)
            detections = self._filter_with_eyes(gray, detections)
            best = self._pick_best(detections)
            if best is not None:
                box, confidence = best
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


def draw_detections(frame: np.ndarray, detections):
    """Render detection boxes and confidence labels on the frame."""
    for (x1, y1, x2, y2), conf in detections:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{conf:.2f}" if conf is not None else "track"
        cv.putText(
            frame,
            label,
            (x1, y1 - 8 if y1 - 8 > 8 else y1 + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )


class EmotionPredictor:
    """Load scaler/PCA/BLS模型并对单帧进行情绪预测。"""

    def __init__(self) -> None:
        self.scaler = self._safe_load(SCALER_PATH)
        self.pca = self._safe_load(PCA_PATH)
        self.model = self._safe_load(BLS_MODEL_PATH)
        self.landmarker = get_landmarker()
        print(f"EmotionPredictor init: model={type(self.model)}, scaler={'ok' if self.scaler is not None else 'None'}, pca={'ok' if self.pca is not None else 'None'}")
        if self.model is None:
            print("Warning: BLS 模型未找到，情绪预测将被跳过。")
        if self.landmarker is None:
            print("Warning: 未检测到 MediaPipe FaceLandmarker（含 blendshape），情绪预测将被跳过。")

    @staticmethod
    def _safe_load(path: Path):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: 加载 {path} 失败: {e}")
        return None

    def _extract_blendshape(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """在给定检测框内提取 52 维 blendshape 特征（MediaPipe Tasks）。"""
        if self.landmarker is None or mp is None:
            return None
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        res = self.landmarker.detect(mp_image)
        if not res.face_blendshapes:
            print("no blendshape")
            return None
        blend = res.face_blendshapes[0]
        scores = np.array([c.score for c in blend], dtype=np.float32)
        print("blendshape scores shape:", scores.shape)
        return scores

    def predict(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
        """返回情绪标签字符串；若不可用则返回 None。"""
        if self.model is None:
            return None
        feats = self._extract_blendshape(frame_bgr, box)
        if feats is None or feats.size == 0:
            return None
        feats = feats.reshape(1, -1)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        if self.pca is not None:
            feats = self.pca.transform(feats)
        try:
            pred = int(self.model.predict(feats)[0])
            print("pred label:", pred)
        except Exception as e:
            print("predict error:", e)
            return None
        if 0 <= pred < len(CLASS_NAMES):
            return CLASS_NAMES[pred]
        return str(pred)


def face_detect_demo(frame: np.ndarray, detector: FaceDetector, predictor: EmotionPredictor):
    """Full detection + emotion pipeline on a single frame."""
    detections = detector.process(frame)
    draw_detections(frame, detections)

    # Emotion prediction overlay
    for (x1, y1, x2, y2), _ in detections:
        label = predictor.predict(frame, (x1, y1, x2, y2)) if predictor else None
        if label:
            cv.putText(frame, label, (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow("Face Detection + Emotion", frame)


def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    detector = FaceDetector()
    predictor = EmotionPredictor()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        # Mirror for selfie-view
        frame = cv.flip(frame, 1)
        face_detect_demo(frame, detector, predictor)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
