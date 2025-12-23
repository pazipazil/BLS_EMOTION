"""
Extract 52-D blendshape features from RAF-DB1 basic split using list_patition_label.txt.

Label mapping (7 classes):
1: Surprise, 2: Fear, 3: Disgust, 4: Happy, 5: Sad, 6: Angry, 7: Neutral

The label file lists train_XXXX.jpg / test_XXXX.jpg and their labels.
We traverse Image/aligned by default (fallback to Image/original if desired), skip files
containing "ИББО", run Mediapipe Face Landmarker to get 52 ARKit-style blendshape scores,
and save results to CSV/NPZ for train/test separately.

Usage:
    python extract_rafdb1_basic_blendshapes.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATA_ROOT = Path("data/RAF-DB1/basic")
LABEL_FILE = DATA_ROOT / "EmoLabel" / "list_patition_label.txt"
IMAGE_DIR = DATA_ROOT / "Image" / "aligned"  # change to "original" if desired
# Use the blendshape-enabled model (v2)
MODEL_PATH = Path("models/face_landmarker_v2_with_blendshapes.task")
# Distinct output names to avoid clobbering previous runs
OUTPUT_TRAIN_CSV = DATA_ROOT / "train_blendshapes_v2.csv"
OUTPUT_TRAIN_NPZ = DATA_ROOT / "train_blendshapes_v2.npz"
OUTPUT_TEST_CSV = DATA_ROOT / "test_blendshapes_v2.csv"
OUTPUT_TEST_NPZ = DATA_ROOT / "test_blendshapes_v2.npz"

LABEL_MAP: Dict[int, int] = {
    1: 0,  # Surprise -> 0
    2: 1,  # Fear -> 1
    3: 2,  # Disgust -> 2
    4: 3,  # Happy -> 3
    5: 4,  # Sad -> 4
    6: 5,  # Angry -> 5
    7: 6,  # Neutral -> 6
}


def load_splits() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    train_list, test_list = [], []
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_FILE}")
    for line in LABEL_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fname, lab = line.split()
        lab = int(lab)
        mapped_lab = LABEL_MAP.get(lab)
        if mapped_lab is None:
            continue
        if fname.startswith("train_"):
            train_list.append((fname, mapped_lab))
        elif fname.startswith("test_"):
            test_list.append((fname, mapped_lab))
    return train_list, test_list


def extract_split(split: List[Tuple[str, int]]) -> Tuple[List[List[str]], List[np.ndarray], List[int]]:
    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = vision.RunningMode
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    FaceLandmarker = vision.FaceLandmarker

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    rows: List[List[str]] = []
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    decode_fail = detect_fail = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        for idx, (fname, label) in enumerate(split, 1):
            img_path = IMAGE_DIR / fname
            if "ИББО" in img_path.name:
                continue
            if not img_path.exists():
                # try original folder if aligned missing
                alt_path = DATA_ROOT / "Image" / "original" / fname.replace("_aligned", "")
                img_path = alt_path
            try:
                data = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    decode_fail += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            except Exception:
                decode_fail += 1
                continue

            try:
                result = landmarker.detect(mp_image)
            except Exception:
                detect_fail += 1
                continue

            if not result.face_landmarks or not result.face_blendshapes:
                detect_fail += 1
                continue

            blendshape_categories = result.face_blendshapes[0]
            if len(blendshape_categories) < 52:
                continue
            scores = np.array([cat.score for cat in blendshape_categories[:52]], dtype=np.float32)

            row = [str(img_path), label] + [f"{v:.6f}" for v in scores.tolist()]
            rows.append(row)
            X_list.append(scores)
            y_list.append(label)

            if idx % 500 == 0:
                print(f"Processed {idx}/{len(split)} images...")

    print(f"Extracted {len(rows)} samples. Skipped {decode_fail} decode failures, {detect_fail} detect/alignment failures.")
    return rows, X_list, y_list


def save_outputs(csv_path: Path, npz_path: Path, rows: List[List[str]], X_list: List[np.ndarray], y_list: List[int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["path", "label"] + [f"b{i}" for i in range(52)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved CSV to {csv_path}")

    if X_list:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        np.savez_compressed(npz_path, X=X, y=y, paths=np.array([r[0] for r in rows]))
        print(f"Saved NPZ to {npz_path} (X shape {X.shape}, y shape {y.shape})")
    else:
        print(f"No features extracted; NPZ not written for {npz_path}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    train_list, test_list = load_splits()
    print(f"Train samples: {len(train_list)}, Test samples: {len(test_list)}")

    print("Extracting train split...")
    train_rows, train_X, train_y = extract_split(train_list)
    save_outputs(OUTPUT_TRAIN_CSV, OUTPUT_TRAIN_NPZ, train_rows, train_X, train_y)

    print("Extracting test split...")
    test_rows, test_X, test_y = extract_split(test_list)
    save_outputs(OUTPUT_TEST_CSV, OUTPUT_TEST_NPZ, test_rows, test_X, test_y)


if __name__ == "__main__":
    main()
