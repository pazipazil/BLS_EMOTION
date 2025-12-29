"""
Mediapipe Face Landmarker verification script that outputs 52 blendshape values.

This script uses the MediaPipe Tasks FaceLandmarker (with blendshape support) to
detect 478 landmarks and compute ARKit-compatible blendshapes. You must provide
the `.task` model file that ships with MediaPipe (e.g.
`face_landmarker_v2_with_blendshapes.task`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


DEFAULT_IMAGE = Path("face2.jpg")
DEFAULT_MODEL = Path("models/face_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to the image used for verification.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the MediaPipe face landmarker task file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = vision.RunningMode
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    mp_image = mp.Image.create_from_file(str(args.image))

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("FaceLandmarker did not detect any faces.")

    landmarks = result.face_landmarks[0]
    print(f"Detected {len(landmarks)} landmarks for the first face (expected 478).")

    first_points = [(lm.x, lm.y, lm.z) for lm in landmarks[:5]]
    print("First 5 normalized landmarks (x, y, z):")
    for point in first_points:
        print(f"  {point}")

    if not result.face_blendshapes:
        raise RuntimeError("Blendshape outputs are empty. "
                           "Ensure the task model includes blendshape heads.")

    blendshape_categories = result.face_blendshapes[0]
    if len(blendshape_categories) < 52:
        raise RuntimeError(
            f"Expected at least 52 blendshape values, but got {len(blendshape_categories)}."
        )

    top52 = blendshape_categories[:52]
    names = [cat.category_name for cat in top52]
    scores = np.array([cat.score for cat in top52], dtype=np.float32)
    print("Blendshape names and scores (first 52 entries):")
    for name, score in zip(names, scores):
        print(f"  {name:<32}: {score:.6f}")

    print("52-D blendshape vector:")
    np.set_printoptions(precision=6, suppress=True)
    print(scores)
    print("Mediapipe Face Landmarker verification completed successfully.")


if __name__ == "__main__":
    main()
