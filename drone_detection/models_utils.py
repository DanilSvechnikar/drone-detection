"""This module contain functions for working with model."""
from pathlib import Path

import cv2
import torch.cuda
from ultralytics import YOLO

from .config import MODELS_DIR

PARAMS_EVAL = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "imgsz": 640,
    "iou": 0.6,
    "conf": 0.20,
}

name_model = "yolov10n.pt"
MODEL = YOLO(MODELS_DIR / name_model)


def evaluate_model_video(path_file: Path) -> None:
    """Evaluate model on video."""
    cap = cv2.VideoCapture(str(path_file))

    cv2.namedWindow("Drone Detection", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = MODEL(frame, save=False, **PARAMS_EVAL)

            annotated_frame = results[0].plot()
            cv2.imshow("Drone Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty("Drone Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
