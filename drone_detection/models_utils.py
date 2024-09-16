"""This module contain functions for working with model."""
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from ultralytics import YOLO


def evaluate_model_video(
    model: YOLO,
    path_file: Path,
    params_eval: DictConfig,
) -> npt.NDArray[np.float32]:
    """Evaluate model on video."""
    cap = cv2.VideoCapture(str(path_file))

    cv2.namedWindow("Drone Detection", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(frame, **params_eval)

            annotated_frame = results[0].plot()
            cv2.imshow("Drone Detection", annotated_frame)

            bbox = results[0].boxes.xywhn.cpu().numpy()
            yield bbox

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # if cv2.getWindowProperty("Drone Detection", cv2.WND_PROP_VISIBLE) < 1:
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def open_web_camera_with_model(
    model: YOLO,
    params_eval: DictConfig,
) -> npt.NDArray[np.float32]:
    """Evaluate model on WebCamera."""
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Web Camera")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, **params_eval)

            annotated_frame = results[0].plot()
            cv2.imshow("Web Camera", annotated_frame)

            bbox = results[0].boxes.xywhn.cpu().numpy()
            yield bbox

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty("Web Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
