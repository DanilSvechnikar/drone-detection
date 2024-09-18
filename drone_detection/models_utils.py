"""This module contain functions for working with model."""
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from ultralytics import YOLO


def resize_with_pad(frame: npt.NDArray, target_size: tuple[int, int]) -> npt.NDArray:
    """Resize image with black padding."""
    height, width = frame.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / width, target_h / height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h))

    # Calculate padding
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_frame


def evaluate_model_video(
    model: YOLO,
    path_file: Path,
    resize_frame: bool,
    params_tracking: DictConfig,
) -> npt.NDArray[np.float32]:
    """Evaluate model on video."""
    cap = cv2.VideoCapture(str(path_file))

    dsize = (params_tracking.imgsz, params_tracking.imgsz)
    if not resize_frame:
        cv2.namedWindow("Drone Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if resize_frame:
                frame = resize_with_pad(frame, target_size=dsize)

            results = model.track(frame, **params_tracking)

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
    resize_frame: bool,
    params_tracking: DictConfig,
) -> npt.NDArray[np.float32]:
    """Evaluate model on WebCamera."""
    cap = cv2.VideoCapture(0)

    dsize = (params_tracking.imgsz, params_tracking.imgsz)
    if not resize_frame:
        cv2.namedWindow("Web Camera")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if resize_frame:
                frame = resize_with_pad(frame, target_size=dsize)

            results = model.track(frame, **params_tracking)

            annotated_frame = results[0].plot()
            cv2.imshow("Web Camera", annotated_frame)

            bbox = results[0].boxes.xywhn.cpu().numpy()
            yield bbox

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # if cv2.getWindowProperty("Web Camera", cv2.WND_PROP_VISIBLE) < 1:
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
