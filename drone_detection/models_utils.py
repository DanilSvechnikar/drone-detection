"""This module contain functions for working with model."""
import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# You can add more, but hz, I did not check
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4"}


def predict_with_model(
    model: YOLO,
    path_file: Path,
    resize_frame: bool,
    draw_best_box: bool,
    params_tracking: DictConfig,
    enable_camera: bool = False,
) -> npt.NDArray[np.float32]:
    """Predict with model on video or camera and yield bbox."""
    # NOTE: If enable_camera is True, then path_file will be ignored
    logger.info(f"Device Type: {params_tracking.device}")

    if enable_camera:
        cap = cv2.VideoCapture(0)
    elif path_file:
        file_extension = path_file.suffix.lower()
        if file_extension in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path_file))
        elif file_extension in IMAGE_EXTS:
            yield predict_on_image(model, path_file, params_tracking)
            return
        else:
            raise ValueError(f"Unsupported file ext: {path_file}")
    else:
        raise ValueError("Pass the path_file or enable_camera!")

    dsize = (params_tracking.imgsz, params_tracking.imgsz)
    if not resize_frame:
        cv2.namedWindow("Drone Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if resize_frame:
                frame = resize_with_pad(frame, target_size=dsize)

            results = model.track(frame, **params_tracking)
            # results = model.predict(frame, verbose=False, save=False, conf=0.5)

            if draw_best_box and len(results[0]):
                ind_max_conf = get_best_box_ind(results)
                annotated_frame = results[0][ind_max_conf].plot()
                bbox = results[0][ind_max_conf].boxes.xywhn.cpu().numpy()
            else:
                annotated_frame = results[0].plot()
                bbox = results[0].boxes.xywhn.cpu().numpy()

            cv2.imshow("Drone Detection", annotated_frame)
            yield bbox

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_on_image(
    model: YOLO,
    path_file: Path,
    params_tracking: DictConfig,
) -> npt.NDArray[np.float32]:
    """Predict with model on image and return bbox."""
    params_predict = OmegaConf.to_container(params_tracking)
    params_predict["verbose"] = True

    keys_to_remove = {"persist", "tracker"}
    for key in keys_to_remove:
        if key in params_predict:
            params_predict.pop(key)

    results = model.predict(path_file, **params_predict)
    results[0].show()

    return results[0].boxes.xywhn.cpu().numpy()


def get_best_box_ind(results):
    """Return the box ind with the highest probability prediction."""
    return np.argmax(results[0].boxes.conf.cpu())


def resize_with_pad(frame: npt.NDArray, target_size: tuple[int, int]) -> npt.NDArray:
    """Resize image with black padding."""
    height, width = frame.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / width, target_h / height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_frame
