"""This file runs the model."""
import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from drone_detection.config import CONFIG_DIR, DEMO_DATA_DIR, MODELS_DIR
from drone_detection.models_utils import (
    evaluate_model_video,
    open_web_camera_with_model,
)

# Supported video extensions (That's not all!)
video_file_extensions = (".mp4", ".avi",)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="inference.yaml")
def inference(cfg: DictConfig) -> None:
    """Using the model."""

    # Data should be in the path ./drone-detection/data/demo_data/
    data_path = DEMO_DATA_DIR / cfg.name_data
    model_path = MODELS_DIR / cfg.name_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.update(cfg, "params_eval.device", device)

    model = YOLO(model_path)

    if cfg.camera:
        for bbox in open_web_camera_with_model(model, cfg.params_eval):
            print(bbox)
        return

    if data_path.suffix.endswith(video_file_extensions):
        for bbox in evaluate_model_video(model, data_path, cfg.params_eval):
            print(bbox)
        return

    results = model.predict(data_path, **cfg.params_eval)
    results[0].show()


if __name__ == '__main__':
    inference()
