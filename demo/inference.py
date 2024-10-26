"""This file runs the model."""

import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from drone_detection import CONFIG_DIR, DEMO_DATA_DIR, MODELS_DIR
from drone_detection.models_utils import predict_with_model


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="inference.yaml")
def inference(cfg: DictConfig) -> None:
    """Using the model."""
    # Data should be in the path ./drone-detection/data/demo_data/
    data_path = DEMO_DATA_DIR / cfg.name_data

    # Model show be in the path ./drone-detection/models/
    model_path = MODELS_DIR / cfg.name_model

    if not torch.cuda.is_available():
        device = "cpu"
        OmegaConf.update(cfg, "params_predict.device", device)

    model = YOLO(model_path)

    for bbox in predict_with_model(
        model=model,
        path_file=data_path,
        resize_frame=cfg.resize_frame,
        draw_best_box=cfg.draw_best_box,
        params_tracking=cfg.params_predict,
        enable_camera=cfg.camera,
    ):
        print(bbox)


if __name__ == '__main__':
    inference()
