"""This file runs the model."""
import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from drone_detection.config import CONFIG_DIR, DEMO_DATA_DIR, MODELS_DIR
from drone_detection.models_utils import evaluate_model_video


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="inference.yaml")
def inference(cfg: DictConfig) -> None:
    """Using the model."""

    # Data should be in the path ./drone-detection/data/demo_data/
    data_path = DEMO_DATA_DIR / cfg.name_data
    model_path = MODELS_DIR / cfg.name_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.update(cfg, "params_eval.device", device)

    model = YOLO(model_path)
    if "mp4" in cfg.name_data:
        evaluate_model_video(model, data_path, cfg.params_eval)
        return

    results = model.predict(data_path, **cfg.params_eval)
    results[0].show()


if __name__ == '__main__':
    inference()
