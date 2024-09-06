"""This file runs the model."""
from drone_detection.config import DATA_DIR
from drone_detection.models_utils import evaluate_model_video

if __name__ == '__main__':
    data_path = DATA_DIR / "test_video.mp4"
    evaluate_model_video(data_path)
