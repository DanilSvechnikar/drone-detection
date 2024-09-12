"""This file contains functions related to augmentation."""
from pathlib import Path

import numpy as np


def read_bboxes(label_fpath: Path) -> tuple[list[list[float]], list[int]]:
    """Reads a bbox from a txt file."""
    bboxes = []
    class_labels = []
    with open(label_fpath, 'r') as file:
        for line in file:
            class_label, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_label))

    return bboxes, class_labels


def save_bboxes(
    label_file: Path,
    bboxes: list[np.float64],
    class_labels: list[int],
) -> None:
    """Saves the bbox to a txt file."""
    with open(label_file, 'w') as file:
        for bbox, class_label in zip(bboxes, class_labels):
            file.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
