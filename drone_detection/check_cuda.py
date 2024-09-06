"""This file checks cuda."""

from torch import cuda
from ultralytics.utils.checks import cuda_is_available

print("Torch cuda:", cuda.is_available())
print("YOLO cuda:", cuda_is_available())
