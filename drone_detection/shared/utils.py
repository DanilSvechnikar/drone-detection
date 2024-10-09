import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm


def inference_model(model, dataloader, device, return_labels: bool = False):
    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device, dtype=torch.float32)
            logits = model(batch_x)

            labels.append(batch_y.cpu().numpy())
            preds.append(logits.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(preds, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(preds, 0)


def print_cuda_info() -> None:
    """Prints cuda info in console."""
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")


def get_n_workers(num_workers: int) -> int:
    """Returns number of workers."""
    max_cpu_count = mp.cpu_count()
    if num_workers < 0:
        num_workers = max_cpu_count
        # logger.info(f"Parameter `num_workers` is set to {num_workers}")

    num_workers = min(max_cpu_count, num_workers)
    return num_workers
