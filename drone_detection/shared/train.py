from collections import namedtuple
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from clearml import Task
from fcnn import FCNN
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig
from pl_model import CustomTensorBoardLogger, CustomTQDMProgressBar, PyLiModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
)

ROOT_PATH = Path().resolve()
CONFIG_PATH = ROOT_PATH


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="train")
def train_model(cfg: DictConfig):
    if cfg.clearml_enable:
        reuse_last_task_id = False
        if cfg.resume_train:
            reuse_last_task_id = True

        task = Task.init(
            project_name=cfg.clearml_proj_name,
            task_name=cfg.task_name,
            reuse_last_task_id=reuse_last_task_id,
        )

        task.connect(cfg)

    dataset_path = ROOT_PATH / cfg.dataset_path
    entire_dataset = datasets.MNIST(
        root=dataset_path, train=True, transform=transforms.ToTensor(), download=True,
    )
    train_ds, val_ds = random_split(entire_dataset, [50000, 10000])

    test_ds = datasets.MNIST(
        root=dataset_path, train=False, transform=transforms.ToTensor(), download=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fcnn = FCNN(
        inp_size=cfg.model.input_size,
        hid_neurons=cfg.model.num_hid_neurons,
        num_cls=cfg.model.num_classes,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fcnn.parameters(), lr=cfg.lr)

    lr_scheduler = None
    if cfg.lr_scheduler_enable:
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )

    Metrics = namedtuple("Metrics", ["accuracy", "precision", "conf_matrix"])
    metrics = Metrics(
        accuracy=MulticlassAccuracy(num_classes=len(cfg.class_labels)),
        precision=MulticlassPrecision(num_classes=len(cfg.class_labels)),
        conf_matrix=MulticlassConfusionMatrix(num_classes=len(cfg.class_labels)),
    )

    model = PyLiModel(
        model=fcnn,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        lr_scheduler=lr_scheduler,
    )

    if cfg.seed_everything_enable:
        L.seed_everything(seed=cfg.seed, workers=cfg.seed_workers)

    profiler = None
    if cfg.profiler_enable:
        profiler = "simple"
        # profiler = PyTorchProfiler()

    save_dir = ROOT_PATH / cfg.save_dir
    tb_logger = CustomTensorBoardLogger(save_dir=save_dir)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience),
        CustomTQDMProgressBar(leave=True),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(monitor="val_loss", filename="best", save_last=True),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=device.type,
        deterministic=cfg.deterministic,
        profiler=profiler,
        logger=tb_logger,
        callbacks=callbacks,
        # fast_dev_run=True,
        # limit_train_batches=0.3,
    )
    tb_logger.trainer = trainer

    ckpt_path = None
    if cfg.resume_train:
        ckpt_path = ROOT_PATH / cfg.model_path

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    train_model()
