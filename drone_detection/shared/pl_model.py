"""This module contains classes related to pytorch lightning model."""
from enum import Enum

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger


class PyLiModel(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn, metrics, lr_scheduler=None):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler

        # self.test_preds = []
        # self.test_labels = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        tb_logger = self.logger
        prototype_array = torch.Tensor(64, 1, 28, 28)
        tb_logger.log_graph(model=self, input_array=prototype_array)

    def _shared_step(self, batch):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        accuracy = self.metrics.accuracy(preds, y)

        self.log(MetricNames.train_loss.value, loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(MetricNames.train_accuracy.value, accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        accuracy = self.metrics.accuracy(preds, y)
        precision = self.metrics.precision(preds, y)

        self.log(MetricNames.val_loss.value, loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(MetricNames.val_accuracy.value, accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(MetricNames.val_precision.value, precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        accuracy = self.metrics.accuracy(preds, y)
        precision = self.metrics.precision(preds, y)

        self.log_dict({
            MetricNames.test_loss.value: loss,
            MetricNames.test_accuracy.value: accuracy,
            MetricNames.test_precision.value: precision,
        },
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

    # def on_test_epoch_end(self) -> None:
    #     test_preds = torch.cat(self.test_preds)
    #     test_labels = torch.cat(self.test_labels)
    #     self.metrics.confusion_matrix(test_preds.cpu().numpy(), test_labels.cpu().numpy())
    #
    #     self.test_preds.clear()
    #     self.test_labels.clear()


    def configure_optimizers(self):
        optimizer = self.optimizer

        if self.lr_scheduler:
            lr_scheduler_config = {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "monitor": MetricNames.val_loss.value,
            }
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }

        return {"optimizer": optimizer}


class CustomTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainer = None

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def _process_metrics(self, metrics, metric_type):
        """Helper function to rename metrics."""
        metric_mappings = {
            MetricNames.train_loss.value: f"{MetricNames.Loss.value}/{MetricNames.train_loss.value}",
            MetricNames.train_accuracy.value: f"{MetricNames.Metrics.value}/{MetricNames.train_accuracy.value}",
            MetricNames.val_loss.value: f"{MetricNames.Loss.value}/{MetricNames.val_loss.value}",
            MetricNames.val_accuracy.value: f"{MetricNames.Metrics.value}/{MetricNames.val_accuracy.value}",
            MetricNames.val_precision.value: f"{MetricNames.Metrics.value}/{MetricNames.val_precision.value}",
            MetricNames.test_loss.value: f"{MetricNames.Test.value}/{MetricNames.test_loss.value}",
            MetricNames.test_accuracy.value: f"{MetricNames.Test.value}/{MetricNames.test_accuracy.value}",
            MetricNames.test_precision.value: f"{MetricNames.Test.value}/{MetricNames.test_precision.value}",
        }

        for metric_name, new_metric_name in metric_mappings.items():
            if metric_name in metrics and metric_name.startswith(metric_type):
                metrics[new_metric_name] = metrics.pop(metric_name)

        return metrics

    def log_metrics(self, metrics, step):
        step = self.trainer.current_epoch

        if self.trainer.state.stage.TRAINING:
            metrics = self._process_metrics(metrics, Stage.train.value)

        if self.trainer.state.stage.VALIDATING:
            metrics = self._process_metrics(metrics, Stage.val.value)
            metrics.pop("epoch", None)

        if self.trainer.state.stage.TESTING:
            metrics = self._process_metrics(metrics, Stage.test.value)
            metrics.pop("epoch", None)

        super().log_metrics(metrics, step)


class CustomTQDMProgressBar(TQDMProgressBar):
    # def init_validation_tqdm(self):
    #     bar = super().init_validation_tqdm()
    #     bar.leave = True
    #     return bar

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class Stage(Enum):
    """This enumeration contains stages."""
    train = "train"
    val = "val"
    test = "test"


class MetricNames(Enum):
    """This enumeration contains metric names."""
    train_loss = f"{Stage.train.value}_loss"
    train_accuracy = f"{Stage.train.value}_acc"

    val_loss = f"{Stage.val.value}_loss"
    val_accuracy = f"{Stage.val.value}_acc"
    val_precision = f"{Stage.val.value}_precision"

    test_loss = f"{Stage.test.value}_loss"
    test_accuracy = f"{Stage.test.value}_acc"
    test_precision = f"{Stage.test.value}_precision"

    Loss = "Loss"
    Metrics = "Metrics"
    Test = "Test"
