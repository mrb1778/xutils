import os
import time
from typing import Tuple, Any, Optional, Literal, Union, Callable

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder, \
    TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Precision, Recall, Accuracy, MetricTracker
from torchmetrics.functional import stat_scores

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing.io import IO

from xutils.core.python_utils import getattr_ignore_case
import xutils.data.data_utils as du
import xutils.dl.pytorch.utils as toru
import xutils.core.file_utils as fu
# import xutils.data.plot_utils as pltu

import numpy as np

torch.autograd.set_detect_anomaly(True)


class WrapperModule(pl.LightningModule):
    def __init__(self,
                 wrapped: pl.LightningModule,
                 learning_rate: float,
                 output_type: Literal["binary", "multiclass", "multilabel"] = "binary",
                 loss_fn: Union[str, Callable] = F.cross_entropy,
                 batch_size: int = 10):  # , num_classes=None):
        super(WrapperModule, self).__init__()
        self.model: pl.LightningModule = wrapped
        self.model.to(toru.get_device())

        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate', 'loss_fn', 'batch_size')

        self.loss_fn = toru.parse_loss_fn(loss_fn)

        # if num_classes is not None:
        #     self.precision = Precision(num_classes)
        #     self.recall = Recall(num_classes)
        train_type = output_type if output_type != 'onehot' else 'multiclass'
        self.accuracy = Accuracy(task=train_type)
        # self.train_loss_tracker = EMATracker(alpha=0.02)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("log", batch_idx % self.trainer.accumulate_grad_batches == 0)
        return loss

    # def _should_log(self, flag):
    #     if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
    #         if isinstance(flag, list):
    #             return flag[0]
    #         return flag
    #     return False

    # def training_step_end(self, outputs):
    #     # Aggregate the losses from all GPUs
    #     loss = outputs["loss"].mean()
    #     self.train_loss_tracker.update(loss.detach())
    #     if self._should_log(outputs["log"]):
    #         self.logger.log_metrics({
    #             "train_loss": self.train_loss_tracker.value
    #         }, step=self.global_step)
    #
    #     self.log("loss", loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # self.log_dict(metrics)
        # print("------val_acc", self.accuracy(y_hat, torch.argmax(y.squeeze(), dim=1)).cpu())
        self.log("time_stamp", time.time())

        # todo: figure out dimensions / unsqueeze
        # self.log("val_loss", self.loss(y_hat, y))
        # self.log("val_acc", self.accuracy(y_hat, torch.argmax(y.squeeze(), dim=1)))
        self.log("val_loss", self.loss(y_hat, y))
        # todo: changes based on loss fn? int vs not?
        # self.log("val_acc", self.accuracy(y_hat, y.int()))
        # print("104 dtype", y_hat.dtype, y.dtype)
        # print("104 type", type(y))
        # exit()
        # todo: see if alwways int? for y https://torchmetrics.readthedocs.io/en/stable/pages/classification.html#using-the-multiclass-parameter
        # self.log("val_acc", self.accuracy(y_hat, y.int()))
        # todo: custom tf onehot handling
        # todo: data type handling
        self.log("val_acc",
                 self.accuracy(y_hat,
                               torch.argmax(y.squeeze(), dim=1) if y.shape[1] > 1 else y.int()),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        # if self.recall:
        #     metrics["recall"] = self.recall(y_hat, y)
        # if self.precision:
        #     metrics["precision"] = self.precision(y_hat, y)

    # noinspection PyMethodMayBeStatic
    # def validation_end(self, outputs):
    #     print("validation_end", outputs[0].keys())
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #
    #     self.log('val_loss', avg_loss)
    #     self.log('val_acc', avg_acc)
    #     # self.log('progress_bar', {'val_loss': avg_loss, 'val_acc': avg_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # self.log_dict(metrics)
        self.log("y_hat", y_hat)
        self.log("y", y)
        self.log("val_loss", self.loss(y_hat, y))

    # def test_epoch_end(self, outputs):
    #     y_hat = torch.cat([tmp['y_hat'] for tmp in outputs])
    #     y = torch.cat([tmp['y'] for tmp in outputs])
    #     du.compare_results(actual=y_hat.cpu().numpy(),
    #                        actual_hot_encoded=True,
    #                        predicted=y.cpu().numpy(),
    #                        predicted_hot_encoded=True)
    #     # confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=10)
    #     #
    #     # df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(10), columns=range(10))
    #     # plt.figure(figsize=(10, 7))
    #     # fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
    #     # plt.close(fig_)
    #     #
    #     # self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def loss(self, y_hat, y):
        # todo: figure out single dimensions
        # return self.loss_fn(y_hat, y.unsqueeze(1))
        return self.loss_fn(y_hat, y)


class EMATracker:
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = new_value
        else:
            self._value = (
                    new_value * self.alpha +
                    self._value * (1 - self.alpha)
            )

    @property
    def value(self):
        return self._value


class NumpyXYDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        y = self.y[idx]
        # todo: determine type?
        return torch.from_numpy(self.x[idx]).float(), \
            torch.from_numpy(y) if isinstance(y, np.ndarray) else \
                torch.tensor([y]).float()
        # return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx])


class DatasetDataModule(pl.LightningDataModule):
    # TODO: add train test split

    def __init__(self,
                 train_dataset=None,
                 test_dataset=None,
                 val_dataset=None,
                 batch_size=4096,
                 num_workers=4):
        super().__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     pass

    # def prepare_data(self):
    #     pass
    #
    # def setup(self, stage=None):
    #     pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)


# PANDAS --------------------------------------


class PandasDataset(Dataset):
    # todo: test is y squeeze makes sense for single column

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), torch.LongTensor(self.targets[idx]).squeeze()


class PandasDataModule(DatasetDataModule):
    def __init__(self,
                 features_col, targets_col,
                 train_df,
                 validation_df=None,
                 test_df=None,
                 batch_size=4096,
                 num_workers=4):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)

        self.features_col = features_col
        self.targets_col = targets_col

        self.train_df = train_df
        self.val_df = validation_df
        self.test_df = test_df

    def setup(self, stage=None):
        self.train_dataset = PandasDataset(self.train_df[self.features_col].values,
                                           self.train_df[self.targets_col].values)

        if self.val_df:
            self.val_dataset = PandasDataset(self.val_df[self.features_col].values,
                                             self.val_df[self.targets_col].values)

        if self.test_df:
            self.test_dataset = PandasDataset(self.test_df[self.features_col].values,
                                              self.test_df[self.targets_col].values)


def load_model(model: Union[pl.LightningModule, Callable[[Any], pl.LightningModule]], path: Union[str, IO], model_kwargs, wrap=False) -> pl.LightningModule:
    if wrap:
        model = wrap_model(model)
    model = model.load_from_checkpoint(checkpoint_path=path,
                                       kwargs=model_kwargs)
    model.to(toru.get_device())
    return model


def wrap_model(base_model: pl.LightningModule):
    # noinspection PyAbstractClass
    class WrapperModuleChild(WrapperModule):
        def __init__(self,
                     learning_rate: float,
                     output_type: Literal["binary", "multiclass", "multilabel"] = "binary",
                     loss_fn=F.cross_entropy):
            super(WrapperModuleChild, self).__init__(
                base_model,
                learning_rate,
                output_type=output_type,
                loss_fn=loss_fn)

    return WrapperModuleChild


def set_deterministic(seed: int = 42) -> None:
    # https://pytorch.org/docs/stable/notes/randomness.html
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def train_model(model,
                train_kwargs,
                checkpoint_path,
                dataset=None,
                data_manager=None,
                epochs=300,
                deterministic=False):
    if deterministic:
        set_deterministic()

    if data_manager is not None and dataset is None:
        dataset = create_data_module(data_manager=data_manager)

    lightning_model_fn = wrap_model(model)
    lightning_model = lightning_model_fn(**train_kwargs)

    temp_checkpoint_path = os.path.join(checkpoint_path, "temp")
    fu.create_dirs(temp_checkpoint_path)
    callback_checkpoint = ModelCheckpoint(monitor='val_loss',
                                          dirpath=temp_checkpoint_path,
                                          filename="checkpoint-{val_acc:.2f}-{val_loss:.2f}-{epoch:02d}-at-{time_stamp}")
    # callback_checkpoint.FILE_EXTENSION = ".pth"
    
    trainer = pl.Trainer(
        # accelerator=pyu.get_device_type(),
        # precision="16",
        # devices=-1,
        callbacks=[
            # EarlyStopping(
            #     monitor='val_loss',
            #     patience=25
            # ),
            TQDMProgressBar(),
            # BatchSizeFinder(),
            # LearningRateFinder(),
            InputMonitor(),
            # CheckBatchGradient(),  #todo: see why broken
            callback_checkpoint
        ],
        enable_progress_bar=True,
        # auto_lr_find=True,
        # auto_scale_batch_size="power",
        max_epochs=epochs,
        logger=TensorBoardLogger(os.path.join(checkpoint_path, 'tensorboard')),

        benchmark=not deterministic,
        deterministic=deterministic)

    # torch._dynamo.config.suppress_errors = True
    # lightning_model = torch.compile(lightning_model)
    trainer.fit(lightning_model, datamodule=dataset)

    print("Best Model", callback_checkpoint.best_model_path)

    model = load_model(model=lightning_model_fn,
                       path=callback_checkpoint.best_model_path,
                       model_kwargs=train_kwargs)

    if dataset.test_dataset is not None:
        results = trainer.test(model, datamodule=dataset)
        print("Test Results", results)

    model_path = fu.move(callback_checkpoint.best_model_path, checkpoint_path)
    fu.remove_dirs(temp_checkpoint_path)

    return model, model_path


def test_model(model, x=None, y=None, trainer=None, data=None):
    if trainer is None:
        trainer = pl.Trainer()

    if data is None:
        DatasetDataModule(test_dataset=NumpyXYDataset(x, y))

    return trainer.test(model, datamodule=data, verbose=True)


def create_data_module(x_train=None, y_train=None,
                       x_validation=None, y_validation=None,
                       x_test=None, y_test=None,
                       data_manager: du.DataManager = None,
                       dataset_fn=None):
    if data_manager is not None:
        x_train = data_manager.x
        y_train = data_manager.y
        if data_manager.validation is not None and data_manager.validation.x is not None:
            x_validation = data_manager.validation.x
            y_validation = data_manager.validation.y

        if data_manager.test is not None and data_manager.test.x is not None:
            x_test = data_manager.test.x
            y_test = data_manager.test.y

    # todo: get dataset generator based on x type or convert on the fly based on type
    if dataset_fn is None:
        dataset_fn = NumpyXYDataset

    return DatasetDataModule(
        train_dataset=dataset_fn(x_train, y_train),
        test_dataset=dataset_fn(x_test, y_test) if x_test is not None else None,
        val_dataset=dataset_fn(x_validation, y_validation) if x_validation is not None else None
    )


def overfit_and_get_averaged_loss(model, data_loader, steps=100, plot=True):
    """Function to overfit the model on a small set of data and return
       loss per steps to determine if there is a bug or not"""
    # Seed for reproducibility
    set_deterministic()
    # Create a metric tracker to get back the loss for each step after training
    metric_tracker = MetricTracker()
    # Create a dataloader for training
    # Creaate a trainer to overfit 10% of the data
    trainer = pl.Trainer(overfit_batches=0.1, callbacks=metric_tracker)
    # Train the model (try to overfit 10% of the data)
    trainer.fit(model, train_dataloaders=data_loader)

    # Return the moving average over the loss with a window of 100 steps
    loss_running_mean = np.convolve(metric_tracker.collection, np.ones(steps) / steps, mode='valid')

    # if plot:
    #     pltu.plot(data=loss_running_mean,
    #               title="With a buggy model",
    #               x_label="Training steps",
    #               y_label="Averaged loss")
    return loss_running_mean


class InputMonitor(pl.Callback):
    def on_train_batch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             batch: Any,
                             batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


class CheckBatchGradient(pl.Callback):
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()

        zero_grad_indexes = list(range(example_input.size(0)))
        zero_grad_indexes.pop(n)

        if example_input.grad[zero_grad_indexes].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
