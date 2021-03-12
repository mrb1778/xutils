from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

from xutils.core.python_utils import getattr_ignore_case
import xutils.dl.sklearn.train_utils as sku


class WrapperModule(LightningModule):
    def __init__(self, wrapped, learning_rate, loss_fn=None):
        super(WrapperModule, self).__init__()
        self.model = wrapped
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate', 'loss_fn')

        if loss_fn is None:
            # todo: auto choose based on type flag
            self.loss_fn = F.cross_entropy
        elif loss_fn == "categorical_cross_entropy":
            # loss_tracker = nn.NLLLoss()
            # self.loss_fn = lambda y_hat, y: loss_tracker(torch.log(y_hat), y)
            self.loss_fn = lambda y_hat, y: (-(y_hat + 1e-5).log() * y).sum(dim=1).mean()
        elif isinstance(loss_fn, str):
            self.loss_fn = getattr_ignore_case(F, loss_fn)
        else:
            self.loss_fn = loss_fn

        self.train_loss_tracker = EMATracker(alpha=0.02)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)

        return {'loss': loss, "log": batch_idx % self.trainer.accumulate_grad_batches == 0}

    def _should_log(self, flag):
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if isinstance(flag, list):
                return flag[0]
            return flag
        return False

    def training_step_end(self, outputs):
        # Aggregate the losses from all GPUs
        loss = outputs["loss"].mean()
        self.train_loss_tracker.update(loss.detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value
            }, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return {'y_hat': y_hat.numpy(), 'y': y.numpy(), **metrics}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([tmp['y_hat'] for tmp in outputs])
        y = torch.cat([tmp['y'] for tmp in outputs])
        sku.compare_results(y_hat, y)
        # confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=10)
        #
        # df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(10), columns=range(10))
        # plt.figure(figsize=(10, 7))
        # fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        #
        # self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def calculate_loss(self, y_hat, y):
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
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()


class DatasetDataModule(LightningDataModule):
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
