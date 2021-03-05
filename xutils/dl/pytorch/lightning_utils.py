from typing import Any

from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F


class WrapperModule(LightningModule):
    def __init__(self, wrapped, learning_rate):
        super(WrapperModule, self).__init__()
        self.save_hyperparameters()
        self.model = wrapped
        self.model.to(self.device)

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics


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

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


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
