from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

from xutils.core.python_utils import getattr_ignore_case


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
            self.loss_fn = lambda y_hat, y: (-(y_hat+1e-5).log() * y).sum(dim=1).mean()
        elif isinstance(loss_fn, str):
            self.loss_fn = getattr_ignore_case(F, loss_fn)
        else:
            self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def calculate_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)


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
