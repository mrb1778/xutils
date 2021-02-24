import pytorch_lightning as pl


class WrapperModule(pl.LightningModule):
    def __init__(self, wrapped):
        super(WrapperModule, self).__init__()
        self.model = wrapped
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)
