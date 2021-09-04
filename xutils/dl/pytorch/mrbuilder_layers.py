import torch

from mrbuilder.builders.pytorch.builder_models import PyTorchBuilderLayer
from mrbuilder.builders.pytorch.layer_registry import register_layer


@register_layer("BiChannelDifference")
class BiChannelDifferenceBuilderLayer(PyTorchBuilderLayer):
    def forward(self, x):
        right = torch.roll(x, 1, 0)
        left = torch.roll(x, -1, 0)
        top = torch.roll(x, -1, 1)
        bottom = torch.roll(x, -1, 1)

        return torch.cat((
                torch.subtract(x, right),
                torch.subtract(x, left),
                torch.subtract(x, top),
                torch.subtract(x, bottom)),
            dim=1
        )

    def get_output_size(self):
        return [4, *self.previous_size[1:]]
