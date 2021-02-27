import io
import torch
from torch import nn as nn


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def model_summary(model, count_params=True) -> str:
    print_buffer = io.StringIO()
    print(model, file=print_buffer)

    value = print_buffer.getvalue()
    return F'{value}\nParameters: {count_parameters(model)}' if count_params else value


def print_model_summary(model, count_params=True):
    print(model_summary(model, count_params))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransferLearnModel(nn.Module):
    def __init__(self, original_model, num_classes, activation_fn=None, freeze_weights=True):
        super(TransferLearnModel, self).__init__()

        original_layers = list(original_model.children())
        original_body_layers = original_layers[:-1]
        original_classification_layer = original_layers[-1]

        if not isinstance(original_classification_layer, nn.Linear):
            raise Exception("Last Layer Must be Linear")  # todo: account for last layer being a activation

        self.features = nn.Sequential(*original_body_layers)

        linear_layer = nn.Linear(original_classification_layer.out_features, num_classes)

        self.classifier = nn.Sequential(linear_layer, activation_fn) if activation_fn is not None else linear_layer

        if freeze_weights:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y