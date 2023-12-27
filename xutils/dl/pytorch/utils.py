import io
from typing import List, Optional

import requests

import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

import xutils.core.python_utils as pyu
import xutils.data.data_utils as du

def has_gpu():
    return torch.cuda.is_available()


def get_device():
    return "cuda:0" if has_gpu() else "cpu"


def get_device_type():
    return "gpu" if has_gpu() else "cpu"


def num_gpus():
    return torch.cuda.device_count()


def free_memory():
    torch.cuda.empty_cache()


def model_summary(model, count_params=True) -> str:
    print_buffer = io.StringIO()
    print(model, file=print_buffer)

    value = print_buffer.getvalue()
    return F'{value}\nParameters: {count_parameters(model)}' if count_params else value


def print_model_summary(model, count_params=True):
    print(model_summary(model, count_params))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    return torch.tensor(accuracy)


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


def save_for_mobile(model, spec, save_as):
    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model)
    extra_files = {
        "model/live.spec.json": spec
    }
    # noinspection PyProtectedMember
    optimized_model._save_for_lite_interpreter(f"{save_as}.ptl", _extra_files=extra_files)

    # https://pytorch.org/mobile/android/
    # import torch
    # import torchvision
    # from torch.utils.mobile_optimizer import optimize_for_mobile
    #
    # model = torchvision.models.mobilenet_v2(pretrained=True)
    # model.eval()
    # example = torch.rand(1, 3, 224, 224)
    # traced_script_module = torch.jit.trace(model, example)
    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("app/src/main/assets/model.ptl")


def load_model(path: str, device: torch.device = None) -> nn.Module:
    if device is None:
        device = get_device()

    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)


def parse_loss_fn(loss_fn):
    if loss_fn is None:
        # todo: auto choose based on type flag
        return F.cross_entropy
    elif loss_fn == "categorical_cross_entropy":
        # https://sparrow.dev/cross-entropy-loss-in-pytorch/
        return lambda y_hat, y: (-(y_hat + 1e-5).log() * y).sum(dim=1).mean()
    elif isinstance(loss_fn, str):
        return pyu.getattr_ignore_case(F, loss_fn) or pyu.getattr_ignore_case(nn, loss_fn)()
    else:
        return loss_fn


class ToTensorConverter:
    def can_convert(self, x) -> bool:
        pass

    def convert(self, x) -> Optional[torch.Tensor]:
        pass


to_tensor_converters: List[ToTensorConverter] = []


def to_tensor_converter(converter):
    instance = converter()
    to_tensor_converters.append(instance)
    return instance


@to_tensor_converter
class NumpyDataManagerToTensorConverter(ToTensorConverter):
    def can_convert(self, x) -> bool:
        return isinstance(x, du.DataManager) and x.x is not None and isinstance(x.x, np.ndarray)

    def convert(self, x: du.DataManager) -> torch.Tensor:
        return torch.from_numpy(x.x).float()


class NumpyArrayToTensorConverter(ToTensorConverter):
    def can_convert(self, x) -> bool:
        return isinstance(x, np.ndarray)

    def convert(self, x) -> torch.Tensor:
        return torch.from_numpy(x).float()


def convert_to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        for converter in reversed(to_tensor_converters):
            if converter.can_convert(x):
                return converter.convert(x)
    raise RuntimeError("No converter found for ", x)


def run_model(model, x=None, add_batch_dimension=False) -> torch.Tensor:
    x = convert_to_tensor(x)
    x = x.to(get_device())

    if add_batch_dimension:
        x = x.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        return model(x).cpu()
