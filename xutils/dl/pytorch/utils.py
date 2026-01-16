import io
from typing import List, Optional, Callable, Sequence

import requests

import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.profiler import profile, record_function, ProfilerActivity

import xutils.core.python_utils as pyu
import xutils.data.data_utils as du


def has_gpu():
    return torch.cuda.is_available()


def has_mps():
    return torch.backends.mps.is_available()


def get_device():
    if has_gpu():
        return "cuda"
    elif has_mps():
        return "mps"
    return "cpu"


def get_device_type():
    if has_gpu():
        return "gpu"
    elif has_mps():
        return "mps"
    return "cpu"


def memory_summary(device: str = None):
    if torch.accelerator.is_available() and hasattr(torch.accelerator, "memory_summary"):
        print(torch.accelerator.memory_summary(abbreviated=False))
    elif hasattr(torch, device):
        backend = getattr(torch, device)
        if hasattr(backend, "memory_summary"):
            print(backend.memory_summary(abbreviated=False))
        elif device == "mps":
            # Extract individual stats for Apple Silicon
            allocated = backend.current_allocated_memory()  # Bytes used by tensors
            driver_total = backend.driver_allocated_memory() # Total Metal driver allocation
            recommended_max = backend.recommended_max_memory() # System-suggested limit

            print(
                f"--- MPS Memory Report (2026) ---\n",
                f"Active Tensors:    {allocated / 1024**2:.2f} MB\n",
                f"Total Driver Alloc: {driver_total / 1024**2:.2f} MB (includes cache)\n",
                f"Recommended Max:    {recommended_max / 1024**2:.2f} MB\n",
                f"Current Cache Size: {(driver_total - allocated) / 1024**2:.2f} MB"
            )
        else:
            print(f"Backend {device} does not support memory_summary.")


def num_gpus():
    return torch.cuda.device_count()


def print_device():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def test_devices():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
        x = torch.ones(1, device=device)
        print(x)
    if torch.cuda.is_available():
        device = torch.device("mps")
        print("Cuda device found.")
        x = torch.ones(1, device=device)
        print(x)

    cpu_device = torch.device("cpu")
    x = torch.ones(1, device=cpu_device)
    print("CPU device test successful.")
    print(x)


def free_memory():
    torch.cuda.empty_cache()
    pyu.free_memory()


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


def profile_execution(description: str, fun: Callable, *args, **kwargs):
    if torch.backends.mps.is_available():
        torch.mps.profiler.start(mode='interval,event')
        fun(*args, **kwargs)
        torch.mps.profiler.stop()
    else:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function(description):
                fun(*args, **kwargs)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def choose_int_type(x: int, large=torch.int64, small=torch.int32):
    return small if x < 2 ** 31 else large


def choose_cell_ref_type(x: int):
    return choose_int_type(x, large=torch.long)


def create_random_index_tensor(shape: Sequence[int], device: str = Optional[str]):
    """
    Creates a matrix of shape where each element is a
    random index in the range [0, y * x - 1], not equal to its own linear index.

    Args:
        shape: The dimensions of the output tensor.
        device: The device to place the tensor on.
    """
    n = shape[:-2]
    y, x = shape[-2:]
    num_elements_yx = y * x
    if num_elements_yx <= 1:
        raise ValueError("Cannot satisfy the condition with only one or fewer elements.")

    # 1. Generate random indices in the range [0, num_elements_yx - 2]
    # The range is one less than the total number of elements.
    random_indices_base = torch.randint(
        0,
        num_elements_yx - 1,
        (*n, y, x),
        dtype=choose_int_type(num_elements_yx, large=torch.float),
        device=device)

    # 2. Calculate the "forbidden" indices (the element's own linear index)
    # Create a 2D tensor of linear indices (0 to y*x-1)
    forbidden_indices_2d = torch.arange(
        num_elements_yx,
        device=device
    ).reshape(y, x).long()
    # Expand this to match the (n, y, x) shape
    forbidden_indices = forbidden_indices_2d.expand(*n, -1, -1)

    # 3. Adjust the generated indices if they are >= the forbidden index.
    # This maps the range [0, ..., k-1, k, ..., N-2] to [0, ..., k-1, k+1, ..., N-1].
    adjustment = (random_indices_base >= forbidden_indices).long()

    random_indices_base += adjustment

    return random_indices_base


def set_from_indices(populate_matrix: torch.Tensor,
                     indexes: torch.Tensor,
                     value: bool = True,
                     reset: bool = True):
    """
    Populates a matrix with values based on linear indices.

    Args:
        populate_matrix (torch.Tensor): A 2D tensor of shape (h, w) to populate.
        indexes (torch.Tensor): A 1D with linear indices.
        value: The value to set at the specified indices. Defaults to True.
        reset: set all the other values to 0. Defaults to True.

    Returns:
        None
    """
    populate_matrix = populate_matrix.view(-1)
    if reset:
        populate_matrix = populate_matrix.zero_()

    populate_matrix.scatter_(dim=0, index=indexes, value=value)


def count_unique_values(tensor, min_count: int = None):
    """
    Counts unique values in a 1D tensor and returns a 2D tensor
    of shape [unique_values, counts].

    Args:
        tensor (torch.Tensor): A 1D input tensor.
        min_count (int): Min Filters counts

    Returns:
        torch.Tensor: A 2D tensor where rows are [value, count].
    """

    # torch.unique returns a tuple: (unique_values_tensor, counts_tensor)
    unique_values, counts = torch.unique(tensor, return_counts=True)

    # Stack the two 1D tensors to create a single 2D tensor of shape [N, 2]
    # We use torch.stack(..., dim=1) to combine them side-by-side (columns)
    stacked = torch.stack((unique_values, counts), dim=1)

    if min_count is not None:
        return stacked[stacked[:, 1] >= min_count]
    else:
        return stacked


def to_linear_index(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    locations = tensor.nonzero()
    if len(locations) == 0:
        return None

    return locations[:, 0] * tensor.shape[1] + locations[:, 1]
