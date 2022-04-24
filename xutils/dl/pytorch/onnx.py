"""
Exports a pytorch model to an ONNX format, and then converts from the
ONNX to a Tensorflow serving protobuf file.

Running example:
python3 pytorch_to_tf_serving.py \
 --onnx-file text.onnx \
 --meta-file text.meta \
 --export-dir serving_model/
"""

import onnx
import torch


def export_onnx(model, dummy_input, file, input_names, output_names,
                num_inputs):
    """
    Converts a Pytorch model to the ONNX format and saves the .onnx model file.
    The first dimension of the input nodes are of size N, where N is the
    minibatch size. This dimensions is here replaced by an arbitrary string
    which the ONNX -> TF library interprets as the '?' dimension in Tensorflow.
    This process is applied because the input minibatch size should be of an
    arbitrary size.

    Example:
    model = torchvision.models.alexnet(pretrained=True)
    img_input = torch.randn(1, 3, 224, 224)

    input_names = ['input_img']
    output_names = ['confidences']

    # Use a tuple if there are multiple model inputs
    dummy_inputs = (img_input)

    export_onnx(model, dummy_inputs, args.onnx_file,
                input_names=input_names,
                output_names=output_names,
                num_inputs=len(dummy_inputs))


    :param model: Pytorch model instance with loaded weights
    :param dummy_input: tuple, dummy input numpy arrays that the model
        accepts in the inference time. E.g. for the Text+Image model, the
        tuple would be (np.float32 array of N x W x H x 3, np.int64 array of
        N x VocabDim). Actual numpy arrays values don't matter, only the shape
        and the type must match the model input shape and type. N represents
        the minibatch size and can be any positive integer. True batch size
        is later handled when exporting the model from the ONNX to TF format.
    :param file: string, Path to the exported .onnx model file
    :param input_names: list of strings, Names assigned to the input nodes
    :param output_names: list of strings, Names assigned to the output nodes
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    """
    # List of onnx.export function arguments:
    # https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
    # ISSUE: https://github.com/pytorch/pytorch/issues/14698
    torch.onnx.export(model, args=dummy_input, input_names=input_names,
                      output_names=output_names, f=file)

    # Reload model to fix the batch size
    model = onnx.load(file)
    model = make_variable_batch_size(num_inputs, model)
    onnx.save(model, file)


def make_variable_batch_size(num_inputs, onnx_model):
    """
    Changes the input batch dimension to a string, which makes it variable.
    Tensorflow interpretes this as the "?" shape.
    `num_inputs` must be specified because `onnx_model.graph.input` is a list
    of inputs of all layers and not just model inputs.

    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    :param onnx_model: ONNX model instance
    :return: ONNX model instance with variable input batch size
    """
    for i in range(num_inputs):
        onnx_model.graph.input[i].type.tensor_type. \
            shape.dim[0].dim_param = 'batch_size'
    return onnx_model
