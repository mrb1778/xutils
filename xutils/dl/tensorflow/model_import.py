"""
Exports a pytorch model to an ONNX format, and then converts from the
ONNX to a Tensorflow serving protobuf file.

Running example:
python3 pytorch_to_tf_serving.py \
 --onnx-file text.onnx \
 --meta-file text.meta \
 --export-dir serving_model/
"""

import tensorflow as tf
from tensorflow.python.saved_model import utils as smutils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from onnx_tf.backend import prepare
import onnx


def export_tf_proto(onnx_file, meta_file):
    """
    Exports the ONNX model to a Tensorflow Proto file.
    The exported file will have a .meta extension.

    :param onnx_file: string, Path to the .onnx model file
    :param meta_file: string, Path to the exported Tensorflow .meta file
    :return: tuple, input and output tensor dictionaries. Dictionaries have a
        {tensor_name: TF_Tensor_op} structure.
    """
    model = onnx.load(onnx_file)

    # Convert the ONNX model to a Tensorflow graph
    tf_rep = prepare(model)
    output_keys = tf_rep.outputs
    input_keys = tf_rep.inputs

    tf_dict = tf_rep.tensor_dict
    input_tensor_names = {key: tf_dict[key] for key in input_keys}
    output_tensor_names = {key: tf_dict[key] for key in output_keys}

    tf_rep.export_graph(meta_file)
    return input_tensor_names, output_tensor_names


def export_for_serving(meta_path, export_dir, input_tensors, output_tensors):
    """
    Exports the Tensorflow .meta model to a frozen .pb Tensorflow serving
       format.

    :param meta_path: string, Path to the .meta TF proto file.
    :param export_dir: string, Path to directory where the serving model will
        be exported.
    :param input_tensor: dict, Input tensors dictionary of
        {name: TF placeholder} structure.
    :param output_tensors: dict, Output tensors dictionary of {name: TF tensor}
        structure.
    """
    g = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_def = tf.GraphDef()

    with g.as_default():
        with open(meta_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        # name argument must explicitly be set to an empty string, otherwise
        # TF will prepend an `import` scope name on all operations
        tf.import_graph_def(graph_def, name="")

        tensor_info_inputs = {name: smutils.build_tensor_info(in_tensor)
                              for name, in_tensor in input_tensors.items()}

        tensor_info_outputs = {name: smutils.build_tensor_info(out_tensor)
                               for name, out_tensor in output_tensors.items()}

        prediction_signature = signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={"predict_images": prediction_signature})
        builder.save()
