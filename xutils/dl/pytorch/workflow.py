import os
import tempfile
import zipfile
from typing import Optional

from mrbuilder.builders import pytorch as mrb
import xutils.dl.pytorch.lightning_utils as lu
import xutils.core.file_utils as fu
import xutils.data.data_utils as du
import xutils.data.json_utils as ju


def get_model_builder_from_config(model_definition, config_path):
    data_manager = du.DataManager()
    data_manager.load_config(config_path,
                             play=False)
    model_builder = get_model_builder(model_definition=model_definition,
                                      input_shape=data_manager.shape_x,
                                      output_size=data_manager.shape_y)
    return model_builder


def get_model_builder(model_definition, input_shape, output_size):
    mrb_net_builder = mrb.build(model_definition)
    model = mrb_net_builder(input_shape, output_size=output_size)
    return model


# class CheckpointPackage:
#     CHECKPOINT_EXTENSION = "ckpt"
#     EXTENSION = "package.zip"
#     META_JSON_NAME = "meta.json"
#     CHECKPOINT_NAME = "checkpoint." + CHECKPOINT_EXTENSION
# 
#     def __init__(self,
#                  model_definition=None,
#                  train_kwargs=None,
#                  input_shape=None,
#                  output_size=None,
#                  checkpoint_name=None,
#                  checkpoint_path: str = None,
#                  checkpoint_file=None) -> None:
#         super().__init__()
#         self.model_definition = model_definition
#         self.train_kwargs = train_kwargs
#         self.input_shape = input_shape
#         self.output_size = output_size
# 
#         self.checkpoint_path = checkpoint_path
#         if checkpoint_name is not None:
#             self.checkpoint_name = checkpoint_name
#         elif checkpoint_path is not None:
#             self.checkpoint_name = fu.get_file_name(self.checkpoint_path)
# 
#         self.checkpoint_file = checkpoint_file
#         self.checkpoint_dir: Optional[tempfile.TemporaryDirectory] = None
# 
#     def to_json(self):
#         return {
#             "model_definition": self.model_definition,
#             "train_kwargs": self.train_kwargs,
#             "input_shape": self.input_shape,
#             "output_size": self.output_size,
#             "checkpoint_name": fu.get_file_name(self.checkpoint_path)
#         }
# 
#     def save(self):
#         package_path = f"{self.checkpoint_path}.{self.EXTENSION}"
# 
#         with zipfile.ZipFile(package_path, mode="w") as archive:
#             archive.writestr(self.META_JSON_NAME, ju.write(self.to_json()))
#             archive.write(self.checkpoint_path, arcname=self.CHECKPOINT_NAME)
# 
#         return package_path
# 
#     def __enter__(self):
#         self.read()
# 
#     def read(self):
#         if self.checkpoint_path.endswith(CheckpointPackage.EXTENSION):
#             with zipfile.ZipFile(self.checkpoint_path, mode="r") as archive:
#                 data = ju.read(archive.read(CheckpointPackage.META_JSON_NAME).decode(encoding="utf-8"))
#                 for name, value in data.items():
#                     setattr(self, name, value)
# 
#                 self.checkpoint_dir = tempfile.TemporaryDirectory()
#                 self.checkpoint_file = archive.extract(CheckpointPackage.CHECKPOINT_NAME,
#                                                        path=os.path.join(self.checkpoint_dir.name,
#                                                                          self.CHECKPOINT_NAME))
#         else:
#             raise ValueError("Not a valid package file")
# 
#     def __exit__(self, *args):
#         self.close()
# 
#     def close(self):
#         self.checkpoint_dir.cleanup()
# 
#     @staticmethod
#     def is_package(path):
#         return path.endswith(CheckpointPackage.EXTENSION) and zipfile.is_zipfile(path)


def train_model(model_definition,
                train_kwargs,
                checkpoint_path,
                data_manager,
                epochs=300,
                deterministic=False,
                delete_checkpoint=False):
    model_builder = get_model_builder(model_definition=model_definition,
                                      input_shape=data_manager.shape_x,
                                      output_size=data_manager.shape_y)

    best_model, best_model_path = lu.train_model(model=model_builder,
                                                 data_manager=data_manager,
                                                 train_kwargs=train_kwargs,
                                                 epochs=epochs,
                                                 checkpoint_path=checkpoint_path,
                                                 deterministic=deterministic)

    # package_path = convert_checkpoint(checkpoint_path=best_model_path,
    #                                   input_shape=data_manager.shape_x,
    #                                   output_size=data_manager.shape_y,
    #                                   model_definition=model_definition,
    #                                   train_kwargs=train_kwargs)
    package_path = save_checkpoint_meta(checkpoint_path=best_model_path,
                                        input_shape=data_manager.shape_x,
                                        output_size=data_manager.shape_y,
                                        model_definition=model_definition,
                                        train_kwargs=train_kwargs)
    if delete_checkpoint:
        fu.delete(checkpoint_path)

    return package_path


META_SUFFIX = ".meta.json"


def save_checkpoint_meta(checkpoint_path, input_shape, output_size, model_definition, train_kwargs):
    return ju.write_to({
        "model_definition": model_definition,
        "train_kwargs": train_kwargs,
        "input_shape": input_shape,
        "output_size": output_size,
        "checkpoint_name": fu.get_file_name(checkpoint_path)
    }, path=checkpoint_path + META_SUFFIX, pretty_print=True)


# def convert_checkpoint(checkpoint_path, input_shape, output_size, model_definition, train_kwargs):
#     package = CheckpointPackage(model_definition=model_definition,
#                                 train_kwargs=train_kwargs,
#                                 input_shape=input_shape,
#                                 output_size=output_size,
#                                 checkpoint_path=checkpoint_path)
#     path = package.save()
#     return path


def load_checkpoint(checkpoint,
                    model_definition=None,
                    input_shape=None,
                    output_size=None,
                    train_kwargs=None):

    if model_definition is None:
        meta_file = ju.read_file(checkpoint + META_SUFFIX)

        model_definition = meta_file.get("model_definition")
        input_shape = meta_file.get("input_shape")
        output_size = meta_file.get("output_size")
        train_kwargs = meta_file.get("train_kwargs")

    model_builder = get_model_builder(model_definition=model_definition,
                                      input_shape=input_shape,
                                      output_size=output_size)

    return lu.load_model(model_or_class=model_builder,
                         path=checkpoint,
                         model_kwargs=train_kwargs,
                         wrap=True)

    # if CheckpointPackage.is_package(checkpoint):
    #     package = CheckpointPackage(checkpoint_path=checkpoint)
    #     with package:
    #         return load_checkpoint(model_definition=package.model_definition,
    #                                input_shape=package.input_shape,
    #                                output_size=package.output_size,
    #                                train_kwargs=package.train_kwargs,
    #                                checkpoint=package.checkpoint_file)
    # else:
    #     model_builder = get_model_builder(model_definition=model_definition,
    #                                       input_shape=input_shape,
    #                                       output_size=output_size)
    # 
    #     return lu.load_model(model_or_class=model_builder,
    #                          path=checkpoint,
    #                          model_kwargs=train_kwargs,
    #                          wrap=True)
