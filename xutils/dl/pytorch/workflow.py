from datetime import datetime
from typing import Dict, Any, Optional, Callable, Tuple

from mrbuilder.builders import pytorch as mrb
import xutils.dl.pytorch.utils as toru
import xutils.dl.pytorch.lightning_utils as lu
import xutils.core.file_utils as fu
import xutils.data.json_utils as ju
import xutils.data.data_utils as du


def get_model(model_definition: Dict[str, Any],
              input_shape: Tuple[int, ...],
              **model_kwargs: Any) -> Any:
    mrb_net_builder = mrb.build(model_definition)
    model = mrb_net_builder(input_shape, **model_kwargs)
    return model


def train_model(model_definition: Dict,
                model_kwargs: Dict,
                train_kwargs: Dict,
                checkpoint_path: str,
                data_manager: du.DataManager,
                epochs: int = 300,
                deterministic: bool = False,
                test_fn: Optional[Callable[[Any], Dict[str, Any]]] = None):
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["output_size"] = data_manager.shape_y

    model_builder = get_model(model_definition=model_definition,
                              input_shape=data_manager.shape_x,
                              **model_kwargs)

    model, model_path = lu.train_model(model=model_builder,
                                       data_manager=data_manager,
                                       train_kwargs=train_kwargs,
                                       epochs=epochs,
                                       checkpoint_path=checkpoint_path,
                                       deterministic=deterministic)

    checkpoint_meta = CheckpointMeta(model_path)
    checkpoint_meta.save_meta(model_definition=model_definition,
                              data_manager=data_manager,
                              input_shape=data_manager.shape_x,
                              model_kwargs=model_kwargs,
                              train_kwargs=train_kwargs)

    if test_fn is not None:
        checkpoint_meta.evaluate(test_fn)

    return model, model_path


META_SUFFIX = ".meta.json"
RESULTS_SUFFIX = ".results.json"


class CheckpointMeta:
    def __init__(self, checkpoint_path) -> None:
        super().__init__()
        self.checkpoint_path: str = checkpoint_path
        self.metadata: Optional[Dict[str, Any]] = None
        self.performance: Optional[Dict[str, Any]] = None

    def save_meta(self,
                  model_definition: Dict,
                  data_manager: du.DataManager,
                  input_shape,
                  model_kwargs: Dict,
                  train_kwargs: Dict):
        return ju.write_to({
            "data_manager": data_manager,
            "model_definition": model_definition,
            "input_shape": input_shape,
            "model_kwargs": model_kwargs,
            "train_kwargs": train_kwargs,
            "checkpoint_name": fu.file_name(self.checkpoint_path),
            "timestamp": str(datetime.now())
        }, path=self.checkpoint_path + META_SUFFIX, pretty_print=True)

    def load_model(self):
        self.metadata = ju.read_file(self.checkpoint_path + META_SUFFIX)

        model_definition = self.metadata.get("model_definition")
        input_shape = self.metadata.get("input_shape")

        model_kwargs = {}
        if "model_kwargs" in self.metadata:
            model_kwargs.update(self.metadata.get("model_kwargs"))
        if "output_size" in self.metadata:
            model_kwargs["output_size"] = self.metadata.get("output_size")
        train_kwargs = self.metadata.get("train_kwargs")

        model_builder = get_model(model_definition=model_definition,
                                  input_shape=input_shape,
                                  **model_kwargs)

        return lu.load_model(model=model_builder,
                             path=self.checkpoint_path,
                             model_kwargs=train_kwargs,
                             wrap=True)

    def evaluate(self, test_fn: Callable[[Any], Dict[str, Any]], save_results: bool = True) -> Dict[str, Any]:
        self.performance = test_fn(checkpoint=self.checkpoint_path)
        if save_results:
            self.save_results(test_results=self.performance)
        return self.performance

    def save_results(self, name_suffix: str = "", test_results: Dict[str, Any] = None):
        if test_results is not None:
            test_results["timestamp"] = str(datetime.now())
        ju.write_to(test_results,
                    path=self.checkpoint_path + name_suffix + RESULTS_SUFFIX, pretty_print=True)

        return test_results

    def get_performance(self,
                        criteria: str = None):
        if self.performance is None:
            self.performance = ju.read_file(self.checkpoint_path + RESULTS_SUFFIX)
        return self.performance if criteria is None else self.performance[criteria]

    def create_datamanager(self,
                           df):
        data_manager = du.DataManager()
        data_manager.set_config(self.metadata["data_manager"],
                                play=False)
        data_manager.df = df
        data_manager.replay_config()

        return data_manager

    def run(self, df,
            confidence: bool = False):
        model = self.load_model()
        data_manager = self.create_datamanager(df)
        results = toru.run_model(model, data_manager)
        results = results.numpy()
        decoded_labels = data_manager.decode_labels(results)
        if confidence:
            return decoded_labels, data_manager.label_confidence(results)
        else:
            return decoded_labels
