from datetime import datetime

from mrbuilder.builders import pytorch as mrb
import xutils.dl.pytorch.lightning_utils as lu
import xutils.core.file_utils as fu
import xutils.data.json_utils as ju
import xutils.data.data_utils as du


def get_model_builder(model_definition, input_shape, **model_kwargs):
    mrb_net_builder = mrb.build(model_definition)
    model = mrb_net_builder(input_shape, **model_kwargs)
    return model


def train_model(model_definition,
                model_kwargs,
                train_kwargs,
                checkpoint_path,
                data_manager,
                epochs=300,
                deterministic=False,
                test_fn=None):
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["output_size"] = data_manager.shape_y

    model_builder = get_model_builder(model_definition=model_definition,
                                      input_shape=data_manager.shape_x,
                                      **model_kwargs)

    best_model, best_model_path = lu.train_model(model=model_builder,
                                                 data_manager=data_manager,
                                                 train_kwargs=train_kwargs,
                                                 epochs=epochs,
                                                 checkpoint_path=checkpoint_path,
                                                 deterministic=deterministic)

    checkpoint_meta = CheckpointMeta(best_model_path)
    checkpoint_meta.save_meta(model_definition=model_definition,
                              data_manager=data_manager,
                              input_shape=data_manager.shape_x,
                              model_kwargs=model_kwargs,
                              train_kwargs=train_kwargs)

    if test_fn is not None:
        checkpoint_meta.evaluate(test_fn)

    return best_model, best_model_path


META_SUFFIX = ".meta.json"
RESULTS_SUFFIX = ".results.json"


class CheckpointMeta:
    def __init__(self, checkpoint) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.metadata = None
        self.performance = None

    def save_meta(self,
                  model_definition,
                  data_manager,
                  input_shape,
                  model_kwargs,
                  train_kwargs):
        return ju.write_to({
            "data_manager": data_manager,
            "model_definition": model_definition,
            "input_shape": input_shape,
            "model_kwargs": model_kwargs,
            "train_kwargs": train_kwargs,
            "checkpoint_name": fu.get_file_name(self.checkpoint),
            "timestamp": str(datetime.now())
        }, path=self.checkpoint + META_SUFFIX, pretty_print=True)

    def load_model(self):
        self.metadata = ju.read_file(self.checkpoint + META_SUFFIX)

        model_definition = self.metadata.get("model_definition")
        input_shape = self.metadata.get("input_shape")

        model_kwargs = {}
        if "model_kwargs" in self.metadata:
            model_kwargs.update(self.metadata.get("model_kwargs"))
        if "output_size" in self.metadata:
            model_kwargs["output_size"] = self.metadata.get("output_size")
        train_kwargs = self.metadata.get("train_kwargs")

        model_builder = get_model_builder(model_definition=model_definition,
                                          input_shape=input_shape,
                                          **model_kwargs)

        return lu.load_model(model_or_class=model_builder,
                             path=self.checkpoint,
                             model_kwargs=train_kwargs,
                             wrap=True)

    def evaluate(self, test_fn=None, test_results=None):
        self.performance = test_fn(self.checkpoint) if test_fn is not None else test_results
        if test_results is not None:
            test_results["timestamp"] = str(datetime.now())
        ju.write_to(self.performance,
                    path=self.checkpoint + RESULTS_SUFFIX, pretty_print=True)

        return self.performance

    def get_performance(self,
                        criteria=None):
        if self.performance is None:
            self.performance = ju.read_file(self.checkpoint + RESULTS_SUFFIX)
        return self.performance if criteria is None else self.performance[criteria]

    def create_datamanager(self, df):
        data_manager = du.DataManager()

        data_manager.set_config(self.metadata["data_manager"],
                                play=False)

        data_manager.df = df
        data_manager.replay_config()

        return data_manager

    def run(self, df):
        model = self.load_model()
        data_manager = self.create_datamanager(df)
        results = lu.run_model(model, data_manager)
        return data_manager.decode_labels(results)

