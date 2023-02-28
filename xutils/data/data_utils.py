from typing import Dict, Any, Union

import numpy as np
import sklearn.utils as sku
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import pickle
from operator import itemgetter
import os
import random as rn
import pandas as pd

import xutils.core.python_utils as pyu
import xutils.core.file_utils as fu
import xutils.data.numpy_utils as nu
import xutils.data.json_utils as ju


def set_random_seed(seed=1235):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)


def get_balanced_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = sku.compute_class_weight('balanced', np.unique(y), y)
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
        # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

    return sample_weights


def compare_results(actual: Union[pd.DataFrame, pd.Series, np.ndarray],
                    predicted: Union[pd.DataFrame, pd.Series, np.ndarray],
                    actual_hot_encoded: bool = False,
                    predicted_hot_encoded: bool = False,
                    print_results: bool = True) -> Dict[str, Any]:
    if actual_hot_encoded:
        actual = np.argmax(actual, axis=1)
    if predicted_hot_encoded:
        predicted = np.argmax(predicted, axis=1)

    if isinstance(actual, pd.DataFrame) or isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(predicted, pd.DataFrame) or isinstance(predicted, pd.Series):
        predicted = predicted.values
    if actual.ndim > 1:
        actual = actual.ravel()
    if predicted.ndim > 1:
        predicted = predicted.ravel()

    e = np.equal(actual, predicted)
    # todo: holds [2] is causing issues with data with output size of 2, prob meant for 3 --> take num / classes or calc
    # holds = np.unique(predicted, return_counts=True)[1][-1]
    # delta = (holds / len(predicted) * 100)

    results = {
        "Size": len(actual),
        # "Actual Shape": actual.shape,
        # "Predicted Shape": predicted.shape,
        "Examples": {
            "First 5": {
                "Actual": actual[0:5],
                "Predicted": predicted[0:5]
            },
            "Last 5": {
                "Actual": actual[-5:],
                "Predicted": predicted[-5:]
            }
        },
        "Base": nu.count_unique(actual),
        "Match": nu.count_unique(predicted[e]),
        "Test": nu.count_unique(predicted),
        "Accuracy": skm.accuracy_score(actual, predicted),
        "Delta": "N/A",
        "F1 score (weighted)": skm.f1_score(actual, predicted, labels=None,
                                            average='weighted', sample_weight=None),
        "F1 score (macro)": skm.f1_score(actual, predicted, labels=None,
                                         average='macro', sample_weight=None),
        "F1 score (micro)": skm.f1_score(actual, predicted, labels=None,
                                         average='micro',
                                         sample_weight=None),  # weighted and micro preferred in case of imbalance
        "Cohen's Kappa": skm.cohen_kappa_score(actual, predicted),
    }

    # conf_mat = skm.confusion_matrix(actual, predicted)
    # results["Confusion Matrix"] = conf_mat
    # recalls = []
    # for i, row in enumerate(conf_mat):
    #     recall = np.round(row[i] / np.sum(row), 2)
    #     results[f"Recall {i}"] = recall
    #     recalls.append(recall)
    #
    # results["Recall Average"] = sum(recalls) / len(recalls)

    if print_results:
        print("----Compare Results----")
        pyu.print_dict(results)

    return results


# todo: bring back? figure out scale / split
# def split_data(x, y, train_split=0.8, scale=False):
#     x_train, x_test, y_train, y_test = train_test_split(
#         x,
#         y,
#         train_size=train_split,
#         test_size=None,
#         random_state=2,
#         shuffle=True,
#         stratify=y)
#
#     x_train, x_validation, y_train, y_validation = train_test_split(
#         x_train,
#         y_train,
#         train_size=train_split,
#         test_size=None,
#         random_state=2,
#         shuffle=True,
#         stratify=y_train)
#
#     data = {
#         "train": {"x": x_train, "y": y_train},
#         "test": {"x": x_test, "y": y_test},
#         "validation": {"x": x_validation, "y": y_validation}
#     }
#
#     if scale:
#         scale_split_data(data)
#
#     return data


def scale_split_data(data):
    mm_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
    data.train.x = mm_scaler.fit_transform(data.train.x)
    data.validation.x = mm_scaler.transform(data.validation.x)
    data.test.x = mm_scaler.transform(data.test.x)


def normalize_min_max(values):
    return preprocessing.MinMaxScaler().fit_transform(values)


def find_top_features(x, y, data_columns, num_top_features):
    x_copy = x.copy()

    select_k_best = SelectKBest(f_classif, k=num_top_features)
    select_k_best.fit(x_copy, y)
    selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(data_columns)

    select_k_best = SelectKBest(mutual_info_classif, k=num_top_features)
    select_k_best.fit(x_copy, y)
    selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(data_columns)

    common = list(set(selected_features_anova).intersection(selected_features_mic))

    feat_idx = []
    for c in common:
        feat_idx.append(data_columns.index(c))
    feat_idx = sorted(feat_idx[0:225])

    return feat_idx


def select_top_features(x, y, data_columns, num_top_features):
    return x[:, find_top_features(x, y, data_columns, num_top_features)]


class DataManager:
    def __init__(self,
                 x=None, y=None,
                 test=None, validation=None,
                 labels=None,
                 label_encoder=None,
                 data_columns=None,
                 data_loader=None,
                 data_enricher=None,
                 df=None) -> None:
        super().__init__()
        self.output_type = None
        self.x = x
        self.y = y
        self.data_columns = data_columns

        self.test = test
        self.validation = validation
        self.labels = labels

        self.top_features = None
        self.label_encoder = label_encoder
        self.data_loader = data_loader
        self.data_enricher = data_enricher

        self.df = df
        self.shape_x = None
        self.shape_y = None

        self.property_config = []
        self.data_config = []
        self.loaded_data_config = []

    def set_output_type(self, output_type):
        self.output_type = output_type
        self.property_config.append({"type": "set_output_type", "kwargs": {"output_type": self.output_type}})

    def load_data(self, *args, **kwargs):
        if self.data_loader is not None:
            self.data_loader(*args, **kwargs)
        else:
            raise Exception("Data loader not set")

    def load_config(self, path, play=True):
        config = ju.read_file(path)
        self.set_config(config, play)

    def set_config(self, config, play=True):
        self.play_config(config["properties"])

        self.loaded_data_config = config["data"]
        if play:
            self.play_config(self.loaded_data_config)

    def play_config(self, config):
        for config_item in config:
            type_ = config_item["type"]
            data_fn = getattr(self, type_)
            if data_fn is None:
                raise Exception("Can not load config, can not invoke", type_)

            if "args" in config_item:
                data_fn(*config_item["args"])
            elif "kwargs" in config_item:
                data_fn(**config_item["kwargs"])
            else:
                data_fn()

    def replay_config(self):
        if self.loaded_data_config is None:
            self.loaded_data_config = self.data_config

        self.data_config = []
        self.play_config(self.loaded_data_config)

    def to_json(self):
        return {
            "properties": self.property_config,
            "data": self.data_config
        }

    def dump_config(self, path: str):
        fu.create_parent_dirs(path)
        with open(path, 'wb') as config_file:
            pickle.dump({
                "properties": self.property_config,
                "data": self.data_config
            }, config_file)

    def set_shape(self, x=None, y=None):
        if x is None:
            x = self.x[0].shape

        if y is None:
            first_y = self.y[0]
            y = first_y.shape[0] if isinstance(first_y, np.ndarray) else 1

        self.shape_x = x
        self.shape_y = y

        self.property_config.append({"type": "set_shape", "kwargs": {"x": self.shape_x, "y": self.shape_y}})

    def enrich_data(self, *args, **kwargs):
        if self.data_enricher is not None:
            self.data_enricher(*args, **kwargs)
        else:
            raise Exception("Data loader not set")

    def split_test(self, split=0.8):
        self.test = DataManager()
        return self._split(self.test, split)

    def split_validation(self, split=0.8):
        self.validation = DataManager()
        return self._split(self.validation, split)

    def _split(self, data_set, split=0.8):
        self.x, data_set.x, self.y, data_set.y = train_test_split(
            self.x,
            self.y,
            train_size=split,
            test_size=None,
            random_state=2,
            shuffle=True,
            stratify=self.y)

        return data_set

    def scale_data(self, scaler=None):
        if scaler is None:
            scaler = nu.get_min_max(self.x)
        self.x = nu.scale_min_max(self.x, scaler)
        self.data_config.append({"type": "scale_data", "kwargs": {"scaler": scaler}})

    def get_balanced_weights(self):
        return get_balanced_weights(self.y)

    def find_top_features(self, num_top_features):
        return find_top_features(
            x=self.x, y=self.y,
            data_columns=self.data_columns,
            num_top_features=num_top_features)

    def select_top_features(self, num_top_features=0, top_features=None):
        if top_features is None:
            top_features = self.find_top_features(num_top_features)
        self.x = self.x[:, top_features]
        self.data_config.append({"type": "select_top_features", "kwargs": {"top_features": top_features}})

    def generate_unique_labels(self):
        self.labels = np.unique(self.y, return_counts=False)
        return self.labels

    def one_hot_encode_labels(self, num_classes=None):
        if num_classes is None:
            num_classes = nu.num_one_hot(self.y)
        self.y = nu.one_hot(self.y, num_classes)
        # self.data_config.append({"type": "one_hot_encode_labels", "kwargs": {"num_classes": num_classes}})

    def encode_labels(self, **kwargs):
        if self.output_type == "onehot":
            self.one_hot_encode_labels(**kwargs)

    def decode_labels(self, y):
        if self.output_type == "onehot":
            return nu.one_hot_reverse(y)
        elif self.output_type == "binary":
            return y.round()
        else:
            return y

    def label_confidence(self, y):
        # todo: fully implement
        if self.output_type == "onehot":
            return nu.one_hot_confidence(y)
        elif self.output_type == "binary":
            return nu.binary_confidence(y)
        else:
            return 1

    # def set_label_encoder(self, label_encoder=None):
    #     self.label_encoder = label_encoder
    #     self.data_config.append({"type": "set_label_encoder", "kwargs": {"label_encoder": label_encoder}})

    def reshape(self, x=None, y=None):
        if x is not None:
            self.x = self.x.reshape(x)
        if y is not None:
            self.y = self.y.reshape(y)
        self.data_config.append({"type": "reshape", "kwargs": {"x": x, "y": y}})

    def modify_x(self, modifier):
        self.x = modifier(self.x)

    def modify_data(self, modifier):
        self.x, self.y = modifier(self.x, self.y)

    def drop_columns(self, *columns):
        self.df.drop(columns=[*columns], inplace=True, errors='ignore')
        self.data_config.append({"type": "drop_columns", "args": columns})

    def data_from_column(self, start=None, end=None, without=None):
        data = self.df
        if start is not None and end is not None:
            data = data.loc[:, start:end]
        if without is not None:
            data = data.drop(without, axis=1, errors="ignore")
        self.x = data.values
        self.labels = list(data.columns)
        self.data_columns = list(data.columns)
        self.data_config.append(
            {"type": "data_from_column", "kwargs": {"start": start, "end": end, "without": without}})

    def label_from_column(self, column):
        self.y = self.df[column].values
        # todo: determine if need to track history for further training processing? dont need for value processing
        # self.data_config.append({"type": "label_from_column", "kwargs": {"column": column}})

    def rolling_window(self, window=30):
        rolling_x = []
        for i in range(window, len(self.x)):
            rolling_x.append(self.x[i - window:i])

        self.x = rolling_x
        self.y = self.y[window:]
        self.data_config.append({"type": "rolling_data", "kwargs": {"window": window}})
