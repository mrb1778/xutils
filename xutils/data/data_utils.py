import numpy as np
import sklearn.utils as sku
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import json
import pickle
from operator import itemgetter
import os
import random as rn

import xutils.core.file_utils as fu


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


def compare_results(actual, predicted, print_results=True):
    actual = np.argmax(actual, axis=1)
    predicted = np.argmax(predicted, axis=1)

    e = np.equal(actual, predicted)
    # todo: holds [2] is causing issues with data with output size of 2, prob meant for 3 --> take num / classes or calc
    print("predicted", predicted)
    print("-------size", len(np.unique(predicted, return_counts=True)[1]))
    # holds = np.unique(predicted, return_counts=True)[1][-1]
    # delta = (holds / len(predicted) * 100)

    if print_results:
        print("Size", len(actual))
        print("Match", np.unique(predicted[e], return_counts=True))
        print("Base", np.unique(actual, return_counts=True))
        print("Test", np.unique(predicted, return_counts=True))
        # print("Delta", delta)

        conf_mat = skm.confusion_matrix(actual, predicted)
        print("Confusion Matrix\n", conf_mat)
        recall = []
        for i, row in enumerate(conf_mat):
            print(
                f"File: data_utils; Method: compare_results; Line: 59; Variable: np.round(row[i] = {np.round(row[i])}")
            print(f"File: data_utils; Method: compare_results; Line: 59; Variable: np.sum(row) = {np.sum(row)}")

            recall.append(np.round(row[i] / np.sum(row), 2))
            print(f"Recall of class {i} = {recall[i]}")
        print("Recall avg", sum(recall) / len(recall))

        print("F1 score (weighted)", skm.f1_score(actual, predicted, labels=None,
                                                  average='weighted', sample_weight=None))
        print("F1 score (macro)", skm.f1_score(actual, predicted, labels=None,
                                               average='macro', sample_weight=None))
        print("F1 score (micro)", skm.f1_score(actual, predicted, labels=None,
                                               average='micro',
                                               sample_weight=None))  # weighted and micro preferred in case of imbalance
        print("Cohen's Kappa", skm.cohen_kappa_score(actual, predicted))

    # return delta


def split_data(x, y, train_split=0.8, scale=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=train_split,
        test_size=None,
        random_state=2,
        shuffle=True,
        stratify=y)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        train_size=train_split,
        test_size=None,
        random_state=2,
        shuffle=True,
        stratify=y_train)

    data = {
        "train": {"x": x_train, "y": y_train},
        "test": {"x": x_test, "y": y_test},
        "validation": {"x": x_validation, "y": y_validation}
    }

    if scale:
        scale_split_data(data)

    return data


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

        self.history = []

    def load_data(self, *args, **kwargs):
        if self.data_loader is not None:
            self.data_loader(*args, **kwargs)
        else:
            raise Exception("Data loader not set")

    def load_config(self, path):
        with open(path, 'rb') as f:
            self.set_config(pickle.load(f))

    def set_config(self, config):
        for config_item in config:
            type_ = config_item["type"]
            data_fn = getattr(self, type_)
            if data_fn is None:
                raise Exception("Can not load config, can not invoke", type_)
            del config_item["type"]

            args = []
            if "args" in config_item:
                args = config_item["args"]
                del config_item["args"]

            data_fn(self, *args, **config_item)

    def dump_config(self, path):
        fu.create_parent_dirs(path)
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)

    def enrich_data(self, *args, **kwargs):
        if self.data_enricher is not None:
            self.data_enricher(*args, **kwargs)
        else:
            raise Exception("Data loader not set")

    def split_train(self, split=0.8):
        self.test = DataManager()
        return self._split(self.test, split)

    def split_validation(self, train_split=0.8):
        if self.test is None:
            raise Exception("Must Split Train 1st")

        self.validation = DataManager()
        return self._split(self.validation, train_split)

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
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
            scaler.fit(self.x)

        self.x = scaler.transform(self.x)

        if self.validation:
            self.validation.x = scaler.transform(self.validation.x)
        if self.test:
            self.test.x = scaler.transform(self.test.x)

        self.history.append({"type": "scale_data", "scaler": scaler})

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

        self.history.append({"type": "select_top_features", "top_features": top_features})

    def generate_unique_labels(self):
        self.labels = np.unique(self.y, return_counts=False)
        return self.labels

    def encode_labels(self):
        self.label_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')
        self.y = self.label_encoder.fit_transform(self.y.reshape(-1, 1))

        if self.validation:
            self.validation.y = self.label_encoder.transform(self.validation.y.reshape(-1, 1))
        if self.test:
            self.test.y = self.label_encoder.transform(self.test.y.reshape(-1, 1))

    def reshape(self, size):
        self.x = self.x.reshape(*size)
        self.history.append({"type": "reshape", "size": size})

    def modify_x(self, modifier):
        self.x = modifier(self.x)
        if self.validation:
            self.validation.x = modifier(self.validation.x)
        if self.test:
            self.test.x = modifier(self.test.x)

    def modify_data(self, modifier):
        self.x, self.y = modifier(self.x, self.y)
        if self.validation:
            self.validation.x, self.validation.y = modifier(self.validation.x, self.validation.y)
        if self.test:
            self.test.x, self.test.y = modifier(self.test.x, self.test.y)

    def drop_columns(self, *columns):
        self.df.drop(columns=[*columns], inplace=True, errors='ignore')

        self.history.append({"type": "drop_columns", "args": columns})

    def data_from_column(self, start, end):
        data = self.df.loc[:, start:end]
        self.x = data.values
        self.labels = list(data.columns)
        self.history.append({"type": "data_from_column", "start": start, "end": end})
        self.data_columns = list(data.columns)

    def label_from_column(self, column):
        self.y = self.df[column].values
        self.history.append({"type": "label_from_column", "column": column})
