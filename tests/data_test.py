import json
import unittest

import numpy as np
from sklearn import preprocessing

import xutils.data.numpy_utils as nu
import xutils.data.json_utils as ju


class DataTest(unittest.TestCase):
    def test_json_dump(self):
        print(json.dumps((41, 58, 63)))
        print(json.dumps([41, 58, "63", [1, ("ddd", "www")]]))

        class X:
            def to_json(self):
                return {
                    "test1": 1,
                    "test2": 2
                }
        x = X()
        print(ju.write((41, 58, 63)))
        print(ju.write([41, 58, "63", [1, ("ddd", "www")]]))
        print(ju.write(x))
        self.assertEqual(ju.write({"x": x}), """{"x": {"test1": 1, "test2": 2}}""")

    def test_onehot(self):
        data = np.array([1, 0, 1])

        print("reshape", data.reshape(-1, 1))
        label_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')
        encoder_output = label_encoder.fit_transform(data.reshape(-1, 1))
        print(encoder_output)
        self.assertTrue(np.array_equal(encoder_output, nu.one_hot(data)))

    def test_minmax(self):
        data = np.array([
            [100, 50, 20],
            [20, 100, 50],
            [50, 20, 100]
        ], dtype='int32')

        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        sk_scaled_data = scaler.fit_transform(data)
        print("minmax", sk_scaled_data)

        min_max = nu.get_min_max(data)
        nu_scaled_data = nu.scale_min_max(data, min_max)
        print("nu.minmax", scaler.transform(data))

        self.assertTrue(np.array_equal(sk_scaled_data, nu_scaled_data))

    def test_binary_confidence(self):
        self.assertEqual(0.8, nu.binary_confidence(np.array([0.9, 0.9, 0.9, 0.9, 0.9])))
        self.assertEqual(0.8, nu.binary_confidence(np.array([0.1, 0.1, 0.1, 0.1, 0.1])))
        self.assertEqual(0.8, nu.binary_confidence(np.array([0.1, 0.1, 0.1, 0.9, 0.9])))
        self.assertEqual(1, nu.binary_confidence(np.array([1, 0, 1, 0, 1])))
        self.assertEqual(1, nu.binary_confidence(np.array([0, 0, 0, 0, 0])))
        self.assertEqual(0.5, nu.binary_confidence(np.array([0.25, 0.25, 0.75, 0.75])))