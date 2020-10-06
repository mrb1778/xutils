from abc import abstractmethod, ABC

from .numpy_utils import normalize, ensure_has_channels


class Dataset(object):
    _num_train = 0
    _num_test = 0
    _image_height = 0
    _image_width = 0
    _image_channels = 0
    _num_classes = 0

    def __init__(self, ):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    def get_data(self):
        data = self._load_data()
        data = self._preprocess_data(data)
        return data

    @abstractmethod
    def get_labels(self):
        pass

    def get_input_dimensions(self):
        return self._image_height, self._image_width, self._image_channels

    def get_num_classes(self):
        return self._num_classes

    def _preprocess_data(self, data):
        (x_train, y_train), (x_test, y_test) = data

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train, x_test, y_train, y_test = self._normalize_data(x_train, x_test, y_train, y_test)

        (
            self._num_train,
            self._image_height,
            self._image_width,
            self._image_channels
        ) = x_train.shape

        return (x_train, y_train), (x_test, y_test)

    def _normalize_data(self, x_train, x_test, y_train, y_test):
        return x_train, x_test, y_train, y_test


class ImageDataset(Dataset, ABC):
    def get_image_channels(self):
        return self._image_channels

    def _normalize_data(self, x_train, x_test, y_train, y_test):
        x_train, x_test = normalize(x_train, x_test)
        return super()._normalize_data(x_train, x_test, y_train, y_test)


class SingleChannelImageDataset(ImageDataset, ABC):
    def _normalize_data(self, x_train, x_test, y_train, y_test):
        x_train, x_test = ensure_has_channels(x_train, x_test, self.get_image_channels())
        return super()._normalize_data(x_train, x_test, y_train, y_test)
