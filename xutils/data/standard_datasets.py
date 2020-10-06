from tensorflow.keras import datasets
from xutils.data.dataset import SingleChannelImageDataset, ImageDataset


class Mnist(SingleChannelImageDataset):
    _num_classes = 10
    _image_channels = 1

    def _load_data(self):
        return datasets.mnist.load_data()

    def get_labels(self):
        return [
            "Zero",  # index 0
            "One",  # index 1
            "Two",  # index 2
            "Three",  # index 3
            "Four",  # index 4
            "Five",  # index 5
            "Six",  # index 6
            "Seven",  # index 7
            "Eight",  # index 8
            "Nine"  # index 9
        ]


class FashionMnist(SingleChannelImageDataset):
    _num_classes = 10
    _image_channels = 1

    def _load_data(self):
        return datasets.fashion_mnist.load_data()

    def get_labels(self):
        return ["T-shirt/top",  # index 0
                "Trouser",  # index 1
                "Pullover",  # index 2
                "Dress",  # index 3
                "Coat",  # index 4
                "Sandal",  # index 5
                "Shirt",  # index 6
                "Sneaker",  # index 7
                "Bag",  # index 8
                "Ankle boot"]  # index 9


class Cifar10(ImageDataset):
    _num_classes = 10
    _image_channels = 3

    def _load_data(self):
        return datasets.cifar10.load_data()

    def get_labels(self):
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]


class Cifar100(ImageDataset):
    _num_classes = 100
    _image_channels = 3

    def _load_data(self, is_fine=True):
        return datasets.cifar100.load_data(label_mode=('fine' if is_fine else 'coarse'))

    def get_labels(self):
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
