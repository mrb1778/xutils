import tensorflow as tf
import numpy as np


def load_image(file_name, label=None, image_size=None, channels=3):
    image_bytes = tf.io.read_file(file_name)
    if tf.image.is_jpeg(image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=channels)
    else:
        image = tf.image.decode_png(image_bytes, channels=channels)

    image = tf.cast(image, tf.float32) / 255.0
    if image_size is not None:
        image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def load_image_split_horizontal(file_name, swap_sides=False):
    image = load_image(file_name)

    w = tf.shape(image)[1]

    center = w // 2
    left_image = image[:, center:, :]
    right_image = image[:, :center, :]

    left_image = tf.cast(left_image, tf.float32)
    right_image = tf.cast(right_image, tf.float32)

    return left_image, right_image


def random_crop(image, height, width, num_channels=3):
    return tf.image.random_crop(
        image,
        size=[height, width, num_channels])


def resize(image, dimensions):
    return tf.image.resize(
        image,
        size=dimensions,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@tf.function()
def random_jitter(image, dimensions, scale_factor=1.12):
    # height, width = tf.shape(image)[:3]
    height, width = dimensions
    image = resize(image, (int(height * scale_factor), int(width * scale_factor)))
    image = random_crop(image, height, width)

    return image


def prep_for_train(image, dimensions):
    image = resize(image, dimensions)
    image = random_jitter(image, dimensions)
    image = tf.image.random_flip_left_right(image)

    return image


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = tf.split(image1, num_or_size_splits=3, axis=3)
    b2, g2, r2 = tf.split(image2, num_or_size_splits=3, axis=3)
    if mode == 'normal':
        b_weight = tf.random.normal(shape=[1], mean=0.114, stddev=0.1)
        g_weight = np.random.normal(shape=[1], mean=0.587, stddev=0.1)
        r_weight = np.random.normal(shape=[1], mean=0.299, stddev=0.1)
    elif mode == 'uniform':
        b_weight = tf.random.uniform(shape=[1], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[1], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[1], minval=0.199, maxval=0.399)
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (b_weight + g_weight + r_weight)
    return output1, output2


