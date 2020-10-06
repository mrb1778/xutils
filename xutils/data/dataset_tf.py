import os
import numpy as np
import tensorflow as tf


def create_dataset_from_csv(file_names, num_columns, has_header=True):
    return tf.data.experimental.CsvDataset(
        file_names,
        [[0.0]] * num_columns,
        header=has_header)


def create_dataset(x, labels, map_fn=None, batch_size=128, num_classes=None):
    dx = tf.data.Dataset.from_tensor_slices(x)

    dy = tf.data.Dataset.from_tensor_slices(labels)
    if num_classes is not None:
        dy = dy.map(lambda z: tf.one_hot(z, num_classes))

    dataset = tf.data.Dataset.zip((dx, dy))
    if map_fn is not None:
        dataset = dataset.map(map_fn)

    dataset = prepare_dataset(dataset)
    return dataset


def prepare_dataset(dataset, batch_size=128, shuffle_buffer_size=500, num_epochs=None):
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset


def create_dataset_image(images, labels, input_shape=None, batch_size=128, num_classes=10):
    def map_fn(image, label):
        x = tf.reshape(tf.cast(image, tf.float32), input_shape)
        x = tf.random_crop(x, input_shape)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.per_image_standardization(x)

        y = tf.one_hot(tf.cast(label, tf.uint8), num_classes)
        return x, y

    return create_dataset(
        images,
        labels,
        map_fn,
        batch_size
    )


def tf_load_image_dataset(label_directories, size=28):
    def _parse_function(filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image)
        image = tf.image.resize_images(image, [size, size])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        return image, label

    file_names = np.array()
    labels = []
    for label_key in label_directories:
        label_files = os.listdir(label_directories[label_key])
        np.concatenate([file_names, label_files])
        np.concatenate([labels, np.repeat(label_key, np.size(label_files))])

    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    dataset = dataset.map(_parse_function)
    return dataset


def tf_image_data_generator(images, labels, is_training, batch_size=128, num_classes=10):
    """Construct a data generator using `tf.Dataset`. """

    def map_fn(image, label):
        """Preprocess raw data to trainable input. """
        x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
        y = tf.one_hot(tf.cast(label, tf.uint8), num_classes)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def get_file_dataset(path, map_fn, batch_size=1, buffer_size=400, is_training=True):
    dataset = tf.data.Dataset.list_files(path)
    if is_training:
        dataset = dataset.shuffle(buffer_size)
    if map_fn is not None:
        dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size)
    return dataset



# # keras
# def make_train_generator():
#     train_datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
#     train_generator = train_datagen.flow_from_directory(
#         'train',
#         target_size=img_shape[:2],
#         batch_size=batch_size
#     )
#     return train_generator
#
#
# def make_test_generator():
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     test_generator = test_datagen.flow_from_directory(
#         'test',
#         target_size=img_shape[:2],
#         batch_size=batch_size
#     )
#     return test_generator
#
#
# def train_input_fn(batch_size):
#     dataset = tf.data.Dataset.from_generator(make_train_generator, output_types=(tf.float32, tf.float32), output_shapes=((batch_size, 224, 224, 3), 4))
#     #dataset = dataset.shuffle(1000)#.repeat().batch(batch_size, drop_remainder=True)
#     return dataset
#
#
# def test_input_fn(batch_size):
#     dataset = tf.data.Dataset.from_generator(make_test_generator, output_types=(tf.float32, tf.float32), output_shapes=((batch_size, 224, 224, 3), 4))
#     #dataset = dataset.shuffle(1000)#.repeat().batch(batch_size, drop_remainder=True)
#     return dataset
