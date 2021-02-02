from pathlib import Path

import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from . import train_utils as tu


def set_random_seed(seed=1235):
    tu.set_random_seed(seed)
    tf.random.set_seed(seed)


def create_image_generator(preprocessing_function=None):
    # This will do preprocessing and realtime data augmentation:
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set x mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each x by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     # randomly shift images horizontally (fraction of total width)
    #     width_shift_range=0.1,
    #     # randomly shift images vertically (fraction of total height)
    #     height_shift_range=0.1,
    #     shear_range=0.,  # set range for random shear
    #     zoom_range=0.,  # set range for random zoom
    #     channel_shift_range=0.,  # set range for random channel shifts
    #     # set mode for filling points outside the x boundaries
    #     fill_mode='nearest',
    #     cval=0.,  # value used for fill_mode = "constant"
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False,  # randomly flip images
    #     # set rescaling factor (applied before any other transformation)
    #     rescale=None,
    #     # set function that will be applied on each x
    #     preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     validation_split=0.0)
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set x mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each x by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    #

    generator = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=5. / 32,
        height_shift_range=5. / 32,
        # featurewise_center=True,
        preprocessing_function=preprocessing_function
    )

    return generator


def create_image_generator_flow(
        batch_size,
        generator=None,
        x_train=None, y_train=None,
        image_path=None,
        image_size=None,
        classes=None):
    if generator is None:
        generator = create_image_generator()

    if x_train is not None:
        generator.fit(x_train)
        return generator.flow(
            x_train,
            y_train,
            batch_size=batch_size
        )
    elif image_path is not None:
        # train_generator = train_datagen.flow_from_directory(
        #     args.train_dir,
        #     target_size=(IM_WIDTH, IM_HEIGHT),
        #     batch_size=batch_size,
        # )
        # validation_generator = test_datagen.flow_from_directory(
        #     args.val_dir,
        #     target_size=(IM_WIDTH, IM_HEIGHT),
        #     batch_size=batch_size,
        # )
        return generator.flow_from_directory(
            directory=image_path,
            batch_size=batch_size,
            classes=classes,
            target_size=image_size)


# def tfdata_generator(images, labels, is_training, input_shape, num_classes, batch_size=128):
#     """Construct a data generator using tf.Dataset"""
#
#     def preprocess_fn(image, label):
#         """A transformation function to preprocess raw data
#         into trainable x. """
#         x = tf.reshape(tf.cast(image, tf.float32), input_shape)
#         x = tf.image.random_flip_left_right(image)
#
#         y = tf.one_hot(tf.cast(label, tf.uint8), num_classes)
#         return x, y
#
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     if is_training:
#         dataset = dataset.shuffle(1000)  # depends on sample size
#
#     # Transform and batch data at the same time
#     dataset = dataset.apply(tf.contrib.data.map_and_batch(
#         preprocess_fn,
#         batch_size,
#         num_parallel_batches=8,  # cpu cores
#         drop_remainder=True if is_training else False))
#     dataset = dataset.repeat()
#     dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#
#     # training_set = tfdata_generator(x_train, y_train, is_training=True, batch_size=_BATCH_SIZE)
#     # testing_set  = tfdata_generator(x_test, y_test, is_training=False, batch_size=_BATCH_SIZE)
#     # dataset.make_one_shot_iterator(),
#     # model.fit(
#     #     training_set.make_one_shot_iterator(),
#     #     steps_per_epoch=len(x_train) // _BATCH_SIZE,
#     #     epochs=_EPOCHS,
#     #     validation_data=testing_set.make_one_shot_iterator(),
#     #     validation_steps=len(x_test) // _BATCH_SIZE,
#     #     verbose = 1)
#
#     return dataset


def fit_model(
        model,
        batch_size,
        epochs,
        callbacks,
        x_train=None, y_train=None,
        validation_data=None,
        model_name='model',
        generator_flow=None,
        dataset=None):
    start_time = time.time()

    if dataset is not None:
        results = model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=x_train.shape[0] // batch_size,
            callbacks=callbacks,
            workers=4,
            verbose=2
        )
    elif generator_flow is not None:
        results = model.fit_generator(
            generator_flow,
            # samples_per_epoch=len(x_train),
            steps_per_epoch=x_train.shape[0] // batch_size,
            # steps_per_epoch=50000 // BATCH_SIZE,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            shuffle=True,
            workers=4,
            verbose=2)
    else:
        results = model.fit(
            x_train, y_train,
            # steps_per_epoch=len(x_train) // _BATCH_SIZE,
            # validation_steps=len(x_test) // _BATCH_SIZE,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
            verbose=2)

    print("Elapsed Training Time: %.3f" % (time.time() - start_time))
    load_model(model, model_name)
    return results


# def lr_scheduler(epoch):
#     return learning_rate * (0.5 ** (epoch // lr_drop))
#
# reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

# def lr_scheduler(epoch):
#     # return learning_rate * (0.5 ** (epoch // lr_drop))
#     initial_lrate = 0.1
#     drop = 0.5
#     epochs_drop = 10.0
#     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#     return lrate
#
#
# reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


def get_model_path(model_name='model'):
    return model_name + '.weights.best.hdf5'


def load_model(model, model_name='model'):
    return model.load_weights(get_model_path(model_name))


def get_callbacks(model_name='model', learning_rate=1e-3, learning_rate_epochs_per_drop=10):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=get_model_path(model_name),
            monitor='val_loss',
            save_best_only=True,
            # save_weights_only=True,
            verbose=1
        ),
        # ReduceLROnPlateau(
        #     monitor='val_loss',
        #     patience=3,
        #     min_lr=1e-7,
        #     factor=0.2,
        #     verbose=1
        # ),
        # ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        step_decay_scheduler(
            initial_learning_rate=learning_rate,
            epochs_per_drop=learning_rate_epochs_per_drop
        ),
        # CyclicLR(base_lr=learning_rate),
        # LearningRateScheduler(lambda epoch: 1. / (1. + epoch)),
        # LRFinder(min_lr=learning_rate * .001,
        #          max_lr=learning_rate * 10,
        #          steps_per_epoch=np.ceil(num_epochs / batch_size),
        #          epochs=3),
        TensorBoard(
            log_dir='./tensorboard',
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False)
    ]


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def plot_history(history, fields=None, title="Model Loss", measure="Loss"):
    if fields is None:
        fields = ["loss", "val_loss"]

    plt.figure()
    for field in fields:
        plt.plot(history.history[field])

    plt.title(title)
    plt.ylabel(measure)
    plt.xlabel("Epoch")
    plt.legend(fields, loc='upper left')
    plt.show()


def plot_sample_results(x_test, y_test, y_hat, labels, num_random=15):
    figure = plt.figure(figsize=(20, 8))
    for i, index in enumerate(np.random.choice(x_test.shape[0], size=num_random, replace=False)):
        ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        # Display each image
        ax.imshow(np.squeeze(x_test[index]))
        predict_index = np.argmax(y_hat[index])
        true_index = np.argmax(y_test[index])
        # Set the title for each image
        ax.set_title("{} ({})".format(labels[predict_index],
                                      labels[true_index]),
                     color=("green" if predict_index == true_index else "red"))

    plt.show(block=True)


def step_decay_scheduler(initial_learning_rate=1e-3, decay_factor=0.5, epochs_per_drop=10):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_learning_rate * (decay_factor ** (epoch // epochs_per_drop))
        # return initial_learning_rate * (decay_factor ** np.floor(epoch / epochs_per_drop))
        # return initial_learning_rate * decay_factor ** epoch

    return LearningRateScheduler(
        schedule,
        verbose=1)


# def lr_scheduler(epoch):
#     return learning_rate * (0.5 ** (epoch // lr_drop))

class LRFinder(Callback):
    """
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    """

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        """Calculate the learning rate."""
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        """Initialize the learning rate to the minimum value at the start of training."""
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        """Helper function to quickly inspect the learning rate schedule."""
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

    def plot_loss(self):
        """Helper function to quickly observe the learning rate experiment results."""
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


# class SGDRScheduler(Callback):
#     """Cosine annealing learning rate scheduler with periodic restarts.
#     # Usage
#         ```python
#             schedule = SGDRScheduler(min_lr=1e-5,
#                                      max_lr=1e-2,
#                                      steps_per_epoch=np.ceil(epoch_size/batch_size),
#                                      lr_decay=0.9,
#                                      cycle_length=5,
#                                      mult_factor=1.5)
#             model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
#         ```
#     # Arguments
#         min_lr: The lower bound of the learning rate range for the experiment.
#         max_lr: The upper bound of the learning rate range for the experiment.
#         steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
#         lr_decay: Reduce the max_lr after the completion of each cycle.
#                   Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
#         cycle_length: Initial number of epochs in a cycle.
#         mult_factor: Scale epochs_to_restart after each full cycle completion.
#     # References
#         Blog post: jeremyjordan.me/nn-learning-rate
#         Original paper: http://arxiv.org/abs/1608.03983
#     """
#
#     def __init__(self,
#                  min_lr,
#                  max_lr,
#                  steps_per_epoch,
#                  lr_decay=1,
#                  cycle_length=10,
#                  mult_factor=2):
#
#         self.min_lr = min_lr
#         self.max_lr = max_lr
#         self.lr_decay = lr_decay
#
#         self.batch_since_restart = 0
#         self.next_restart = cycle_length
#
#         self.steps_per_epoch = steps_per_epoch
#
#         self.cycle_length = cycle_length
#         self.mult_factor = mult_factor
#
#         self.history = {}
#
#     def clr(self):
#         """Calculate the learning rate."""
#         fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
#         lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
#         return lr
#
#     def on_train_begin(self, logs={}):
#         """Initialize the learning rate to the minimum value at the start of training."""
#         logs = logs or {}
#         K.set_value(self.model.optimizer.lr, self.max_lr)
#
#     def on_batch_end(self, batch, logs={}):
#         """Record previous batch statistics and update the learning rate."""
#         logs = logs or {}
#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
#
#         self.batch_since_restart += 1
#         K.set_value(self.model.optimizer.lr, self.clr())
#
#     def on_epoch_end(self, epoch, logs={}):
#         """Check for end of current cycle, apply restarts when necessary."""
#         if epoch + 1 == self.next_restart:
#             self.batch_since_restart = 0
#             self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
#             self.next_restart += self.cycle_length
#             self.max_lr *= self.lr_decay
#             self.best_weights = self.model.get_weights()
#
#     def on_train_end(self, logs={}):
#         """Set weights to the values from the end of the most recent cycle for best performance."""
#         self.model.set_weights(self.best_weights)


# class CyclicLR(Callback):
#     """This callback implements a cyclical learning rate policy (CLR).
#     The method cycles the learning rate between two boundaries with
#     some constant frequency.
#     # Arguments
#         base_lr: initial learning rate which is the
#             lower boundary in the cycle.
#         max_lr: upper boundary in the cycle. Functionally,
#             it defines the cycle amplitude (max_lr - base_lr).
#             The lr at any cycle is the sum of base_lr
#             and some scaling of the amplitude; therefore
#             max_lr may not actually be reached depending on
#             scaling function.
#         step_size: number of training iterations per
#             half cycle. Authors suggest setting step_size
#             2-8 x training iterations in epoch.
#         mode: one of {triangular, triangular2, exp_range}.
#             Default 'triangular'.
#             Values correspond to policies detailed above.
#             If scale_fn is not None, this argument is ignored.
#         gamma: constant in 'exp_range' scaling function:
#             gamma**(cycle iterations)
#         scale_fn: Custom scaling policy defined by a single
#             argument lambda function, where
#             0 <= scale_fn(x) <= 1 for all x >= 0.
#             mode paramater is ignored
#         scale_mode: {'cycle', 'iterations'}.
#             Defines whether scale_fn is evaluated on
#             cycle number or cycle iterations (training
#             iterations since start of cycle). Default is 'cycle'.
#     The amplitude of the cycle can be scaled on a per-iteration or
#     per-cycle basis.
#     This class has three built-in policies, as put forth in the paper.
#     "triangular":
#         A basic triangular cycle w/ no amplitude scaling.
#     "triangular2":
#         A basic triangular cycle that scales initial amplitude by half each cycle.
#     "exp_range":
#         A cycle that scales initial amplitude by gamma**(cycle iterations) at each
#         cycle iteration.
#     For more detail, please see paper.
#     # Example for CIFAR-10 w/ batch size 100:
#         ```python
#             clr = CyclicLR(base_lr=0.001, max_lr=0.006,
#                                 step_size=2000., mode='triangular')
#             model.fit(X_train, Y_train, callbacks=[clr])
#         ```
#     Class also supports custom scaling functions:
#         ```python
#             clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
#             clr = CyclicLR(base_lr=0.001, max_lr=0.006,
#                                 step_size=2000., scale_fn=clr_fn,
#                                 scale_mode='cycle')
#             model.fit(X_train, Y_train, callbacks=[clr])
#         ```
#     # References
#       - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
#     """
#
#     def __init__(
#             self,
#             base_lr=0.001,
#             max_lr=0.006,
#             step_size=2000.,
#             mode='triangular',
#             gamma=1.,
#             scale_fn=None,
#             scale_mode='cycle'):
#         super(CyclicLR, self).__init__()
#
#         assert mode in ['triangular', 'triangular2',
#                         'exp_range'], "mode must be one of 'triangular', 'triangular2', or 'exp_range'"
#         self.base_lr = base_lr
#         self.max_lr = max_lr
#         self.step_size = step_size
#         self.mode = mode
#         self.gamma = gamma
#         if scale_fn is None:
#             if self.mode == 'triangular':
#                 self.scale_fn = lambda x: 1.
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'triangular2':
#                 self.scale_fn = lambda x: 1 / (2. ** (x - 1))
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'exp_range':
#                 self.scale_fn = lambda x: gamma ** (x)
#                 self.scale_mode = 'iterations'
#         else:
#             self.scale_fn = scale_fn
#             self.scale_mode = scale_mode
#         self.clr_iterations = 0.
#         self.trn_iterations = 0.
#         self.history = {}
#
#         self._reset()
#
#     def _reset(self, new_base_lr=None, new_max_lr=None,
#                new_step_size=None):
#         """Resets cycle iterations.
#         Optional boundary/step size adjustment.
#         """
#         if new_base_lr is not None:
#             self.base_lr = new_base_lr
#         if new_max_lr is not None:
#             self.max_lr = new_max_lr
#         if new_step_size is not None:
#             self.step_size = new_step_size
#         self.clr_iterations = 0.
#
#     def clr(self):
#         cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
#         x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
#         if self.scale_mode == 'cycle':
#             return self.base_lr + (self.max_lr - self.base_lr) * \
#                    np.maximum(0, (1 - x)) * self.scale_fn(cycle)
#         else:
#             return self.base_lr + (self.max_lr - self.base_lr) * \
#                    np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
#
#     def on_train_begin(self, logs={}):
#         logs = logs or {}
#
#         if self.clr_iterations == 0:
#             K.set_value(self.model.optimizer.lr, self.base_lr)
#         else:
#             K.set_value(self.model.optimizer.lr, self.clr())
#
#     def on_batch_end(self, epoch, logs=None):
#
#         logs = logs or {}
#         self.trn_iterations += 1
#         self.clr_iterations += 1
#         K.set_value(self.model.optimizer.lr, self.clr())
#
#         self.history.setdefault(
#             'lr', []).append(
#             K.get_value(
#                 self.model.optimizer.lr))
#         self.history.setdefault('iterations', []).append(self.trn_iterations)
#
#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs['lr'] = K.get_value(self.model.optimizer.lr)


def evaluate_model_results(model, x_train, y_train, x_valid, y_valid, x_test, y_test, labels):
    # loss, acc, topk_acc = model.evaluate_generator(
    #     test_gen,
    #     steps=10000 // BATCH_SIZE)
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print('\n', 'Train accuracy:', score_train[1])

    score_test = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score_test[1])

    score_validation = model.evaluate(x_valid, y_valid, verbose=0)
    print('\n', 'Validation accuracy:', score_validation[1])

    y_hat = model.predict(x_test)
    residuals = np.argmax(y_hat, 1) != np.argmax(y_test, 1)
    loss = sum(residuals) / len(residuals)
    print("the validation 0/1 loss is: ", loss)

    plot_sample_results(x_test, y_test, y_hat, labels)

    # other metric
    # yPred = np.argmax(predicted_x, axis=1)
    # accuracy = metrics.categorical_accuracy(y_test, yPred) * 100
    # error = 100 - accuracy
    # print("Accuracy : ", accuracy)
    # print("Error : ", error)


# def plot_confusion_matrix(num_classes, zero_diagonal=False):
#     """Plot a confusion matrix."""
#     # Calculate confusion matrix
#     y_val_i = y_val.flatten()
#     y_val_pred = model.predict(X_val)
#     y_val_pred_i = y_val_pred.argmax(1)
#     cm = np.zeros((num_classes, num_classes), dtype=np.int)
#     for i, j in zip(y_val_i, y_val_pred_i):
#         cm[i][j] += 1
#
#     acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
#     print("Validation accuracy: %0.4f" % acc)
#
#
#     n = len(cm)
#     size = int(n / 4.)
#     fig = plt.figure(figsize=(size, size), dpi=80, )
#     plt.clf()
#     ax = fig.add_subplot(111)
#     ax.set_aspect(1)
#     res = ax.imshow(np.array(cm), cmap=plt.cm.viridis,
#                     interpolation='nearest')
#     width, height = cm.shape
#     fig.colorbar(res)
#     plt.savefig('confusion_matrix.png', format='png')


# def init_system():
#     pass
#     # readd for tf2
#     # config = tf.ConfigProto()
#     # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#     # config.log_device_placement = True  # to log device placement (on which device the operation ran)
#     # # (nothing gets printed in Jupyter, only if you run it standalone)
#     # sess = tf.Session(config=config)
#     # set_session(sess)  # set this TensorFlow session as the default session for Keras
#
#     # config = K.tf.ConfigProto()
#     # config.gpu_options.allow_growth = True
#     # session = K.tf.Session(config=config)


# def limit_mem():
#     pass
#     # tf.config.gpu.set_per_process_memory_growth(True)
#     # K.get_session().close()
#     # cfg = tf.ConfigProto(inter_op_parallelism_threads=True,
#     #                      intra_op_parallelism_threads=True,
#     #                      allow_soft_placement=True)
#     # cfg.gpu_options.allow_growth = True
#     # K.set_session(tf.Session(config=cfg))


def compile_model(model, optimizer='adam', loss_fn='categorical_crossentropy', is_categorical=True, is_binary=False):
    if loss_fn == 'wasserstein_loss':
        loss_fn = wasserstein_loss
    elif is_binary:
        loss_fn = 'binary_crossentropy'

    metrics = ['accuracy']
    if is_categorical:
        metrics.append(top_k_categorical_accuracy)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def get_optimizer(optimizer_name="adam", learning_rate=1e-3, learning_rate_decay=0, beta_1=0.9):
    # sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    # opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    # optimizer=sgd,
    # optimizer=opt_rms,
    # optimizer=optimizers.SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True),
    if optimizer_name == "adam":
        return Adam(lr=learning_rate, beta_1=beta_1)
    elif optimizer_name == "sgd":
        return SGD(lr=learning_rate, decay=learning_rate_decay, momentum=0.9, nesterov=True)


def get_distribution_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:
        return tf.distribute.get_strategy()


def save_model(model, model_name, path):
    # model_path = str(Path(path, model_name + ".json"))
    weights_path = str(Path(path, model_name + "_weights.hdf5"))
    # options = {
    #     "file_arch": model_path,
    #     "file_weight": weights_path
    # }
    # json_string = model.to_json()
    # open(options['file_arch'], 'w').write(json_string)
    model.save_weights(weights_path)


def init_logs(path, model):
    callback = TensorBoard(path)
    callback.set_model(model)
    return callback


# def write_log(callback, names, logs, batch_no):
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()


# class TensorNetwork:
#     def __init__(self, graph_def=None, path_to_graph_def=None):
#         if graph_def is not None:
#             self.graph_def = graph_def
#         elif path_to_graph_def is not None:
#             self.graph_def = load_graph_def(path_to_graph_def)
#
#         self.weight_biases = None
#         self.last_layer = None
#         self.session = None
#         self.saver = tf.train.Saver()
#         self.optimizer = None
#
#         self.x = None
#         self.y = None
#
#     def get_model_dir(self):
#         """Helper function to get the model directory.
#
#         Returns:
#             string of model directory
#         """
#         return "{0}_{1}_{2}_{3}".format(self.f.experiment_name,
#                                         self.f.dataset,
#                                         self.f.batch_size,
#                                         self.f.output_size)
#
#     def create_weight_biases(self, sizes):
#         self.weight_biases = create_weight_biases(sizes)
#
#     def create_layers(self, input_layer, default_fn=tf.nn.elu, last_fn=tf.nn.softmax, dropout=None):
#         self.last_layer = create_layers(input_layer, self.weight_biases, default_fn, last_fn, dropout)
#
#     def set_session(self, session):
#         self.session = session
#
#     def start_session(self):
#         self.session = tf.Session()
#         return session
#
#     def set_optimizer(self, ):
#         self.optimizer = get_optimizer()
#
#     def save(self, path, step=None):
#         return self.saver.save(
#             self.session,
#             path,
#             write_meta_graph=False,
#             global_step=step)
#
#     def load(self, directory, path):
#         ckpt = tf.train.get_checkpoint_state(directory)
#         if ckpt and ckpt.model_checkpoint_path:
#             # load model
#             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#             self.saver.restore(self.sess,
#                                os.path.join(checkpoint_dir, ckpt_name))
#             return True
#         else:
#             return False
#
#         return self.saver.restore(self.session, path)
#
#     def run(self, x, y,
#             batch_size,
#             batch_generator,
#             training_iters,
#             display_step=10,
#             keep_prob=tf.placeholder(tf.float32)):
#         cost_function = get_cost_function()  # (,,batch_size)
#         optimizer = get_optimizer(cost_function)
#         accuracy = get_accuracy()
#
#         init = tf.global_variables_initializer()
#
#         # Launch the graph
#         with tf.Session() as sess:
#             sess.run(init)
#             step = 1
#             # Keep training until reach max iterations
#             while step * batch_size < training_iters:
#                 batch_x, batch_y = batch_generator(batch_size)
#                 # Run optimization op (backprop)
#                 sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#                 if step % display_step == 0:
#                     # Calculate batch loss and accuracy
#                     # _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
#
#                     loss, acc = sess.run([cost_function, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
#                     print("Iter " + str(step * batch_size) + ", Mini batch Loss= " +
#                           "{:.6f}".format(loss) + ", Training Accuracy= " +
#                           "{:.5f}".format(acc))
#                 step += 1
#             print("Optimization Finished!")
#
#             # Calculate accuracy for 256 mnist test images
#             # print("Testing Accuracy:",
#             #       sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#             #                                     y: mnist.test.labels[:256],
#             #                                     keep_prob: 1.}))


# def get_output_tensor_by_name(graph, layer, prefix="import"):  # renamed from T
#     """Helper for getting layer output tensor"""
#     return graph.get_tensor_by_name("{}/{}:0".format(prefix, layer))
#
#
# def get_tensor_channel(name="mixed4d_3x3_bottleneck_pre_relu", channel=139):
#     """# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
#     # to have non-zero gradients for features with negative initial activations.
#     """
#     tensor_by_name = get_output_tensor_by_name(name)
#     return tensor_by_name[:, :, :, channel]


# def wrap_tf_func(*argtypes):  # renamed from tffunc
#     """Helper that transforms TF-graph generating function into a regular one.
#     See "resize" function below.
#     """
#     placeholders = list(map(tf.placeholder, argtypes))
#
#     def wrap(f):
#         out = f(*placeholders)
#
#         def wrapper(*args, **kw):
#             return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
#
#         return wrapper
#
#     return wrap


# def create_op(func, **placeholders):
#     op = func(**placeholders)
#
#     def f(**kwargs):
#         feed_dict = {}
#         for argname, argvalue in kwargs.items():
#             placeholder = placeholders[argname]
#             feed_dict[placeholder] = argvalue
#         return tf.get_default_session().run(op, feed_dict=feed_dict)
#
#     return f


def print_model_info(graph, op_type='Conv2D', print_all=False):
    if print_all:
        for op in graph.get_operations():
            if 'import/' in op.name:
                print("Name: {}, Type: {}".format(op.name, op.type))

    layers = [op.name for op in graph.get_operations() if op.type == op_type and 'import/' in op.name]

    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print(layers)
    print('Total number of feature channels:', sum(feature_nums))


def get_tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


# def load_graph_def(path):
#     with tf.gfile.FastGFile(path, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         return graph_def


# def create_weight_bias(num_in, num_out, patch_size=None, random_bias=True):
#     weight_shape = [num_in, num_out] \
#         if patch_size is None \
#         else [patch_size, patch_size, num_in, num_out]
#     weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1))
#
#     bias_initial_value = tf.ones([num_out] / 10) \
#         if random_bias \
#         else tf.zeros([num_out])
#     biases = tf.Variable(bias_initial_value)
#     return weights, biases


# def create_weight_biases(sizes):
#     weight_biases = []
#
#     previous_size = None
#     for idx, size in enumerate(sizes):
#         if idx > 0 and previous_size is not None:
#             weight_biases.append(create_weight_bias(previous_size, size))
#         previous_size = size
#
#     return weight_biases


def create_linear_layer(input_layer, weight_biases, default_fn=tf.nn.elu, last_fn=tf.nn.softmax, dropout=None):
    previous_output = None

    for idx, weight, bias in enumerate(weight_biases):
        if idx == 0:
            previous_output = input_layer
        else:
            fn = default_fn if idx < weight_biases.size - 1 else last_fn

            previous_output = fn(tf.matmul(previous_output, weight) + bias)

            if dropout is not None:
                previous_output = tf.nn.dropout(previous_output, dropout)

    return previous_output


def create_layers(input_layer, weight_biases,
                  default_fn=tf.nn.elu,  # relu can get stuck, so leaky relu? elu?
                  last_fn=tf.nn.softmax,  # softmax = classification, linear for regression
                  dropout=None):
    with tf.name_scope('Model'):
        for idx, weight, bias in enumerate(weight_biases):
            if idx == 0:
                previous_output = input_layer
            else:
                fn = default_fn if idx < weight_biases.size - 1 else last_fn

                previous_output = fn(tf.matmul(previous_output, weight) + bias)

                if dropout is not None:
                    previous_output = tf.nn.dropout(previous_output, dropout)

        return previous_output


def get_cost_function(predictions, labels, batch_size=100, reduce_fn=tf.reduce_mean):
    """Distribution cost function.  Distance b/t predicted and output labels (one hot).  Cross entropy"""
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
        cost = reduce_fn(cross_entropy) * batch_size
        tf.summary.scalar("loss", cost)
        return cost


def get_accuracy(prediction, y):
    with tf.name_scope('Accuracy'):
        accuracy = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy


def get_decay_learning_rate(step, max_learning_rate=0.003, min_learning_rate=0.0001, decay_speed=2000.0):
    return min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-step / decay_speed)


def layer_conv(features, weights, bias):
    conv = tf.nn.conv2d(features, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def layer_relu(features):
    return tf.nn.relu(features)


def conv2d(x, W, b, strides=1, activation_fn=tf.nn.elu):
    """Conv2D wrapper, with bias and relu activation"""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return activation_fn(x)


def layer_pool(features, pooling='max', k=2):
    """creates more / smaller versions of x.  max pooling: classification. loses information about original x area.
    average pooling does not"""

    if pooling == 'avg':
        return tf.nn.avg_pool(features, ksize=(1, k, k, 1), strides=(1, k, k, 1), padding='SAME')
    else:
        return tf.nn.max_pool(features, ksize=(1, k, k, 1), strides=(1, k, k, 1), padding='SAME')


def conv_net(x, weights, biases, dropout, activation_fn=tf.nn.elu):
    # Reshape x picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = layer_pool(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = layer_pool(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer x
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = activation_fn(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# def rnn(x,  # tf.placeholder("float", [None, n_steps, n_input])
#         weights,  # tf.Variable(tf.random_normal([n_hidden, n_classes]))
#         biases,  # tf.Variable(tf.random_normal([n_classes]))
#         n_steps=28,  # time steps
#         n_hidden=128  # hidden layer num of features
#         ):
#     # Prepare data shape to match `rnn` function requirements
#     # Current data x shape: (batch_size, n_steps, n_input)
#     # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
#
#     # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#     x = tf.unstack(x, n_steps, 1)
#
#     # Define a lstm cell with tensorflow
#     lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#
#     # Get lstm cell output
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights) + biases


# def load_graph(filename):
#     """Unpersists graph from file as default graph."""
#     with tf.gfile.FastGFile(filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#
#
# def load_labels(filename):
#     """Read in labels, one label per line."""
#     return [line.rstrip() for line in tf.gfile.GFile(filename)]

#
# # ops https://github.com/affinelayer/pix2pix-tensorflow/blob/master/tools/tfimage.py
# downscale = create_op(
#     func=tf.image.resize_images,
#     images=tf.placeholder(tf.float32, [None, None, None]),
#     size=tf.placeholder(tf.int32, [2]),
#     method=tf.image.ResizeMethod.AREA,
# )
#
# upscale = create_op(
#     func=tf.image.resize_images,
#     images=tf.placeholder(tf.float32, [None, None, None]),
#     size=tf.placeholder(tf.int32, [2]),
#     method=tf.image.ResizeMethod.BICUBIC,
# )
#
# decode_jpeg = create_op(
#     func=tf.image.decode_jpeg,
#     contents=tf.placeholder(tf.string),
# )
#
# decode_png = create_op(
#     func=tf.image.decode_png,
#     contents=tf.placeholder(tf.string),
# )
#
# rgb_to_grayscale = create_op(
#     func=tf.image.rgb_to_grayscale,
#     images=tf.placeholder(tf.float32),
# )
#
# grayscale_to_rgb = create_op(
#     func=tf.image.grayscale_to_rgb,
#     images=tf.placeholder(tf.float32),
# )
#
# encode_jpeg = create_op(
#     func=tf.image.encode_jpeg,
#     image=tf.placeholder(tf.uint8),
# )
#
# encode_png = create_op(
#     func=tf.image.encode_png,
#     image=tf.placeholder(tf.uint8),
# )
#
# crop = create_op(
#     func=tf.image.crop_to_bounding_box,
#     image=tf.placeholder(tf.float32),
#     offset_height=tf.placeholder(tf.int32, []),
#     offset_width=tf.placeholder(tf.int32, []),
#     target_height=tf.placeholder(tf.int32, []),
#     target_width=tf.placeholder(tf.int32, []),
# )
#
# pad = create_op(
#     func=tf.image.pad_to_bounding_box,
#     image=tf.placeholder(tf.float32),
#     offset_height=tf.placeholder(tf.int32, []),
#     offset_width=tf.placeholder(tf.int32, []),
#     target_height=tf.placeholder(tf.int32, []),
#     target_width=tf.placeholder(tf.int32, []),
# )
#
# to_uint8 = create_op(
#     func=tf.image.convert_image_dtype,
#     image=tf.placeholder(tf.float32),
#     dtype=tf.uint8,
#     saturate=True,
# )
#
# to_float32 = create_op(
#     func=tf.image.convert_image_dtype,
#     image=tf.placeholder(tf.uint8),
#     dtype=tf.float32,
# )


def gram_matrix(x, area, depth):
    f = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(f), f)
    return g


def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # # precision = TP/TP+FP, recall = TP/TP+FN
    # rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    # def get_precision(i, conf_mat):
    #     print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
    #     precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
    #     recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
    #     tf.add(i, 1)
    #     return i, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    # def condition(i, conf_mat):
    #     return tf.less(i, 3)
    #
    # i = tf.constant(3)
    # i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(tf.math.multiply(f1s, weights))
    return weighted_f1


def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})


def test_install():
    print(sys.version)
    print('Tensorflow Version:', tf.version.VERSION)
    print("Local Devices", tf.config.list_physical_devices('GPU'))
    hello = tf.constant('Hello, TensorFlow!')
    print("Hello", hello)

    x = [[2.]]
    m = tf.matmul(x, x)
    print("hello, {}".format(m))


class KerasTrainer(tu.Trainer):
    def __init__(self):
        pass

# https://gist.github.com/datlife/abfe263803691a8864b7a2d4f87c4ab8
# def tfdata_generator(images, labels, is_training, num_classes, batch_size=128):
#     """Construct a data generator using `tf.Dataset`. """
#
#     def map_fn(image, label):
#         """Preprocess raw data to trainable x. """
#         x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
#         y = tf.one_hot(tf.cast(label, tf.uint8), num_classes)
#         return x, y
#
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#
#     if is_training:
#         dataset = dataset.shuffle(1000)  # depends on sample size
#
#     dataset = dataset.map(map_fn)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat()
#     dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#
#     return dataset
#
#
# class DataGenerator(Sequence):
#     """Generates data for Keras"""
#
#     def __init__(self, list_ids, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_ids = list_ids
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.floor(len(self.list_ids) / self.batch_size))
#
#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         # Find list of IDs
#         list_ids_temp = [self.list_ids[k] for k in indexes]
#
#         # Generate data
#         X, y = self.__data_generation(list_ids_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         self.indexes = np.arange(len(self.list_ids))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, list_ids_temp):
#         """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty(self.batch_size, dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_ids_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')
#
#             # Store class
#             y[i] = self.labels[ID]
#
#         return X, to_categorical(y, num_classes=self.n_classes)
