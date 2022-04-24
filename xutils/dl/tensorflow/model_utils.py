import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.backend import *
from xutils import SubPixelUpscaling
import mrbuilder

ExpandDimension = lambda axis: Lambda(lambda x: expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: squeeze(x, axis))


def conv_act_bn(
        x,
        size,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        activation_alpha=0.3,
        sep_conv=False,
        conv_transpose=False,
        kernel_regularizer=None,
        l2_weight_decay=None,
        momentum=0.99,
        do_dropout=False,
        dropout_rate=0.4,
        dilation_rate=(1, 1),
        do_batch_norm=True):

    if l2_weight_decay is not None:
        kernel_regularizer = l2(l2_weight_decay)

    if sep_conv:
        x = SeparableConv2D(
            size,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=kernel_regularizer)(x)
    elif conv_transpose:
        x = Conv2DTranspose(
            size,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=kernel_regularizer)(x)
    else:
        x = Conv2D(
            int(size),
            kernel_size,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            kernel_regularizer=kernel_regularizer)(x)

    if activation is not None:
        if activation == 'LeakyReLU':
            x = LeakyReLU(activation_alpha)(x)
        else:
            x = Activation(activation)(x)

    if do_batch_norm:
        x = BatchNormalization(momentum=momentum)(x)

    if do_dropout:
        x = Dropout(dropout_rate)(x)

    return x


mrbuilder.register_layer_builder(
    "conv_act_bn",
    lambda layer_options, layer_connection:
        conv_act_bn(x=layer_connection,
                    size=layer_options("size"),
                    kernel_size=layer_options("kernel", 3),
                    strides=layer_options("strides", 1),
                    dilation_rate=layer_options("dilation", (1, 1)),
                    activation=layer_options("activation", "relu"),
                    activation_alpha=layer_options("activationAlpha", 0.3),
                    padding=layer_options("padding", "same"),
                    momentum=layer_options("momentum", 0.99),
                    do_dropout=layer_options("doDropout", False),
                    dropout_rate=layer_options("dropoutRate", 0.4),
                    do_batch_norm=layer_options("doBatchNorm", True),
                    sep_conv=layer_options("sepConv", False),
                    conv_transpose=layer_options("convTranspose", False),
                    l2_weight_decay=layer_options("weightDecay")))


def simple_factorized_conv(filters, kernel, *args, **kwargs):
    kwargs["activation"] = None

    def __inner(inp):
        cnn1 = Conv2D(filters, (kernel[0], 1), *args, **kwargs)(inp)
        cnn2 = Conv2D(filters, (1, kernel[1]), *args, **kwargs)(cnn1)

        return cnn2

    return __inner


def sobel_layer(x):
    sobel = tf.image.sobel_edges(x)
    reshape = tf.reshape(sobel, [-1, x.shape[1], x.shape[2], x.shape[3] * 2])
    return reshape


def sobel_layer_stacked(x):
    sobel = sobel_layer(x)
    return concat_channels(x, sobel)


def concat_channels(*args, channel_axis=3):
    return tf.concat(args, axis=channel_axis)


def channel_difference(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    below_delta = tf.subtract(pad, below)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.subtract(pad, right)

    return concat_channels(below_delta, right_delta)


def bi_channel_difference(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    below_delta = tf.subtract(pad, below)

    above = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])
    above_delta = tf.subtract(pad, above)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.subtract(pad, right)

    left = tf.pad(x, [[0, 0], [1, 1], [2, 0], [0, 0]])
    left_delta = tf.subtract(pad, left)

    return concat_channels(below_delta, above_delta, right_delta, left_delta)


def bi_channel_difference_fixed(x):
    x = convert_to_channels_first(x)

    pad = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]])

    below = tf.pad(x, [[0, 0], [0, 0], [0, 2], [1, 1]])
    below_delta = tf.subtract(pad, below)

    above = tf.pad(x, [[0, 0], [0, 0], [2, 0], [1, 1]])
    above_delta = tf.subtract(pad, above)

    right = tf.pad(x, [[0, 0], [0, 0], [1, 1], [0, 2]])
    right_delta = tf.subtract(pad, right)

    left = tf.pad(x, [[0, 0], [0, 0], [1, 1], [2, 0]])
    left_delta = tf.subtract(pad, left)

    x = concat_channels(below_delta, above_delta, right_delta, left_delta, channel_axis=1)

    return convert_to_channels_last(x)


mrbuilder.register_layer_builder(
    "bi_channel_difference_fixed",
    lambda layer_options, layer_connection:
        Lambda(bi_channel_difference_fixed)(layer_connection))


def bi_channel_difference_stacked(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    below_delta = tf.subtract(pad, below)

    above = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])
    above_delta = tf.subtract(pad, above)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.subtract(pad, right)

    left = tf.pad(x, [[0, 0], [1, 1], [2, 0], [0, 0]])
    left_delta = tf.subtract(pad, left)

    return concat_channels(pad, below_delta, above_delta, right_delta, left_delta)


def squared_channel_difference(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    below_delta = tf.squared_difference(pad, below)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.squared_difference(pad, right)

    return concat_channels(below_delta, right_delta)


def squared_bi_channel_difference(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    below_delta = tf.squared_difference(pad, below)

    above = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])
    above_delta = tf.squared_difference(pad, above)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.squared_difference(pad, right)

    left = tf.pad(x, [[0, 0], [1, 1], [2, 0], [0, 0]])
    left_delta = tf.squared_difference(pad, left)

    return concat_channels(below_delta, above_delta, right_delta, left_delta)


def slope_channel_difference(x):
    pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

    # below = tf.pad(x, [[0, 0], [0, 2], [1, 1], [0, 0]])
    # below_delta = tf.subtract(pad, below)

    above = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])
    above_delta = tf.subtract(pad, above)

    right = tf.pad(x, [[0, 0], [1, 1], [0, 2], [0, 0]])
    right_delta = tf.subtract(pad, right)

    # left = tf.pad(x, [[0, 0], [1, 1], [2, 0], [0, 0]])
    # left_delta = tf.subtract(pad, left)

    return concat_channels(
        tf.divide(above_delta, tf.add(right_delta, 0.1e-5)),
        above_delta,
        right_delta
    )


# def relative_channel_difference(x, num_channels=16):
#     x = convert_to_channels_first(x)
#
#
#
#     # rolls = []
#     # for i in range(num_channels):
#     #     x = tf.manip.roll(x, shift=1, axis=2)
#     #     rolls.append(x)
#     #
#     # print("Rolls size", len(rolls))
#     #
#     # differences = []
#     # for i, roll in enumerate(rolls):
#     #     for j, sub_roll in enumerate(rolls):
#     #         if i < j:
#     #             differences.append(
#     #                 tf.subtract(roll, sub_roll)
#     #             )
#     # x = concat_channels(
#     #     *differences,
#     #     channel_axis=1
#     # )
#
#     return convert_to_channels_last(x)

# def relative_channel_difference_map(x, num_channels=16):
#     return tf.map_fn(relative_channel_difference, x)
#
#
# def relative_channel_difference(x, num_channels=16):
#     x = convert_to_channels_first_map(x)
#     roll_i = x
#     roll_j = x
#     concat_result = None
#     for i in range(num_channels):
#         for j in range(num_channels):
#             if i < j:
#                 sub = tf.subtract(roll_i, roll_j)
#                 if concat_result is None:
#                     concat_result = sub
#                 else:
#                     concat_result = concat_channels(
#                         concat_result,
#                         sub,
#                         channel_axis=0
#                     )
#             roll_j = tf.manip.roll(roll_j, shift=1, axis=1)
#         roll_i = tf.manip.roll(roll_i, shift=1, axis=1)
#
#     concat_result = convert_to_channels_last_map(concat_result)
#     return concat_result


# def mapped_channel_difference(x, dimension=2):
#     num_channels = x.shape[dimension]
#     differences = []
#     for i in range(num_channels):
#         x_i = x[i]
#         for j in range(num_channels):
#             if i < j:
#                 differences.append(
#                     tf.subtract(x_i, x[j])
#                 )
#
#     return tf.stack(differences)

def mapped_channel_difference(x):
    num_channels = x.shape[0]
    differences = []
    for i in range(num_channels):
        x_i = x[i]
        for j in range(num_channels):
            if i < j:
                differences.append(
                    tf.subtract(x_i, x[j])
                )

    return tf.requirements(differences)


def relative_channel_difference(x):
    x = convert_to_channels_first(x)
    x = tf.map_fn(mapped_channel_difference, x)
    x = convert_to_channels_last(x)
    return x


def pool_1d(x, vertical=True, max_pool=True):
    if vertical:
        x = switch_xy(x)

    pool_dimensions = [1, 1, x.shape[2], 1]
    x = pool(x, pool_dimensions, max_pool=max_pool)
    return x


def pool_channel_position(x, vertical=True, max_pool=True):
    # x = pool_1d(x, vertical, max_pool)
    # x = convert_to_channels_first(x)

    # dimension_shape = x.shape[2]
    # positional_multiplier = tf.to_float(tf.range(start=dimension_shape, limit=0, delta=-1))
    # positional_multiplier = tf.reshape(
    #     positional_multiplier,
    #     shape=[positional_multiplier.shape[0], 1])
    #
    # x = x * positional_multiplier

    # x = tf.reduce_max(
    #     x,
    #     axis=2,
    #     keepdims=True)

    # x = tf.transpose(x, [0, 2, 3, 1])

    # x = tf.transpose(x, [0, 3, 2, 1])
    # x = tf.squeeze(x, [2, 3])
    # x = tf.reshape(x, [tf.shape(x)[0], 1, 1, tf.shape(x)[1] * tf.shape(x)[2] * tf.shape(x)[3]])

    return x


def conv_to_1d(x, previous_size, size=64):
    vertical = conv_act_bn(x, size, kernel_size=(1, previous_size))
    horizontal = conv_act_bn(x, size, kernel_size=(previous_size, 1))
    x = Concatenate()([vertical, horizontal])
    return x


mrbuilder.register_layer_builder(
    "conv_to_1d",
    lambda layer_options, layer_connection:
    conv_to_1d(layer_connection,
                layer_options("previous_size"),
                layer_options("size")))


def relative_channel_position(x, max_pool=False):
    # vertical = pool_channel_position(x, vertical=True, max_pool=max_pool)
    # horizontal = pool_channel_position(x, vertical=False, max_pool=max_pool)

    # vertical = tf.transpose(vertical, [0, 3, 2, 1])
    # vertical = relative_channel_difference(vertical)
    # vertical = tf.transpose(vertical, [0, 1, 3, 2])

    # horizontal = tf.transpose(horizontal, [0, 3, 2, 1])
    # horizontal = relative_channel_difference(horizontal)
    # horizontal = tf.transpose(horizontal, [0, 1, 3, 2])

    # horizontal = relative_channel_difference(horizontal)

    # x = concat_channels(horizontal, vertical, channel_axis=2)
    # # x = tf.stack([horizontal, vertical])
    # x = tf.reshape(x, [x.shape[0], 1, 1, x.shape[1]])
    return x
    # return horizontal
    # return horizontal


def convert_to_channels_first(x):
    return tf.transpose(x, [0, 3, 1, 2])


def convert_to_channels_first_map(x):
    return tf.transpose(x, [2, 0, 1])


def convert_to_channels_last(x):
    return tf.transpose(x, [0, 2, 3, 1])


def switch_xy(x):
    return tf.transpose(x, [0, 2, 1, 3])


def pool(x, size, strides=None, max_pool=True, padding='SAME'):
    if strides is None:
        strides = size

    if max_pool:
        return tf.nn.max_pool(
            x,
            ksize=size,
            strides=strides,
            padding=padding
        )
    else:
        return tf.nn.avg_pool(
            x,
            ksize=size,
            strides=strides,
            padding=padding
        )


def channel_wise_fc_layer(x):
    _, height, width, n_feat_map = x.get_shape().as_list()
    input_reshape = tf.reshape(x, [-1, width * height, n_feat_map])
    input_transpose = tf.transpose(input_reshape, [2, 0, 1])

    W = tf.get_variable(
        "W",
        shape=[n_feat_map, width * height, width * height],  # (512,49,49)
        initializer=tf.random_normal_initializer(0., 0.005))
    output = tf.matmul(input_transpose, W)

    output_transpose = tf.transpose(output, [1, 2, 0])
    output_reshape = tf.reshape(output_transpose, [-1, height, width, n_feat_map])

    return output_reshape


# def squash(vectors, axis=-1):
#     """
#     The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
#     :param vectors: some vectors to be squashed, N-dim tensor
#     :param axis: the axis to squash
#     :return: a Tensor with same shape as x vectors
#     """
#     s_squared_norm = sum(square(vectors), axis, keepdims=True)
#     scale = s_squared_norm / (1 + s_squared_norm) / sqrt(s_squared_norm + epsilon())
#     return scale * vectors


def squeeze_expand(x, squeeze=16, expand=64):
    x = Conv2D(squeeze, (1, 1), activation='relu', padding='valid')(x)
    left = Conv2D(expand, (1, 1), activation='relu', padding='valid')(x)
    right = Conv2D(expand, (3, 3), activation='relu', padding='same')(x)
    # return Concatenate(3)([left, right])
    return concatenate_channels([left, right])

mrbuilder.register_layer_builder(
    "squeeze_expand",
    lambda layer_options, layer_connection:
        squeeze_expand(
            layer_connection,
            layer_options("squeeze", 16),
            layer_options("expand", 64)))


def concatenate_channels(layers):
    return Concatenate(3)(layers)


mrbuilder.register_layer_builder(
    "concatenate_channels",
    lambda layer_options, layer_connection:
        concatenate_channels(layer_connection))


def upsample(x, size, type='deconv', activation='relu', weight_decay=1E-4):
    """ SubpixelConvolutional Upscaling (factor = 2)
    Args:
        x: keras tensor
        size: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    """
    """
    # Simpe Conv2DTranspose
    # Not good, compared to upsample + conv2d below.
    x= Conv2DTranspose( filters, shape, padding='same',
        strides=(2, 2), kernel_initializer=Args.kernel_initializer )(x)

    # simple and works
    #x = UpSampling2D( (2, 2) )( x )
    #x = Conv2D( filters, shape, padding='same' )( x )

    # Bilinear2x... Not sure if it is without bug, not tested yet.
    # Tend to make output blurry though
    #x = bilinear2x( x, filters )
    #x = Conv2D( filters, shape, padding='same' )( x )

    x = BatchNormalization(momentum=Args.bn_momentum)( x )
    x = LeakyReLU(alpha=Args.alpha_G)( x )
    """

    if type == 'upsampling':
        x = UpSampling2D()(x)
    elif type == 'subpixel':
        x = Conv2D(size, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_normal')(x)
        x = SubPixelUpscaling(scale_factor=2)(x)
        x = Conv2D(size, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_normal')(x)
    else:
        x = Conv2DTranspose(
            size, (3, 3),
            activation='relu',
            padding='same',
            strides=(2, 2),
            kernel_initializer='he_normal',
            kernel_regularizer=l2(weight_decay))(x)

    return x

mrbuilder.register_layer_builder(
    "upsample",
    lambda layer_options, layer_connection:
        upsample(
            layer_connection,
            layer_options("size"),
            type=layer_options("type", default_value="deconv"),
            weight_decay=layer_options("weight_decay", None)))


#  Transfer Utils ------------------------------------------------------------

def add_predict_layer(model, num_classes, num_fully_connected=None):
    x = model.output
    x = GlobalAveragePooling2D()(x)
    if num_fully_connected is not None:
        x = Dense(num_fully_connected, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x


def make_trainable(model, trainable=True):
    for layer in model.layers:
        layer.trainable = trainable


def freeze_all_layers(model):
    make_trainable(model, False)


def setup_for_transfer_learning(model, num_classes, num_fully_connected=1000):
    freeze_all_layers(model)
    return add_predict_layer(model, num_classes, num_fully_connected)


#  CAPSULE UTILS ------------------------------------------------------------


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as x vectors
    """
    s_squared_norm = sum(square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / sqrt(s_squared_norm + epsilon())
    return scale * vectors


class Length(layers.Layer):
    """
        Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
        inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
        output: shape=[dim_1, ..., dim_{n-1}]
    """

    def call(self, inputs, **kwargs):
        return sqrt(sum(square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
        Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
        Output shape: [None, d2]
    """

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - max(x, 1, True)) / epsilon() + 1
            mask = clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def primary_capsule(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    :param padding:
    :param strides:
    :param kernel_size:
    """
    output = layers.Conv2D(
        filters=dim_vector * n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)


def expand_conv(init, base, k, stride):
    shortcut = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    shortcut = Activation('relu')(shortcut)

    x = ZeroPadding2D((1, 1))(shortcut)
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='valid', kernel_initializer='he_normal',
                      use_bias=False)(x)

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal',
                      use_bias=False)(x)

    # Add shortcut

    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal',
                             use_bias=False)(shortcut)

    m = Add()([x, shortcut])

    return m


def conv_block(x, n, stride, k=1, dropout=0.0):
    init = x

    x = BatchNormalization(
        momentum=0.1,
        epsilon=1e-5,
        gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(
        n * k,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False)(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = BatchNormalization(
        momentum=0.1,
        epsilon=1e-5,
        gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(
        n * k,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False)(x)

    m = Add()([init, x])
    return m


def sample_random_frames(model_input, num_frames, num_samples):
    """Samples a random set of frames of size num_samples.
    Args:
      model_input: A tensor of size batch_size x max_frames x feature_size
      num_frames: A tensor of size batch_size x 1
      num_samples: A scalar
    Returns:
      `model_input`: A tensor of size batch_size x num_samples x feature_size
    """
    batch_size = tf.shape(model_input)[0]
    frame_index = tf.cast(
        tf.multiply(
            tf.random_uniform([batch_size, num_samples]),
            tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
    batch_index = tf.tile(
        tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
    index = tf.requirements([batch_index, frame_index], 2)
    return tf.gather_nd(model_input, index)


def frame_pooling(frames, method, **unused_params):
    """Pools over the frames of a video.
    Args:
      frames: A tensor with shape [batch_size, num_frames, feature_size].
      method: "average", "max", "attention", or "none".
    Returns:
      A tensor with shape [batch_size, feature_size] for average, max, or
      attention pooling. A tensor with shape [batch_size*num_frames, feature_size]
      for none pooling.
    Raises:
      ValueError: if method is other than "average", "max", "attention", or
      "none".
    """
    if method == "average":
        return tf.reduce_mean(frames, 1)
    elif method == "max":
        return tf.reduce_max(frames, 1)
    elif method == "none":
        feature_size = frames.shape_as_list()[2]
        return tf.reshape(frames, [-1, feature_size])
    else:
        raise ValueError("Unrecognized pooling method: %s" % method)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / tf.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return tf.math.acos(tf.clip_by_value(tf.tensordot(v1_u, v2_u, 1), -1.0, 1.0))
