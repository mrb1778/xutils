import numpy as np

from xutils import layer_pool, layer_conv, layer_relu


def convert_net(weights, input_image, pooling, layers):
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]

            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)
            current = layer_conv(current, kernels, bias)
        elif kind == 'relu':
            current = layer_relu(current)
        elif kind == 'pool':
            current = layer_pool(current, pooling)
        net[name] = current

    assert len(net) == len(layers)
    return net
