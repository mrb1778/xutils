import cv2
from PIL import Image
import numpy as np

from gradcam.gradcam import *  # https://github.com/jacobgil/pytorch-grad-cam


def do_grad_cam(path):
    """Gradient-weighted Class Activation Mapping, or more simply Grad-CAM, helps us get what the network is seeing,
    and helps us see which neurons are firing in a particular layer given the image as input."""

    # Initialise the grad cam object.
    # we use model.features as the feature extractor and use the layer no. 35 for gradients.
    grad_cam = GradCam(model=model, feature_module=model.features, \
                       target_layer_names=["35"], use_cuda=True)

    # read in the image, and prepare it for the network
    orig_im = cv2.imread(path)
    img = Image.fromarray(orig_im)
    inp = val_transformer(img).unsqueeze(0)

    # main inference
    mask = grad_cam(inp, None)

    # create the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    #add the heatmap to the original image
    cam = heatmap + np.float32(cv2.resize(orig_im, (224,224))/255.)
    cam = cam / np.max(cam)

    # BGR -> RGB since OpenCV operates with BGR values.
    cam = cam[:,:,::-1]

    return cam