from pathlib import Path

import cv2
import dlib
import requests
from io import BytesIO

import numpy as np
import scipy.misc
# import PIL
import os

# from deep dream
# from PIL import Image
from PIL import Image
# from imutils import video
from tqdm import tqdm


def load(name='input', directory='input', extension='jpg'):
    path = os.path.join(directory, name + "." + extension)
    # return np.float32(Image.open(path))
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:  # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:  # PNG with alpha channel
        img = img[:, :, :3]
    return img


def convert_255_save(image, name="test", directory="output", fmt='jpg'):
    image = convert_255(image)
    save(image, name, directory, fmt)


def clip_255_save(image, name="test", directory="output", fmt='jpg'):
    image = image_clip_255(image)
    save(image, name, directory, fmt)


def save(image, name="test", directory="output", fmt='jpg'):
    # Image.fromarray(image).save(os.path.join(directory, name + "." + fmt), fmt)
    scipy.misc.imsave(os.path.join(directory, name + "." + fmt), arr=image)


def show(data, type='RGB'):
    img = Image.fromarray(data, type)
    img.show()
    return img


def convert_255(image):
    return np.uint8(np.clip(image, 0, 1) * 255)


def normalize_to_mean_std(image, s=0.1):  # renamed from visstd
    """Normalize the image range for visualization"""
    return (image - image.mean()) / max(image.std(), 1e-4) * s + 0.5


def normalize_to_1_1(image):
    return (image / 127.5) - 1


def resize(image, size):
    return scipy.misc.imresize(image, size)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # return the warped image
    return warped


# from style transfer


def save_img(out_path, image):
    image = image_clip_255(image)
    scipy.misc.imsave(out_path, image)


def image_clip_255(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# def scale_img(style_path, style_scale):
#     scale = float(style_scale)
#     o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
#     new_shape = (int(o0 * scale), int(o1 * scale), o2)
#     style_target = get_img(style_path, img_size=new_shape)
#     return style_target
#
#
# def get_img(src, img_size=False):
#     img = scipy.misc.imread(src, mode='RGB')
#     if not (len(img.shape) == 3 and img.shape[2] == 3):
#         img = np.dstack((img, img, img))
#     if img_size:
#         img = resize(img, img_size)
#     return img


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


# todo: remove in favor of create_random_images
def generate_noise(sample_size, z_dim):
    """generate noise vector.

    Args:
        sample_size: sample/batch size
        z_dim: dimensionality of z noise

    Returns:
        random noise, dimensionality is (sample_size, z_dim)
    """
    return np.random.uniform(-1, 1,
                             size=(sample_size, z_dim)).astype(np.float32)


# todo: move to tf.random.normal / data util
def create_random_images(num, size):
    return np.random.normal(loc=0, scale=1, size=(num, size))


def get_gaussian_map():
    gaussian_map = np.zeros((368, 368), dtype='float32')
    for x_p in range(368):
        for y_p in range(368):
            dist_sq = (x_p - 368 / 2) * (x_p - 368 / 2) + \
                      (y_p - 368 / 2) * (y_p - 368 / 2)
            exponent = dist_sq / 2.0 / (21 ** 2)
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map.reshape((1, 368, 368, 1))


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def dodge(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def burn(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


def grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def down_up_sample(image, num_down_samples=2, num_bilateral_filters=5):
    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    sampled_image = image
    for _ in range(num_down_samples):
        sampled_image = cv2.pyrDown(sampled_image)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(num_bilateral_filters):
        sampled_image = cv2.bilateralFilter(sampled_image, 9, 9, 7)

    # upsample image to original size
    for _ in range(num_down_samples):
        sampled_image = cv2.pyrUp(sampled_image)

    # make sure resulting image has the same dims as original
    # print(image.shape[:2])
    return cv2.resize(sampled_image, (image.shape[1], image.shape[0]))


def cartoonize(image, num_down_samples=2, num_bilateral_filters=5):
    img_color = down_up_sample(image, num_down_samples, num_bilateral_filters)

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)


def create_hed_network(prototxt_path, caffemodel_path):
    class CropLayer(object):
        def __init__(self):
            self.x_start = 0
            self.x_end = 0
            self.y_start = 0
            self.y_end = 0

        # Our layer receives two inputs. We need to crop the first input blob
        # to match a shape of the second one (keeping batch size and number of channels)
        def getMemoryShapes(self, inputs):
            input_shape, target_shape = inputs[0], inputs[1]
            batch_size, num_channels = input_shape[0], input_shape[1]
            height, width = target_shape[2], target_shape[3]

            self.y_start = (input_shape[2] - target_shape[2]) // 2
            self.x_start = (input_shape[3] - target_shape[3]) // 2
            self.y_end = self.y_start + height
            self.x_end = self.x_start + width

            return [[batch_size, num_channels, height, width]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.y_start:self.y_end, self.x_start:self.x_end]]

    cv2.dnn_registerLayer('Crop', CropLayer)

    # Load the model.
    return cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


def get_edges_hed(image, hed_net):
    input_blob = cv2.dnn.blobFromImage(
        image,
        swapRB=False)
    hed_net.setInput(input_blob)
    out = hed_net.forward()
    out = out[0, 0]
    out = 255 * out
    generated_image = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return generated_image


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def get_face_predictor_detector(face_landmark_shape_file):
    return dlib.shape_predictor(face_landmark_shape_file), \
           dlib.get_frontal_face_detector()


def outline_face(predictor, detector, image, down_sample_ratio=1.):
    frame_resize = cv2.resize(image, None, fx=1 / down_sample_ratio, fy=1 / down_sample_ratio)
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    black_image = np.zeros(image.shape, np.uint8)

    # Perform if there is a face detected
    if len(faces) == 1:
        for face in faces:
            detected_landmarks = predictor(gray, face).parts()
            landmarks = [[p.x * down_sample_ratio, p.y * down_sample_ratio] for p in detected_landmarks]

            jaw = reshape_for_polyline(landmarks[0:17])
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            lower_nose = reshape_for_polyline(landmarks[30:35])
            left_eye = reshape_for_polyline(landmarks[42:48])
            right_eye = reshape_for_polyline(landmarks[36:42])
            outer_lip = reshape_for_polyline(landmarks[48:60])
            inner_lip = reshape_for_polyline(landmarks[60:68])

            color = (255, 255, 255)
            thickness = 3

            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)


def process_image(input_path,
                  mode="canny",
                  mode_type="wide",
                  show_image=False,
                  out_single_path=None,
                  out_bi_path=None,
                  **kwargs):
    image = cv2.imread(input_path)

    if mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if mode_type == "wide":
            generated_image = cv2.Canny(blurred, 10, 200)
        elif mode_type == "tight":
            generated_image = cv2.Canny(blurred, 225, 250)
        else:
            sigma = kwargs["sigma"] if kwargs and "sigma" in kwargs else 0.33

            v = np.median(blurred)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            generated_image = cv2.Canny(image, lower, upper)

        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_GRAY2RGB)
    elif mode == "laplacian":
        generated_image = cv2.Laplacian(image, cv2.CV_64F)
    elif mode == "cartoon":
        generated_image = cartoonize(image)
    elif mode == "cartoon-outline":
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        generated_image = down_up_sample(img_gray, num_down_samples=1, num_bilateral_filters=8)
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_GRAY2RGB)
    elif mode == "draw":
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        generated_image = cv2.divide(img_gray, img_blur, scale=256)
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_GRAY2RGB)
    elif mode == "threshold-gaussian":
        image_process = cv2.imread(input_path, cv2.CV_8UC1)
        generated_image = cv2.adaptiveThreshold(image_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                7, 10)
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_GRAY2RGB)
    elif mode == "hed":
        generated_image = get_edges_hed(image, kwargs["hed_net"])
    else:
        raise TypeError("Mode Not Supported: ", mode)

    if show_image:
        cv2.imshow(input_path, np.hstack([generated_image, image]))
        cv2.waitKey(0)

    if out_single_path is not None:
        cv2.imwrite(out_single_path, generated_image)

    if out_bi_path is not None:
        cv2.imwrite(str(out_bi_path), np.hstack([generated_image, image]))

    return generated_image


def process_batch_images(input_path,
                         mode="canny",
                         mode_type="wide",
                         show_image=False,
                         out_single_path=None,
                         out_bi_path=None,
                         **kwargs):
    if kwargs is None:
        kwargs = {}

    if mode == "hed":
        kwargs["hed_net"] = create_hed_network(kwargs["prototxt_path"], kwargs["caffemodel_path"])

    path = Path(input_path)
    if path.is_dir():
        for file_name in tqdm(path.glob("**/*")):
            # print(file_name)
            process_image(str(file_name),
                          mode,
                          mode_type,
                          show_image,
                          out_single_path=str(Path(out_single_path, file_name.name)) if out_single_path else None,
                          out_bi_path=str(Path(out_bi_path, file_name.name)) if out_bi_path else None,
                          **kwargs)
    # elif input_path.endswith(".mp4"):
    #     cap = cv2.VideoCapture(input_path)
    #
    #     fps = video.FPS().start()
    #
    #     count = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()


def save_array_as_images(x, img_width, img_height, path, file_names):
    os.makedirs(path)
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
        img = Image.fromarray(x_temp[i], 'RGB')
        img.save(os.path.join(path, str(file_names[i]) + '.png'))
    return x_temp


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp
