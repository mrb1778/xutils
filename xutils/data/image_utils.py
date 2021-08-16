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


def load(name='x', directory='x', extension='jpg'):
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

        # Our layer receives two inputs. We need to crop the first x blob
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


def reshape_as_images(x, img_height, img_width=None):
    if img_width is None:
        img_width = img_height

    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp


def random_crop(img, w, h):
    height, width = img.shape[:2]

    h_rnd = height - h
    w_rnd = width - w

    y = np.random.randint(0, h_rnd) if h_rnd > 0 else 0
    x = np.random.randint(0, w_rnd) if w_rnd > 0 else 0

    return img[y:y + height, x:x + width]


def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[..., np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat(img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[..., 0:target_channels]

    return img


def cut_odd_image(img):
    h, w, c = img.shape
    wm, hm = w % 2, h % 2
    if wm + hm != 0:
        img = img[0:h - hm, 0:w - wm, :]
    return img


def overlay_alpha_image(img_target, img_source, xy_offset=(0, 0)):
    (h, w, c) = img_source.shape
    if c != 4:
        raise ValueError("overlay_alpha_image, img_source must have 4 channels")

    x1, x2 = xy_offset[0], xy_offset[0] + w
    y1, y2 = xy_offset[1], xy_offset[1] + h

    alpha_s = img_source[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img_target[y1:y2, x1:x2, c] = (alpha_s * img_source[:, :, c] +
                                       alpha_l * img_target[y1:y2, x1:x2, c])


def apply_random_rgb_levels(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    np_rnd = rnd_state.rand

    inBlack = np.array([np_rnd() * 0.25, np_rnd() * 0.25, np_rnd() * 0.25], dtype=np.float32)
    inWhite = np.array([1.0 - np_rnd() * 0.25, 1.0 - np_rnd() * 0.25, 1.0 - np_rnd() * 0.25], dtype=np.float32)
    inGamma = np.array([0.5 + np_rnd(), 0.5 + np_rnd(), 0.5 + np_rnd()], dtype=np.float32)

    outBlack = np.array([np_rnd() * 0.25, np_rnd() * 0.25, np_rnd() * 0.25], dtype=np.float32)
    outWhite = np.array([1.0 - np_rnd() * 0.25, 1.0 - np_rnd() * 0.25, 1.0 - np_rnd() * 0.25], dtype=np.float32)

    result = np.clip((img - inBlack) / (inWhite - inBlack), 0, 1)
    result = (result ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    result = np.clip(result, 0, 1)

    if mask is not None:
        result = img * (1 - mask) + result * mask

    return result


def apply_random_hsv_shift(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h = (h + rnd_state.randint(360)) % 360
    s = np.clip(s + rnd_state.random() - 0.5, 0, 1)
    v = np.clip(v + rnd_state.random() - 0.5, 0, 1)

    result = np.clip(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR), 0, 1)
    if mask is not None:
        result = img * (1 - mask) + result * mask

    return result


def apply_random_sharpen(img, chance, kernel_max_size, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    sharp_rnd_kernel = rnd_state.randint(kernel_max_size) + 1

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        if rnd_state.randint(2) == 0:
            result = blur_sharpen(result, 1, sharp_rnd_kernel, rnd_state.randint(10))
        else:
            result = blur_sharpen(result, 2, sharp_rnd_kernel, rnd_state.randint(50))

        if mask is not None:
            result = img * (1 - mask) + result * mask

    return result


def apply_random_motion_blur(img, chance, mb_max_size, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    mblur_rnd_kernel = rnd_state.randint(mb_max_size) + 1
    mblur_rnd_deg = rnd_state.randint(360)

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        result = LinearMotionBlur(result, mblur_rnd_kernel, mblur_rnd_deg)
        if mask is not None:
            result = img * (1 - mask) + result * mask

    return result


def apply_random_gaussian_blur(img, chance, kernel_max_size, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        gblur_rnd_kernel = rnd_state.randint(kernel_max_size) * 2 + 1
        result = cv2.GaussianBlur(result, (gblur_rnd_kernel,) * 2, 0)
        if mask is not None:
            result = img * (1 - mask) + result * mask

    return result


def apply_random_resize(img, chance, max_size_per, interpolation=cv2.INTER_LINEAR, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h, w, c = result.shape

        trg = rnd_state.rand()
        rw = w - int(trg * int(w * (max_size_per / 100.0)))
        rh = h - int(trg * int(h * (max_size_per / 100.0)))

        result = cv2.resize(result, (rw, rh), interpolation=interpolation)
        result = cv2.resize(result, (w, h), interpolation=interpolation)
        if mask is not None:
            result = img * (1 - mask) + result * mask

    return result


def apply_random_nearest_resize(img, chance, max_size_per, mask=None, rnd_state=None):
    return apply_random_resize(img, chance, max_size_per, interpolation=cv2.INTER_NEAREST, mask=mask,
                               rnd_state=rnd_state)


def apply_random_bilinear_resize(img, chance, max_size_per, mask=None, rnd_state=None):
    return apply_random_resize(img, chance, max_size_per, interpolation=cv2.INTER_LINEAR, mask=mask,
                               rnd_state=rnd_state)


def apply_random_jpeg_compress(img, chance, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h, w, c = result.shape

        quality = rnd_state.randint(10, 101)

        ret, result = cv2.imencode('.jpg', np.clip(img * 255, 0, 255).astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ret:
            result = cv2.imdecode(result, flags=cv2.IMREAD_UNCHANGED)
            result = result.astype(np.float32) / 255.0
            if mask is not None:
                result = img * (1 - mask) + result * mask

    return result


def apply_random_overlay_triangle(img, max_alpha, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    h, w, c = img.shape
    pt1 = [rnd_state.randint(w), rnd_state.randint(h)]
    pt2 = [rnd_state.randint(w), rnd_state.randint(h)]
    pt3 = [rnd_state.randint(w), rnd_state.randint(h)]

    alpha = rnd_state.uniform() * max_alpha

    tri_mask = cv2.fillPoly(np.zeros_like(img), [np.array([pt1, pt2, pt3], np.int32)], (alpha,) * c)

    if rnd_state.randint(2) == 0:
        result = np.clip(img + tri_mask, 0, 1)
    else:
        result = np.clip(img - tri_mask, 0, 1)

    if mask is not None:
        result = img * (1 - mask) + result * mask

    return result


def _min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    return cv2.resize(x, (s1, s0), interpolation=cv2.INTER_LANCZOS4)


def _d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def _get_image_gradient(dist):
    cols = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]))
    rows = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]))
    return cols, rows


def _generate_lighting_effects(content):
    h512 = content
    h256 = cv2.pyrDown(h512)
    h128 = cv2.pyrDown(h256)
    h64 = cv2.pyrDown(h128)
    h32 = cv2.pyrDown(h64)
    h16 = cv2.pyrDown(h32)
    c512, r512 = _get_image_gradient(h512)
    c256, r256 = _get_image_gradient(h256)
    c128, r128 = _get_image_gradient(h128)
    c64, r64 = _get_image_gradient(h64)
    c32, r32 = _get_image_gradient(h32)
    c16, r16 = _get_image_gradient(h16)
    c = c16
    c = _d_resize(cv2.pyrUp(c), c32.shape) * 4.0 + c32
    c = _d_resize(cv2.pyrUp(c), c64.shape) * 4.0 + c64
    c = _d_resize(cv2.pyrUp(c), c128.shape) * 4.0 + c128
    c = _d_resize(cv2.pyrUp(c), c256.shape) * 4.0 + c256
    c = _d_resize(cv2.pyrUp(c), c512.shape) * 4.0 + c512
    r = r16
    r = _d_resize(cv2.pyrUp(r), r32.shape) * 4.0 + r32
    r = _d_resize(cv2.pyrUp(r), r64.shape) * 4.0 + r64
    r = _d_resize(cv2.pyrUp(r), r128.shape) * 4.0 + r128
    r = _d_resize(cv2.pyrUp(r), r256.shape) * 4.0 + r256
    r = _d_resize(cv2.pyrUp(r), r512.shape) * 4.0 + r512
    coarse_effect_cols = c
    coarse_effect_rows = r
    eps = 1e-10

    max_effect = np.max((coarse_effect_cols ** 2 + coarse_effect_rows ** 2) ** 0.5,
                        axis=0,
                        keepdims=True, ).max(1, keepdims=True)
    coarse_effect_cols = (coarse_effect_cols + eps) / (max_effect + eps)
    coarse_effect_rows = (coarse_effect_rows + eps) / (max_effect + eps)

    return np.stack([np.zeros_like(coarse_effect_rows), coarse_effect_rows, coarse_effect_cols], axis=-1)


def apply_random_relight(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    def_img = img

    if rnd_state.randint(2) == 0:
        light_pos_y = 1.0 if rnd_state.randint(2) == 0 else -1.0
        light_pos_x = rnd_state.uniform() * 2 - 1.0
    else:
        light_pos_y = rnd_state.uniform() * 2 - 1.0
        light_pos_x = 1.0 if rnd_state.randint(2) == 0 else -1.0

    light_source_height = 0.3 * rnd_state.uniform() * 0.7
    light_intensity = 1.0 + rnd_state.uniform()
    ambient_intensity = 0.5

    light_source_location = np.array([[[light_source_height, light_pos_y, light_pos_x]]], dtype=np.float32)
    light_source_direction = light_source_location / np.sqrt(np.sum(np.square(light_source_location)))

    lighting_effect = _generate_lighting_effects(img)
    lighting_effect = np.sum(lighting_effect * light_source_direction, axis=-1).clip(0, 1)
    lighting_effect = np.mean(lighting_effect, axis=-1, keepdims=True)

    result = def_img * (ambient_intensity + lighting_effect * light_intensity)  # light_source_color
    result = np.clip(result, 0, 1)

    if mask is not None:
        result = def_img * (1 - mask) + result * mask

    return result


def LinearMotionBlur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k)


def blur_sharpen(img, sharpen_mode=0, kernel_size=3, amount=100):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if amount > 0:
        if sharpen_mode == 1:  # box
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, kernel_size // 2] = 1.0
            box_filter = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
            kernel = kernel + (kernel - box_filter) * amount
            return cv2.filter2D(img, -1, kernel)
        elif sharpen_mode == 2:  # gaussian
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)
            return img
    elif amount < 0:
        n = -amount
        while n > 0:

            img_blur = cv2.medianBlur(img, 5)
            if int(n / 10) != 0:
                img = img_blur
            else:
                pass_power = (n % 10) / 10.0
                img = img * (1.0 - pass_power) + img_blur * pass_power
            n = max(n - 10, 0)

        return img
    return img


def draw_polygon(image, points, color, thickness=1):
    points_len = len(points)
    for i in range(0, points_len):
        p0 = tuple(points[i])
        p1 = tuple(points[(i + 1) % points_len])
        cv2.line(image, p0, p1, color, thickness=thickness)


def draw_rect(image, rect, color, thickness=1):
    l, t, r, b = rect
    draw_polygon(image, [(l, t), (r, t), (r, b), (l, b)], color, thickness)
