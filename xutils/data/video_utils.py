import argparse
import cv2
import os


def to_images(video_path, images_path, skip_time=None, file_prefix="frame", file_postfix=".jpg", start_time=0):
    video = cv2.VideoCapture(video_path)
    if start_time > 0:
        video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    count = 0
    success = True
    while success:
        if skip_time:
            video.set(cv2.CAP_PROP_POS_MSEC, (count * skip_time))
        success, image = video.read()
        cv2.imwrite(os.path.join(images_path, "".join([file_prefix, str(count), file_postfix])), image)
        count = count + 1


def to_video(images_path, video_path, fps=29, file_prefix="frame", file_postfix=".jpg"):
    image_array = []
    files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and f.endswith('.jpg')]
    file_prefix_len = len(file_prefix)
    file_postfix_len = len(file_postfix)
    files.sort(key=lambda x: int(x[file_prefix_len:-file_postfix_len]))
    size = None
    for i in range(len(files)):
        img = cv2.imread(os.path.join(images_path, files[i]))
        size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(video_path, fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--to", default="i", help="v or i")
    a.add_argument("--video", help="path to video")
    a.add_argument("--images", help="path to images")
    args = a.parse_args()
    if args.to == 'i':
        to_images(args.video, args.images)
    else:
        to_video(args.images, args.video)
