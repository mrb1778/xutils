import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

def plot_series(data, title, x_label, y_label):
    ax = pd.DataFrame(data).plot(title=title, logy=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot()
    plt.show()


def plot_array(data, labels, title=None, pivot=False):
    print("data b4", data)
    if pivot:
        data = np.array(data).T
    print("data after", data)

    plt.subplots()
    for i in range(len(data)):
        print("data i", data[1], "label", labels[i])
        # plt.plot(data[i], label=labels[i])
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_from_map(data_map, title=None):
    # fig, ax = plt.subplots()
    for label, data in data_map.items():
        plt.plot(data, label=label)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_images(images, cols=10, title=None, process_fn=None):
    n_images = len(images)
    rows = n_images // cols

    plt.figure()
    for i in range(n_images):
        img = images[i]
        if process_fn is not None:
            img = process_fn(img)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.show()


def show_images(rows, columns, path):
    fig = plt.figure(figsize=(15, 15))
    files = os.listdir(path)
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(files))
        img = np.asarray(Image.open(os.path.join(path, files[index])))
        fig.add_subplot(rows, columns, i)
        plt.title(files[i], fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(img)
    plt.show()
