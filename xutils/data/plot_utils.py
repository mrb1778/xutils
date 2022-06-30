import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns
import random 

from xutils.dl.pytorch.grad_cam import do_grad_cam


def plot(data, title, x_label, y_label):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


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


def plot_images_directory(rows, columns, path):
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


def plot_confusion_matrix(data, positive_label="Positive", negative_label="Negative"):
    ax = plt.subplot()
    sns.heatmap(data, annot=True, ax=ax, cmap='Blues')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels([positive_label, negative_label])
    ax.yaxis.set_ticklabels([positive_label, negative_label])


def plot_mistakes(targets, predictions, scores, show_examples=True, image_paths=None):
    targets = np.array(targets)
    predictions = np.array(predictions)
    scores = np.array(scores)

    misclassified_indexes = np.nonzero(targets != predictions)
    misclassified_scores = scores[misclassified_indexes[0]]

    # plot the historgram of misclassified scores
    plt.hist(misclassified_scores)
    plt.xlabel("scores")
    plt.ylabel("No. of examples")
    plt.show()

    if show_examples:
        def _plot(title, items):
            indices = np.nonzero(items)[0]
            images = [image_paths[i] for i in random.sample(indices, 10)]
            plot_images(images, cols=10, title=title, process_fn=do_grad_cam)

        true_positives = np.logical_and(predictions == 1, targets == 1)
        true_positives = np.logical_and(true_positives, scores > 0.9)
        _plot("True Positive", true_positives)

        false_positives = np.logical_and(predictions == 1, targets == 0)
        false_positives = np.logical_and(false_positives, scores > 0.9)
        _plot("False Positive", false_positives)

        true_negatives = np.logical_and(predictions == 0, targets == 0)
        _plot("True Negatives", true_negatives)

        false_negatives = np.logical_and(predictions == 0, targets == 1)
        false_negatives = np.logical_and(false_negatives, scores < 0.1)
        _plot("False Negatives", false_negatives)
