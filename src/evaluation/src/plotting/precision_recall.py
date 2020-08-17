from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_threshold_f1_plot(f1_scores: Union[np.ndarray,List], thresholds: Union[np.ndarray,List], title: str) -> plt.axes:
    """
    Draws plot for f1_score and thresholds

    :param f1_scores: np.ndarray of f1 scores values, 1 dimensional
    :param thresholds: np.ndarray of thresholds values, 1 dimensional
    :param title: a string to be displayed at the top of plot
    :return: matplotlib axis
    """
    fig, ax = plt.subplots()
    ax.plot(f1_scores, thresholds)

    ax.set_title(title)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Threshold")

    return ax


def plot_precision_recall_threshold_plot(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, title: str, n_thresholds: int = 10
) -> plt.axes:
    """
    Draws precision-recall plot with n_thresholds evenly spaced thresholds points

    :param precision: np.ndarray of precision values, 1 dimensional
    :param recall: np.ndarray of recall values, 1 dimensional
    :param thresholds: np.ndarray of thresholds values, 1 dimensional
    :param title: a string to be displayed at the top of plot
    :param n_thresholds: number of threshold points to be drawn at the plot
    :return: matplotlib axis
    """

    n = len(precision)
    step_width = int(n / n_thresholds)
    indexes = [int(step_width * idx) for idx in range(0, n_thresholds)]

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    texts = []
    coordinates_x = []
    coordinates_y = []
    for i in indexes:
        txt = thresholds[i]
        _x = recall[i]
        _y = precision[i]
        texts.append(ax.text(_x, _y, txt))
        coordinates_x.append(_x)
        coordinates_y.append(_y)
    adjust_text(texts)
    ax.scatter(coordinates_x, coordinates_y, c="r")

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    return ax
