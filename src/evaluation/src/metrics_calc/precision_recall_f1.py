from typing import Tuple, List
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from src.evaluation.src.plotting.calibration_curve import plot_calibration_curve
from src.evaluation.src.plotting.precision_recall import (
    plot_precision_recall_threshold_plot,
    plot_threshold_f1_plot,
)
import matplotlib.pyplot as plt


def calculate_precision_recall_and_get_its_plots(
    class_name: str, Y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.axes, plt.axes]:
    """
    Calclates precision-recall values and gets plots for precision-recall plot and calibration-curve

    :param class_name: 
    :param Y: True values 1 dimensional
    :param X: Predicted probabilities 1 dimension
    :return: returns tuple of (precision, recall, thresholds, precision_recall_axis, calibration_curve_axis)
    """

    precision, recall, thresholds = precision_recall_curve(Y, X)
    precision_recall_axis = plot_precision_recall_threshold_plot(
        precision, recall, thresholds, f"Precision-recall curve for class {class_name}"
    )
    calibration_curve_axis = plot_calibration_curve(Y, X, class_name)

    return precision, recall, thresholds, precision_recall_axis, calibration_curve_axis


def calculate_f1_vs_thresholds(
    Y: np.ndarray, X: np.ndarray, thresholds: np.ndarray, n_thresholds: int = 15
) -> Tuple[List, List]:
    r"""
    Calculates F1 score for given threshold values

    :param Y: True values 1 dimensional
    :param X: Predicted probabilities 1 dimension
    :param thresholds: np.ndarray of thresholds 1 dimensional
    :param n_thresholds: number of equal steps to calculate F1 vs number of thresholds
    :return: tuple of np.ndarray of F1 scores and their respective thresholds that's length is equal to `n_thresholds`
    """
    n = len(thresholds)
    step_width = int(n / n_thresholds)
    indexes = [int(step_width * idx) for idx in range(0, n_thresholds)]
    Y = np.asarray(Y)
    f1s = []
    ths = []
    for i in indexes:
        th = thresholds[i]
        x_rounded = np.where(X >= th, 1, 0)
        f1 = f1_score(Y, x_rounded)
        f1s.append(f1)
        ths.append(th)

    return f1s, ths


def calculate_f1_vs_thresholds_and_get_its_plots(
    class_name: str, Y: np.ndarray, X: np.ndarray, thresholds: np.ndarray, n_thresholds: int = 15
) -> Tuple[List, List, plt.axes]:
    """
     Calculates F1 score for given threshold values and returns plot of it

    :param class_name: class name
    :param Y: True values 1 dimensional
    :param X: Predicted probabilities 1 dimension
    :param thresholds: np.ndarray of thresholds 1 dimensional
    :param n_thresholds: number of equal steps to calculate F1 vs number of thresholds
    :return: Tuple of np.ndarray of F1 scores and their respective thresholds that's length is equal to `n_thresholds`; and matplotlib ax
    """
    f1s, ths = calculate_f1_vs_thresholds(Y, X, thresholds, n_thresholds)
    ax = plot_threshold_f1_plot(
        f1s, ths, f"Threshold-F1 score curve for class {class_name}"
    )

    return f1s, ths, ax
