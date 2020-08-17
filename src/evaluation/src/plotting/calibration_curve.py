import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def plot_calibration_curve(y: np.ndarray, x: np.ndarray, class_name: str) -> plt.axes:
    """
    Return calibration curve https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html

    :param y: np.ndarray of true labels 1D
    :param x: np.ndarray of predicted probabilities 1D
    :param class_name: class name
    :return: return matplotlib axes
    """

    fop, mpv = calibration_curve(y, x, n_bins=10, normalize=True)

    fig, ax = plt.subplots()
    # plot perfectly calibrated
    ax.plot([0, 1], [0, 1], linestyle="--")
    # plot model reliability
    ax.plot(mpv, fop, marker=".")
    ax.set_title(f"calibration curve for class: {class_name}")
    ax.set_xlabel("mean predicted value")
    ax.set_ylabel("true fraction of positive cases in each bin")

    return ax
