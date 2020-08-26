import numpy as np
from src.evaluation.src.evaluations.binary_classification_result import (
    BinaryClassificationResult,
)
from sklearn.metrics import precision_recall_curve

from src.evaluation.src.plotting.calibration_curve import plot_calibration_curve
from src.evaluation.src.plotting.precision_recall import (
    plot_precision_recall_threshold_plot,
)
from src.evaluation.src.plotting.confusion_matrix import plot_confusion_matrix
from src.evaluation.src.metrics_calc.precision_recall_f1 import (
    calculate_f1_vs_thresholds_and_get_its_plots,
)


def __get_confusion_matrix(
    class_name, org_y, org_x, f1_scores, thresholds, initial_threshold=None
):
    """
    Calculates confusion matrix and selects best threshold if not provided

    :param class_name: class name
    :param org_y: 1d array of binary values (0-1)
    :param org_x: 1d array of predicted probabilities between 0 and 1
    :param f1_scores: calculated f1 scores
    :param thresholds: thresholds for which f1 scores were calculated
    :param initial_threshold: threshold to calculate confusion matrix, If none, best threshold with best f1 score is used
    :return: matplotlib.axis
    """

    if initial_threshold is None:
        best_f1_idx = np.argmax(f1_scores)
        selected_threshold = thresholds[best_f1_idx]
    else:
        selected_threshold = initial_threshold

    x = np.where(org_x >= selected_threshold, 1, 0)

    cf_axis = plot_confusion_matrix(org_y, x, class_name)

    return cf_axis


def calculate_binary_classification_metrics(
    y: np.ndarray, x: np.ndarray, class_name: str, prediction_threshold: float = None
) -> BinaryClassificationResult:
    """
    Calculates precision, recall, thresholds, F1 scores and plots:
        - precision, recall, thresholds plot

        - calibration curve plot

        - confusion matrix plot for best F1 score if `IOU_threshold` is None, for `IOU_threshold` otherwise

        - F1 vs threshold plot


    :param y: 1 dimensional list of true binary values (0 or 1)
    :param x: 1 dimensional list of predicted probabilities (between 0 and 1)
    :param class_name: class name of class
    :param prediction_threshold: used to calculate confusion matrix, if None then best F1 score threshold is used
    :return: BinaryClassificationResult
    """
    precision, recall, thresholds = precision_recall_curve(y, x)

    p_r_t_axis = plot_precision_recall_threshold_plot(
        precision,
        recall,
        thresholds,
        f"Precision-recall curve for class '{class_name}'",
    )

    cc_axis = plot_calibration_curve(y, x, class_name)

    f1s, ths, f1_ths_ax = calculate_f1_vs_thresholds_and_get_its_plots(
        class_name, y, x, thresholds
    )

    cf_axis = __get_confusion_matrix(
        class_name, y, x, f1s, ths, initial_threshold=prediction_threshold
    )

    return BinaryClassificationResult(
        precision, recall, thresholds, f1s, ths, p_r_t_axis, cc_axis, cf_axis, f1_ths_ax
    )
