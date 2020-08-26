from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join


def save_ax(ax, save_path):
    ax.get_figure().savefig(save_path)


def create_folder(paths: List):
    folder_path = os.path.join(*paths)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


class BinaryClassificationResult:
    def __init__(
        self,
        precision,
        recall,
        thresholds,
        f1s,
        ths,
        p_r_t_axis,
        cc_axis,
        cf_axis,
        f1_ths_ax,
    ):

        self._cf_axis = cf_axis
        self._cc_axis = cc_axis
        self._p_r_t_axis = p_r_t_axis
        self._f1_ths_ax = f1_ths_ax
        self._ths = ths
        self._f1s = f1s
        self._thresholds = thresholds
        self._recall = recall
        self._precision = precision

    def get_precision_recall_thresholds(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns precision, recall and thresholds

        :return: Tuple of 3 list: precision, recall, thresholds
        """
        return (self._precision, self._recall, self._thresholds)

    def get_precision_recall_threshold_plot(self) -> plt.axis:
        """
        Returns plot of precision, recall and thresholds

        :return: matplotlib.axis
        """
        return self._p_r_t_axis

    def get_calibration_curve(self) -> plt.axis:
        """
        Returns calibration curve

        :return: matpllib.axis
        """

        return self._cc_axis

    def get_confusion_matrix(self) -> plt.axis:
        """
        Returns confusion matrix

        :return: matpllib.axis
        """

        return self._cf_axis

    def get_f1_thresholds_and_plot(self) -> Tuple[np.ndarray, np.ndarray, plt.axis]:
        """
        Return plot of F1 scores vs Thresholds

        :return: Tuple of
        """
        return self._f1s, self._ths, self._f1_ths_ax

    def save_to_disk(self, root_save_folder):

        try:
            full_results_path = create_folder([root_save_folder, "evaluation"])
            (
                precision_full,
                recall_full,
                thresholds_full,
            ) = self.get_precision_recall_thresholds()

            # https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
            np.savetxt(join(full_results_path, "precision_full.txt"), precision_full)
            np.savetxt(join(full_results_path, "recall_full.txt"), recall_full)
            np.savetxt(join(full_results_path, "thresholds_full.txt"), thresholds_full)

            p_r_t_p_plot = self.get_precision_recall_threshold_plot()
            save_ax(
                p_r_t_p_plot,
                join(full_results_path, "full_precision_recall_threshold_plot.jpg"),
            )

            cm = self.get_confusion_matrix()
            save_ax(cm, join(full_results_path, "confusion_matrix.jpg"))

            cc = self.get_calibration_curve()
            save_ax(cc, join(full_results_path, "calibration_curve.jpg"))

            (
                f1_values,
                threshold_for_f1_values,
                f1_ths_ax,
            ) = self.get_f1_thresholds_and_plot()
            np.savetxt(join(full_results_path, "f1_values.txt"), f1_values)
            np.savetxt(
                join(full_results_path, "threshold_for_f1_values.txt"),
                threshold_for_f1_values,
            )
            save_ax(f1_ths_ax, join(full_results_path, "f1_thresholds.jpg"))

            return True, "OK"

        except Exception as e:
            debug = 5
            print(e)
            return False, e
