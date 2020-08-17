from evaluation.examples.detection_data_mock import get_detection_data_mock
from evaluation.examples.gt_data_mocks import get_ground_truth_data_mock
from evaluation.src.evaluations.binary_evaluation import calculate_binary_classification_metrics
import matplotlib.pyplot as plt

"""
Example how to use all these clunky functions, lol
"""


if __name__ == '__main__':

    #select folders
    GT_FOLDER = r"C:\Users\barte\Desktop\trash\gt_folder" #folder for ground truths .txt files
    DET_FOLDER = r"C:\Users\barte\Desktop\trash\detections_folder" #folder for detections .txt files
    OUTPUT_FOLDER = r"C:\Users\barte\Desktop\trash\evaluator_results" #folder for outputs
    IOU_THRESHOLD = 0.3
    PREDICTION_THRESHOLD = 0.5

    #get data lists
    #IMPORTANT have a look into this function to see what is the required data format
    ground_truth_data_list = get_ground_truth_data_mock(GT_FOLDER)
    my_models_predictions_list = get_detection_data_mock(DET_FOLDER)

    evaluator = ObjectDetectionEvaluator(ground_truth_folder=GT_FOLDER,
                                         detections_folder= DET_FOLDER,
                                         ground_truth_generator=ground_truth_data_list,
                                         detections_generator= my_models_predictions_list)

    # metrics: list of dictionaries per class
    # maP: float
    # X_all: np.array of scores of all predictions, regardless of their class
    # Y_all: np.array of binary labels (0 or 1) where all classes are combined
    class_metrics, mAP, X_all, y_all =  evaluator.calculate_mAP(IOU_THRESHOLD, OUTPUT_FOLDER)

    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('total mAP:',mAP_str)

    #unpack class_metrics and shit
    for class_data in class_metrics:
        cl = class_data["class"]
        ap = class_data["AP"]
        precision = class_data["precision"]
        recall = class_data["recall"]
        totalPositives = class_data["total positives"]
        total_TP = class_data["total TP"]
        total_FP = class_data["total FP"]
        X = class_data['X']
        y = class_data['Y']

        if totalPositives > 0:
            prec = ["%.2f" % p for p in precision]
            rec = ["%.2f" % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('#### Class:', cl)
            print("AP: %s (%s)" % (ap_str, cl))
            print("\nAP: %s" % ap_str)
            print("\nPrecision: %s" % prec)
            print("\nRecall: %s" % rec)

        #now you can get all the metrics for each class in more details
        #Important, because mocked data is limited (40 predictions, mostly the same) plots look funky
        binary_evaluation_result = calculate_binary_classification_metrics(y, X, cl ,PREDICTION_THRESHOLD)

        #precision vs recall for each threshold
        (precision_full, recall_full, thresholds_full) = binary_evaluation_result.get_precision_recall_thresholds()

        #show plot for precision, recall, thresholds
        p_r_t_p_plot = binary_evaluation_result.get_precision_recall_threshold_plot()
        p_r_t_p_plot.get_figure().show()


        #show calibration curve
        cal_curve = binary_evaluation_result.get_calibration_curve()
        cal_curve.get_figure().show()

        #show confusion matrix
        cm = binary_evaluation_result.get_confusion_matrix()
        cm.get_figure().show()

        #show f1 vs threshold plot
        f1_values, threshold_for_f1_values, f1_ths_ax = binary_evaluation_result.get_f1_thresholds_and_plot()
        f1_ths_ax.get_figure().show()













