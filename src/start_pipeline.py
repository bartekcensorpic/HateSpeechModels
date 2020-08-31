import os

from sklearn.metrics import classification_report
import numpy as np
from src.pipeline.get_data import get_dfs
from src.pipeline.get_models import (
    get_bert_tf2,
    get_glove_simple,
    get_glove_complex,
    get_bert_transformers,
    get_distilbert_transformers2_simple,
    get_distilbert_transformers_complex,
    get_mobilebert_transformers,
)
from src.evaluation.src.evaluations.binary_evaluation import (
    calculate_binary_classification_metrics,
)


def start(
    dataset_path,
    root_save_path,
    num_epochs,
    max_items=None,
    model_name="mobilebert_transformers",
    glove_txt_path=None,
):
    PREDICTION_THRESHOLD = 0.5
    train_X, train_y, test_X, test_y = get_dfs(
        dataset_path, separator=";", max_items=max_items, max_seq_length=512
    )

    model_map = {
        "glove_simple": get_glove_simple(root_save_path, glove_txt_path),
        "glove_complex": get_glove_complex(root_save_path, glove_txt_path),
        "bert_transformers": get_bert_transformers(root_save_path),
        "bert_tf2": get_bert_tf2(root_save_path),
        "distilbert_transformers_complex": get_distilbert_transformers_complex(
            root_save_path
        ),
        "distilbert_transformers2_simple": get_distilbert_transformers2_simple(
            root_save_path
        ),
        "mobilebert_transformers": get_mobilebert_transformers(root_save_path),
    }

    train_function, output_save_path = model_map[model_name]

    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)

    predictions, test_y = train_function(train_X, train_y, test_X, test_y, num_epochs)

    binary_evaluation_result = calculate_binary_classification_metrics(
        test_y, predictions, "hate_speech", PREDICTION_THRESHOLD
    )
    binary_evaluation_result.save_to_disk(output_save_path)

    # arbitrary 0.5 threshold
    predicted_classes = np.round(predictions).reshape(-1)
    report = classification_report(
        test_y, predicted_classes  # , target_names=['normal', 'hate_speech']
    )

    report_path = os.path.join(output_save_path, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write(str(report))
