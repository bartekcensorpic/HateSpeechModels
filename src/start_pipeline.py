from sklearn.metrics import classification_report

from src.pipeline.get_data import get_dfs
from src.pipeline.get_models import (get_bert_tf2,
                                     get_glove,
                                     get_bert_transformers,
                                     get_distilbert_transformers2_simple,
                                     get_distilbert_transformers_complex,
                                     get_mobilebert_transformers)
from src.evaluation.src.evaluations.binary_evaluation import calculate_binary_classification_metrics


def start(
            dataset_path,
            root_save_path,
            num_epochs,
            max_items=None,
            model_name='mobilebert_transformers',
            glove_txt_path = None
          ):
    PREDICTION_THRESHOLD = 0.5
    train_X, train_y, test_X, test_y = get_dfs(dataset_path, separator=';', max_items=max_items, max_seq_length=512)


    model_map = {
        'glove': get_glove(train_X, train_y, test_X, test_y,root_save_path,num_epochs,glove_txt_path),
        'bert_transformers': get_bert_transformers(train_X, train_y, test_X, test_y,root_save_path,num_epochs),
        'bert_tf2': get_bert_tf2(train_X, train_y, test_X, test_y,root_save_path,num_epochs),
        'distilbert_transformers_comples': get_distilbert_transformers_complex(train_X, train_y, test_X, test_y,root_save_path,num_epochs),
        'distilbert_transformers2_simple': get_distilbert_transformers2_simple(train_X, train_y, test_X, test_y,root_save_path,num_epochs),
        'mobilebert_transformers': get_mobilebert_transformers(train_X, train_y, test_X, test_y,root_save_path,num_epochs),
    }

    train_function, output_save_path  = model_map[model_name]

    predictions, test_y = train_function()

    binary_evaluation_result = calculate_binary_classification_metrics(test_y, predictions, 'hate_speech', PREDICTION_THRESHOLD)
    binary_evaluation_result.save_to_disk(output_save_path)

    #todo when convertet to 0-1
    #    report = classification_report(
    #    true_classes, predicted_classes, target_names=class_labels
    #)





