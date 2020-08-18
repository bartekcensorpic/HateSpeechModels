from sklearn.metrics import classification_report

from src.pipeline.get_data import get_dfs
from src.models.Glove.train_glove import train_glove
from src.models.Bert.bert_transformers_train import train_bert_transformers
from src.utils.file_utils import get_output_folder
from src.evaluation.src.evaluations.binary_evaluation import calculate_binary_classification_metrics


def get_bert_transformers(train_X, train_y, test_X, test_y,root_save_path):
    output_save_path = get_output_folder(root_save_path, 'bert')

    def train():

        return train_bert_transformers(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path
        )

    return train, output_save_path

def get_glove(train_X, train_y, test_X, test_y,root_save_path):
    output_save_path = get_output_folder(root_save_path, 'glove')
    def train():
        return train_glove(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            pretrained_model_path=r"C:\Users\barte\Downloads\glove.6B\glove.6B.100d.txt"
        )
    return train, output_save_path





def start(dataset_path,root_save_path):
    PREDICTION_THRESHOLD = 0.5
    train_X, train_y, test_X, test_y = get_dfs(dataset_path, separator=';')


    model_map = {
        'glove': get_glove(train_X, train_y, test_X, test_y,root_save_path),
        'bert_transformers': get_bert_transformers(train_X, train_y, test_X, test_y,root_save_path),

    }



    train_function, output_save_path  = model_map['bert_transformers']

    predictions, test_y = train_function()

    binary_evaluation_result = calculate_binary_classification_metrics(test_y, predictions, 'hate_speech', PREDICTION_THRESHOLD)
    binary_evaluation_result.save_to_disk(output_save_path)

    #todo when convertet to 0-1
    #    report = classification_report(
    #    true_classes, predicted_classes, target_names=class_labels
    #)





