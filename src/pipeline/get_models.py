from src.models.Glove.train_glove import train_glove
from src.models.Glove.glove_complex import train_glove_complex
from src.models.Bert.bert_transformers_train import train_bert_transformers
from src.models.Bert.bert_for_tf2_train import train_bert_for_tf2
from src.models.DistilBert.distilbert_transformers import train_distilbert_transformers
from src.models.DistilBert.distilbert_transformers2 import (
    train_distilbert_transformers2,
)
from src.models.MobileBert.mobilebert_transformers import train_mobilebert_transformers
from src.utils.file_utils import get_output_folder




def get_mobilebert_transformers(root_save_path):
    output_save_path = get_output_folder(root_save_path, "mobilebert_transformers")

    def train(train_X, train_y, test_X, test_y, num_epochs):

        return train_mobilebert_transformers(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            num_epochs=num_epochs,
        )

    return train, output_save_path


def get_distilbert_transformers2_simple(root_save_path):
    output_save_path = get_output_folder(
        root_save_path, "distilbert_transformers2_simple"
    )

    def train(train_X, train_y, test_X, test_y, num_epochs):

        return train_distilbert_transformers2(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            num_epochs=num_epochs,
        )

    return train, output_save_path


def get_distilbert_transformers_complex(root_save_path):
    output_save_path = get_output_folder(
        root_save_path, "distilbert_transformers_complex"
    )

    def train(train_X, train_y, test_X, test_y, num_epochs):

        return train_distilbert_transformers(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            num_epochs=num_epochs,
        )

    return train, output_save_path


def get_bert_tf2(root_save_path):
    output_save_path = get_output_folder(root_save_path, "bert_for_tf2")

    def train(train_X, train_y, test_X, test_y, num_epochs):

        return train_bert_for_tf2(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            num_epochs=num_epochs,
        )

    return train, output_save_path


def get_bert_transformers(root_save_path):
    output_save_path = get_output_folder(root_save_path, "bert_transformers")

    def train(train_X, train_y, test_X, test_y, num_epochs):

        return train_bert_transformers(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            num_epochs=num_epochs,
        )

    return train, output_save_path


def get_glove_complex(root_save_path, glove_txt_path):
    output_save_path = get_output_folder(root_save_path, "glove_complex")

    def train(train_X, train_y, test_X, test_y, num_epochs):
        if glove_txt_path is None:
            raise ValueError("Glove model requires path to file glove.6B.100d.txt ")

        return train_glove_complex(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            pretrained_model_path=glove_txt_path,  # r"C:\Users\barte\Downloads\glove.6B\glove.6B.100d.txt",
            num_epochs=num_epochs,
        )

    return train, output_save_path

def get_glove_simple(root_save_path, glove_txt_path):
    output_save_path = get_output_folder(root_save_path, "glove_simple")

    def train(train_X, train_y, test_X, test_y, num_epochs):
        if glove_txt_path is None:
            raise ValueError("Glove model requires path to file glove.6B.100d.txt ")

        return train_glove(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            save_folder_path=output_save_path,
            pretrained_model_path=glove_txt_path,  # r"C:\Users\barte\Downloads\glove.6B\glove.6B.100d.txt",
            num_epochs=num_epochs,
        )

    return train, output_save_path
