import tensorflow as tf
from transformers import (
    DistilBertTokenizer,
    TFDistilBertModel,
)

from src.models.shared_utils.callbacks import get_callbacks
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.utils.keras_metrics import f1_m, precision_m, recall_m
import numpy as np
import os

# https://www.kaggle.com/xhlulu/disaster-nlp-distilbert-in-tf
# todo experiment with the tokenization function guy there uses


def encode_examples(X, y, tokenizer, max_length):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    # token_type_ids_list = []
    # attention_mask_list = []
    label_list = []

    ds = np.zeros((X.shape[0], 2, 1), dtype="object")
    _y = np.expand_dims(y, axis=1)
    _X = np.expand_dims(X, axis=1)
    ds[:, 0] = _X
    ds[:, 1] = _y


    for sentence, label in ds:

        s = sentence[0]
        bert_input = convert_example_to_feature(tokenizer, s, max_length)

        input_ids_list.append(bert_input["input_ids"])
        # token_type_ids_list.append(bert_input['token_type_ids'])
        # attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list, label_list)).map(
        map_example_to_dict
    )


def map_example_to_dict(input_ids, label):
    res = {"input_word_ids": input_ids}, label
    return res


def convert_example_to_feature(tokenizer, sentence, max_length):
    # combine step for tokenization, WordPiece vector mapping and will add also special tokens and truncate reviews longer than our max length

    return tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,  # add [CLS], [SEP]
        max_length=max_length,  # max length of the text that can go to BERT
        pad_to_max_length=True,  # add [PAD] tokens
        return_attention_mask=True,  # add attention mask to not focus on pad tokens
        truncation=True,
    )


def build_model(
    learning_rate, max_len=512,
):

    transformer_layer = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer_layer(input_word_ids)[0]

    cnn_filters = 100
    dnn_units = 256
    dropout_rate = 0.2

    cnn_layer1 = Conv1D(
        filters=cnn_filters, kernel_size=2, padding="valid", activation="relu"
    )
    cnn_layer2 = Conv1D(
        filters=cnn_filters, kernel_size=3, padding="valid", activation="relu"
    )
    cnn_layer3 = Conv1D(
        filters=cnn_filters, kernel_size=4, padding="valid", activation="relu"
    )
    pool = GlobalMaxPool1D()

    dense_1 = Dense(units=dnn_units, activation="relu")
    dropout = Dropout(rate=dropout_rate)
    last_dense = Dense(units=1, activation="sigmoid")

    l_1 = cnn_layer1(sequence_output)
    l_1 = pool(l_1)
    l_2 = cnn_layer2(sequence_output)
    l_2 = pool(l_2)
    l_3 = cnn_layer3(sequence_output)
    l_3 = pool(l_3)

    concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
    concatenated = dense_1(concatenated)
    concatenated = dropout(concatenated)
    model_output = last_dense(concatenated)

    model = Model(inputs=input_word_ids, outputs=model_output)
    model.compile(
        Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", f1_m, precision_m, recall_m],
    )

    print(model.summary())

    return model


def train_distilbert_transformers(
    train_X, train_y, test_X, test_y, save_folder_path, num_epochs
):
    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", do_lower_case=True
    )
    vocabulary = tokenizer.get_vocab()
    batch_size = 32
    learning_rate = 2e-5
    max_length = 160

    model = build_model(learning_rate, max_len=max_length)

    # train dataset
    ds_train_encoded = (
        encode_examples(train_X, train_y, tokenizer, max_length)
        .shuffle(10000)
        .batch(batch_size)
    )

    # test dataset
    ds_test_encoded = encode_examples(test_X, test_y, tokenizer, max_length).batch(
        batch_size
    )

    number_of_epochs = num_epochs
    print("[#### info lol ###]number of epochs:", number_of_epochs)

    [
        learning_rate_reduction,
        checkpoint_callback,
        tensorboard_callback,
    ] = get_callbacks(
        best_model_checkpoint_path=os.path.join(
            save_folder_path, "distilbert_transformers_model"
        ),
        csv_logger_path=os.path.join(save_folder_path, "history_log.csv"),
        tensorboard_logdir=os.path.join(save_folder_path, "tensorboard"),
    )

    bert_history = model.fit(
        ds_train_encoded,
        epochs=number_of_epochs,
        callbacks=[learning_rate_reduction, tensorboard_callback, checkpoint_callback],
        validation_data=ds_test_encoded,
    )


    predictions = model.predict_proba(ds_test_encoded)  # get probabilities of class 1

    return predictions, test_y
