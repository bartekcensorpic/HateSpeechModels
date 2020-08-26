import os

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
from tensorflow.keras import layers
from bert.tokenization.bert_tokenization import FullTokenizer
import bert
import re
import random
import numpy as np
import math
from src.utils.keras_metrics import f1_m, precision_m, recall_m
from src.models.shared_utils.callbacks import get_callbacks


class TEXT_MODEL(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimensions=128,
        cnn_filters=50,
        dnn_units=512,
        model_output_classes=2,
        dropout_rate=0.1,
        training=False,
        name="text_model",
    ):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(
            filters=cnn_filters, kernel_size=2, padding="valid", activation="relu"
        )
        self.cnn_layer2 = layers.Conv1D(
            filters=cnn_filters, kernel_size=3, padding="valid", activation="relu"
        )
        self.cnn_layer3 = layers.Conv1D(
            filters=cnn_filters, kernel_size=4, padding="valid", activation="relu"
        )
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(
                units=model_output_classes, activation="softmax"
            )

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat(
            [l_1, l_2, l_3], axis=-1
        )  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


def remove_tags(TAG_RE, text):
    return TAG_RE.sub("", text)


def preprocess_text(TAG_RE, sen):
    # Removing html tags
    sentence = remove_tags(TAG_RE, sen)

    # Remove punctuations and numbers
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Removing multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


def tokenize_reviews(tokenizer, text_sentences):
    tokenized_sentence = tokenizer.tokenize(text_sentences)
    ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    return ids


def pad_sentence(sentence, max_padding_length):
    padded = pad_sequences([sentence], maxlen=max_padding_length, padding="post")
    padded = padded.reshape(-1).tolist()
    return padded


def process_dataset(tokenizer, X, y, BATCH_SIZE, max_padding_length: None):
    tokenized_train_X = [tokenize_reviews(tokenizer, sentence) for sentence in X]

    reviews_with_len = [
        [review, y[i], len(review)] for i, review in enumerate(tokenized_train_X)
    ]

    reviews_with_len.sort(key=lambda x: x[2])

    if max_padding_length is None:
        max_padding_length = reviews_with_len[-1][2]

    sorted_reviews_labels = [
        (pad_sentence(review_lab[0], max_padding_length), review_lab[1])
        for review_lab in reviews_with_len
    ]

    processed_dataset = tf.data.Dataset.from_generator(
        lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32)
    )
    batched_dataset = processed_dataset.padded_batch(
        BATCH_SIZE, padded_shapes=((None,), ())
    )
    return batched_dataset, max_padding_length


def train_bert_for_tf2(train_X, train_y, test_X, test_y, save_folder_path, num_epochs):

    TAG_RE = re.compile(r"<[^>]+>")

    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    BertTokenizer = bert.tokenization.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        trainable=False,
    )
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


    BATCH_SIZE = 32

    train_dataset, max_padding_length = process_dataset(
        tokenizer, train_X, train_y, BATCH_SIZE, max_padding_length=None
    )
    test_dataset, _ = process_dataset(
        tokenizer, test_X, test_y, BATCH_SIZE, max_padding_length=max_padding_length
    )

    # TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
    # TEST_BATCHES = TOTAL_BATCHES // 10
    # batched_dataset.shuffle(TOTAL_BATCHES)
    # test_data = batched_dataset.take(TEST_BATCHES)
    # train_data = batched_dataset.skip(TEST_BATCHES)

    VOCAB_LENGTH = len(tokenizer.vocab)
    EMB_DIM = 200
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 2

    DROPOUT_RATE = 0.2

    NB_EPOCHS = num_epochs
    print("[#### info lol ###]number of epochs:", NB_EPOCHS)

    text_model = TEXT_MODEL(
        vocabulary_size=VOCAB_LENGTH,
        embedding_dimensions=EMB_DIM,
        cnn_filters=CNN_FILTERS,
        dnn_units=DNN_UNITS,
        model_output_classes=OUTPUT_CLASSES,
        dropout_rate=DROPOUT_RATE,
    )

    [
        learning_rate_reduction,
        checkpoint_callback,
        tensorboard_callback,
    ] = get_callbacks(
        best_model_checkpoint_path=os.path.join(save_folder_path, "bert_for_tf2_model"),
        csv_logger_path=os.path.join(save_folder_path, "history_log.csv"),
        tensorboard_logdir=os.path.join(save_folder_path, "tensorboard"),
    )

    if OUTPUT_CLASSES == 2:
        text_model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", f1_m, precision_m, recall_m],
        )
    else:
        text_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["sparse_categorical_accuracy", f1_m, precision_m, recall_m],
        )

    text_model.fit(
        train_dataset,
        epochs=NB_EPOCHS,
        callbacks=[learning_rate_reduction, checkpoint_callback, tensorboard_callback],
        validation_data=test_dataset,
    )

    results = text_model.evaluate(test_dataset)
    print(results)

    predictions = text_model.predict(test_dataset)

    return predictions, test_y
