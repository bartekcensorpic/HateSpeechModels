import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers

from models.shared_utils.callbacks import get_callbacks


def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)

    return np.array(all_tokens)



def train_distilbert_transformers2(train_X, train_y, test_X, test_y,save_folder_path):

    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    n_epochs = 1
    print('[#### info lol ###]number of epochs:', n_epochs)

    transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = build_model(transformer_layer, max_len=160)
    model.summary()

    train_input = bert_encode(train_X, tokenizer, max_len=160)
    test_input = bert_encode(test_X, tokenizer, max_len=160)


    callbacks = get_callbacks(
        best_model_checkpoint_path=os.path.join(save_folder_path,'distilbert_transformers2_model.h5'),
        csv_logger_path=os.path.join(save_folder_path, 'history_log.csv'),
        tensorboard_logdir=os.path.join(save_folder_path, 'tensorboard'),
    )

    train_history = model.fit(
        train_input, train_y,
        validation_split=0.1,
        epochs=n_epochs,
        batch_size=32,
        callbacks=callbacks
    )

    predictions = model.predict(test_input, verbose=1)

    return predictions, test_y