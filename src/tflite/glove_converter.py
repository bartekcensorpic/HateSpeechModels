import pickle

import transformers
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def glove_convertion():



    PATH_TO_H5 = r"C:\Users\barte\Documents\C5_connection\models\2020-08-27-14-28-07_glove\glove_model.h5"
    TF_LITE_SAVE_PATH = r"C:\Users\barte\Documents\C5_connection\models\2020-08-27-14-28-07_glove\glove_model.tflite"
    TOKENIZER_PATH = r"C:\Users\barte\Documents\C5_connection\models\2020-08-27-14-28-07_glove\tokenizer.pickle"

    # LOAD MODEL
    base_model = tf.keras.models.load_model (
        PATH_TO_H5,
        custom_objects={"f1_m":f1_m, "precision_m":precision_m, "recall_m":recall_m},

    )


    #CONVERT MODEL TO TFLITE
    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    #converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    #https://github.com/huggingface/tflite-android-transformers/blob/master/models_generation/distilbert.py
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(TF_LITE_SAVE_PATH, "wb") as saver:
        saver.write(tflite_model)


    # LOAD TFLITE MODEL
    tflite_model = tf.lite.Interpreter(model_path=TF_LITE_SAVE_PATH)

    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    BATCH_SIZE = 2 #whatever you want
    MAX_LENGTH = 8192

    #MAKE SURE INPUTS AND OUTPUTS ARE AS THEY SHOULD BE
    tflite_model.resize_tensor_input(
        input_details[0]["index"], (BATCH_SIZE, MAX_LENGTH)
    )

    tflite_model.resize_tensor_input(
        output_details[0]["index"], (1, 1)
    )

    tflite_model.allocate_tensors()

    # LOAD TOKENIZER
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)


    #SENTENCES THAT YOU WANT TO TEST
    #[IMPORTANT] THEY SHOULD BE AS MANY OF THEM, AS BATCH_SIZE INDICATES
    test_sentences = ['Fucking piece of shit cops, protect and serve my ass', 'you are really nice person']
    encoded_docs = tokenizer.texts_to_sequences(test_sentences)
    # this is just adding zeroes to each sentence till the MAX_LENGTH is reached
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding="post")
    padded_docs = padded_docs.astype('float32')


    tflite_model.set_tensor(input_details[0]["index"], padded_docs)
    tflite_model.invoke()

    tflite_q_model_predictions = tflite_model.get_tensor(
        output_details[0]["index"]
    )

    print(tflite_q_model_predictions)
    print('hate speech from out dataset, should be close to 1:', tflite_q_model_predictions[0])
    print('normal sentence, should be close 0:', tflite_q_model_predictions[1])
    debug = 5


if __name__ == '__main__':
    glove_convertion()