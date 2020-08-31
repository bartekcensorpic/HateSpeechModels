#https://www.tensorflow.org/lite/tutorials/model_maker_text_classification

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tflite_model_maker import configs
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier


from models.shared_utils.callbacks import get_callbacks
from utils.keras_metrics import f1_m, precision_m, recall_m



def train_mobilebert_tflite_maker(
    train_X, train_y, test_X, test_y, save_folder_path, num_epochs
):

    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    spec = model_spec.get('mobilebert_classifier')

    #todo train and test data
    #todo metrics
    model = text_classifier.create(train_data, model_spec=spec)

    loss, acc = model.evaluate(test_data)

    config = configs.QuantizationConfig.create_dynamic_range_quantization(
        optimizations=[tf.lite.Optimize.OPTIMIZE_FOR_LATENCY])
    config._experimental_new_quantizer = True

    #todo path to tflite
    model.export(export_dir='mobilebert/', quantization_config=config)


    predictions = model.predict(test_input, verbose=1)

    return predictions, test_y
