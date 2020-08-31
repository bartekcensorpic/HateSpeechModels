import transformers
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[: max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len

        all_tokens.append(tokens)

    return np.array(all_tokens)


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




def distilbert_transformers2_simple_conversion():



    #path to a folder with .pb file, assets and variable folders
    PATH_TO_PB = r"C:\Users\barte\Documents\C5_connection\models\2020-08-26-09-20-50_distilbert_transformers2_simple\distilbert_transformers2_model"
    TH_LITE_SAVE_PATH = r"C:\Users\barte\Documents\C5_connection\models\2020-08-26-09-20-50_distilbert_transformers2_simple\distilbert_model.tflite"
    base_model = tf.keras.models.load_model(
        PATH_TO_PB,
        custom_objects={"f1_m":f1_m, "precision_m":precision_m, "recall_m":recall_m},

    )
    input_spec = tf.TensorSpec([1, 160], tf.int32)
    base_model._set_inputs(input_spec, training=False)

    #using keras
    #converter = tf.lite.TFLiteConverter.from_keras_model(base_model)

    # using pb
    converter = tf.lite.TFLiteConverter.from_saved_model(PATH_TO_PB)
    #https://github.com/huggingface/tflite-android-transformers/blob/master/models_generation/distilbert.py
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    with open(TH_LITE_SAVE_PATH, "wb") as saver:
        saver.write(tflite_model)


    tflite_model = tf.lite.Interpreter(model_path=TH_LITE_SAVE_PATH)


    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    MAX_LENGTH = 160
    BATCH_SIZE = 2


    tflite_model.resize_tensor_input(
        input_details[0]["index"], (BATCH_SIZE, MAX_LENGTH)
    )

    tflite_model.resize_tensor_input(
        output_details[0]["index"], (1, 1)
    )

    #[WARNING] next line throws this error and i haven't managed to solve it yet:
    #RuntimeError: Regular TensorFlow ops are not supported by this interpreter. Make sure you apply/link the Flex delegate before inference.Node number 0 (FlexIdentity) failed to prepare.
    tflite_model.allocate_tensors()


    #prepare inputs
    #have a look there, perhaps it helps with using it in Java:
    # https://github.com/huggingface/tflite-android-transformers/tree/master/bert/src/main/java/co/huggingface/android_transformers/bertqa
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    test_sentences = ['Fucking piece of shit cops, protect and serve my ass', 'you are really nice person']
    test_sentences = np.asarray(test_sentences).reshape(-1)
    padded_docs = bert_encode(test_sentences, tokenizer, max_len=MAX_LENGTH)


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
    distilbert_transformers2_simple_conversion()

