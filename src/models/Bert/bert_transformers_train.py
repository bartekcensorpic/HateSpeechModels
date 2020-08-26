import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

from src.models.shared_utils.callbacks import get_callbacks
from src.utils.keras_metrics import f1_m, precision_m, recall_m
import numpy as np
import os


def encode_examples(X, y, tokenizer, max_length):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
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
        token_type_ids_list.append(bert_input["token_type_ids"])
        attention_mask_list.append(bert_input["attention_mask"])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
    ).map(map_example_to_dict)


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return (
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        },
        label,
    )


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


def train_bert_transformers(
    train_X, train_y, test_X, test_y, save_folder_path, num_epochs
):
    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    vocabulary = tokenizer.get_vocab()
    batch_size = 6
    learning_rate = 2e-5

    # can be up to 512 for BERT
    max_length = 512
    # COMMENT ABOUT IMDB DATASET
    # the recommended batches size for BERT are 16,32 ... however on this dataset we are overfitting quite fast
    # and smaller batches work like a regularization.
    # You might play with adding another dropout layer instead.
    # END COMMENT ABOUT IMDB DATASET

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

    # we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
    number_of_epochs = num_epochs
    print("[#### info lol ###]number of epochs:", number_of_epochs)
    # model initialization
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

    # classifier Adam recommended
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=[metric, f1_m, precision_m, recall_m]
    )

    [
        learning_rate_reduction,
        checkpoint_callback,
        tensorboard_callback,
    ] = get_callbacks(
        best_model_checkpoint_path=os.path.join(
            save_folder_path, "bert_transformers_model"
        ),
        csv_logger_path=os.path.join(save_folder_path, "history_log.csv"),
        tensorboard_logdir=os.path.join(save_folder_path, "tensorboard"),
    )

    bert_history = model.fit(
        ds_train_encoded,
        epochs=number_of_epochs,
        callbacks=[learning_rate_reduction, checkpoint_callback, tensorboard_callback],
        validation_data=ds_test_encoded,
    )


    predictions = model.predict_proba(ds_test_encoded)  # get probabilities of class 1

    return predictions, test_y
