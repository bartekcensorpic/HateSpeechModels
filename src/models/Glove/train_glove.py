from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D

from src.models.shared_utils.callbacks import get_callbacks
from src.utils.keras_metrics import f1_m, precision_m, recall_m
import os

def get_max_length(train_X):
    longest_sent = max(train_X, key=len)
    length = len(longest_sent)
    return length



def train_glove(train_X, train_y, test_X, test_y,save_folder_path, pretrained_model_path):
    train_X = train_X.reshape(-1)
    test_X = test_X.reshape(-1)
    train_y = array(train_y)
    test_y = array(test_y)
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(train_X)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(train_X)

    max_length = get_max_length(train_X)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    #prepare test set
    test_sequences = t.texts_to_sequences(test_X)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')


    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(pretrained_model_path, mode='rt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # define model
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False) # todo: try trainable=True
    model.add(e)
    model.add(Conv1D(32, 8, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    # summarize the model
    model.summary()
    # fit the model

    callbacks = get_callbacks(
        best_model_checkpoint_path=os.path.join(save_folder_path,'glove_model.h5'),
        csv_logger_path=os.path.join(save_folder_path, 'history_log.csv'),
        tensorboard_logdir=os.path.join(save_folder_path, 'tensorboard'),
    )

    history = model.fit(padded_docs,
                        train_y,
                        epochs=50,
                        verbose=1,
                        callbacks = callbacks,
                        validation_data=(test_padded, test_y))

    #todo to be used later
    losses_and_shit = model.evaluate(test_padded, test_y)

    predictions = model.predict_proba(test_padded)#get probabilities of class 1


    return predictions, test_y

