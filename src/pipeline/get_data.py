import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def get_dfs(data_path, separator, max_seq_length = 256):

    #data = pd.read_csv(data_path, sep=';')
    data = pd.read_csv(data_path, sep=separator)
    #todo stratify
    X_train, X_test = train_test_split(data, test_size=0.2)

    train_text = X_train['Comment_text'].tolist()
    train_text = [' '.join(str(t).split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = X_train['Hateful_or_not'].tolist()

    test_text = X_test['Comment_text'].tolist()
    test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = X_test['Hateful_or_not'].tolist()

    train_X = train_text
    train_y = train_label
    test_X = test_text
    test_y = test_label

    return train_X, train_y, test_X, test_y