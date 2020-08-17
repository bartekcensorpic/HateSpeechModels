import numpy as np


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y: np.ndarray, x:np.ndarray, class_name:str)-> plt.axes:
    """
    Plots confusion matrix

    :param y: true labels (binary 0-1)
    :param x: predicted labels (binary (0-1)
    :param class_name: class name
    :return: matplotlib.axis
    """
    cm = confusion_matrix(y,x)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels([f'non-{class_name}', class_name]);
    ax.yaxis.set_ticklabels([f'non-{class_name}', class_name]);

    return ax

