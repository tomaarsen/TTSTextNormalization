
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.metrics import confusion_matrix

import math

def MakeFormat(prefixes, step, limit, base):
    def Format(n, suffix='', places=2):
        if abs(n) < limit:
            if n == int(n):
                return "%s %s" % (n, suffix)
            else:
                return "%.1f %s" % (n, suffix)
        magnitude = math.log(abs(n) / limit, base) / step
        magnitude = min(int(magnitude)+1, len(prefixes)-1)

        return '%.1f%s%s' % (
        float(n) / base ** (magnitude * step),
        prefixes[magnitude], suffix)
    return Format

DecimalFormat = MakeFormat(
    prefixes = ['', 'k', 'm', 'b', 't'],
    step = 3,
    limit = 100,
    base = 10)

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    This function prints and plots the confusion matrix.
    """
    title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, norm=LogNorm())
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.autoscale(tight=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = 500000
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, DecimalFormat(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)#3
    fig.tight_layout()
    return cm

if __name__ == "__main__":
    # The csv must contain Predicted and Correct columns
    df = pd.read_csv("best/out_v1_limit.csv")
    labels = ["PLAIN", "PUNCT", "DATE", "LETTERS", "CARDINAL", "VERBATIM", "DECIMAL", "MEASURE", "MONEY", "ORDINAL", "TIME", "ELECTRONIC", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"]

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(df["Predicted"].to_numpy().astype(np.int8), df["Correct"].to_numpy().astype(np.int8), labels)
    plt.show()
