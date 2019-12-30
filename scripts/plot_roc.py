import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

labels = ["PLAIN", "PUNCT", "DATE", "LETTERS", "CARDINAL", "VERBATIM", "DECIMAL", "MEASURE", "MONEY", "ORDINAL", "TIME", "ELECTRONIC", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"]

# The csv must contain Correct and Predicted columns
df = pd.read_csv("best/out_v1.csv")
n_classes = len(labels)

y_test = label_binarize(df["Correct"].to_numpy(), classes=range(n_classes))
y_score = label_binarize(df["Predicted"].to_numpy(), classes=range(n_classes))

# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc))
    print(f"{labels[i]: <10} - {roc_auc:.4f} AUC")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('ROC "Curve" of binary multi-class token classification', fontsize=18)
plt.legend(loc="lower right", fontsize=12)
plt.show()
