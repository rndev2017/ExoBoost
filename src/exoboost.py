import os
import math
import re
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
# os.environ["PATH"] += os.pathsep + "\\path\\to\\graphviz"

data = pd.read_csv(
    "path\\to\\data.csv").drop(columns=["Unnamed: 0"]) # FIX ME!


X = data.iloc[:, 0:4]
Y = data.iloc[:, -1]

train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.1, shuffle=True, stratify=Y, random_state=40)

# model
model = xgb.XGBClassifier(objective='multi:softmax',
                          learning_rate=0.2,
                          gamma=4,
                          max_depth=10,
                          min_child_weight=2,
                          num_class=3,
                          n_estimators=3000,
                          silent=0,
                          subsample=0.8)
eval_set = [(train_X, train_Y), (test_X, test_Y)]
model.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=5000,
          eval_metric=["merror"], eval_set=eval_set, verbose=True)

predictions = model.predict(X)
test_y_arr = np.array(Y)


def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy of the model.

       Arguments:
           y_true {numpy.array} -- the true labels corresponding to each input
           y_pred {numpy.array} -- the model's predictions

       Returns:
           accuracy {str} -- the accuracy of the model (%)
    """
    correctpred, total = 0, 0

    for index in range(len(y_pred)):
        if(y_pred[index] == y_true[index]):
            correctpred = correctpred + 1
        total = total+1

    return 'accuracy='+str((correctpred*100)/total)


def calculate_recall(y_true, y_pred):
    """Calculates the model's recall on all three classes.

       Arguments:
           y_true {numpy.array} -- the true labels corresponding to each input
           y_pred {numpy.array} -- the model's predictions

       Returns:
           recall {pandas.DataFrame} -- a DataFrame that contains the
                                        recall of the model
    """
    recall_per_class, classes = [], []
    label_map = ['0', '1', '2']

    for target_label in np.unique(y_true):
        recall_numerator = np.logical_and(
            y_pred == target_label, y_true == target_label).sum()
        recall_denominator = (y_true == target_label).sum()
        recall_per_class.append(recall_numerator / recall_denominator)
        classes.append(label_map[target_label])
    recall = pd.DataFrame({'recall': recall_per_class, 'class_label': classes})
    recall.sort_values('class_label', ascending=False, inplace=True)

    return recall

def generate_classification_report(y_true, y_pred):
    """Using sklearn to build a text report showing the main classification
       metrics.

       Arguments:
           y_true {numpy.array} -- the true labels corresponding to each input
           y_pred {numpy.array} -- the model's predictions

       Returns:
           report {str} -- contains the F1-score, precision, recall, & support
                           of each class
    """
    return classification_report(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    """Using sklearn to compute a confusion matrix to evaluate
       the accuracy of a classification.

       Arguments:
           y_true {numpy.array} -- the true labels corresponding to each input
           y_pred {numpy.array} -- the model's predictions

       Returns:
           confusion_matrix {numpy.array} -- the calculated confusion matrix
    """
    return confusion_matrix(y_true, y_pred)

def calculate_mmc(y_true, y_pred):
    """Using sklearn to compute the Matthews correlation coefficient (MCC).

       Arguments:
           y_true {numpy.array} -- the true labels corresponding to each input
           y_pred {numpy.array} -- the model's predictions

       Returns:
           mmc {float} -- the calculated Matthews correlation coefficient
    """
    return matthews_corrcoef(y_true, y_pred)
