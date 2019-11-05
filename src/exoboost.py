import os
import math
import re
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef


class ExoBoost(xgb.XGBClassifier):


    def __init__(self, obj, lr, num_classes):
        """Constructs a ExoBoost object.

           Arguments:
               obj {str} -- what the model is trying to do (loss function?)
               lr {float} -- learning rate for the model
               num_class {int} -- the number of labels that the model classifies
        """
        xgb.XGBClassifier.__init__(self, objective=obj, learning_rate=lr,
                                   num_class=num_classes,
                                   gamma=4,
                                   max_depth=10,
                                   min_child_weight=2,
                                   n_estimators=3000,
                                   silent=0,
                                   subsample=0.8)



    def prepare_data(filepath):
        """Prepares data for model.

           Arguments:
               file_path {str} -- the path to data

           Returns:
               X {pandas.DataFrame} -- a DataFrame of features
               Y {pandas.DataFrame} -- a DataFrame of labels
        """
        data = pd.read_csv(file_path)
        X = data.iloc[:, 0:4]
        Y = data.iloc[:, -1]
        return X, Y


    def train_test_split(X, y, test_size=0.1, shuffle=True, random_state=40):
        """Seperates X & Y into test and validation sets.

           Arguments:
               X {pandas.DataFrame} -- a DataFrame of features
               y {pandas.DataFrame} -- a DataFrame of labels
               test_size {float} -- represents how much of entire data is
                                    reserved for the test set (default: 0.1 or 10%)
               shuffle {bool} -- should the data be shuffled? (default: True)
               random_state {int} -- the seed to choose randomness of the shuffle
                                     (default: 40)

           Returns:
               train_X {pandas.DataFrame} -- training set of features
               test_X {pandas.DataFrame} -- test set of features
               train_Y {pandas.DataFrame} -- training set of labels
               test_Y {pandas.DataFrame} -- test set of labels
        """
        train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.1, shuffle=True, stratify=Y, random_state=40)
        return train_X, test_X, train_Y, test_Y


    def create_eval_set(train_X, test_X, train_Y, test_Y):
        """Creates an evaluation set.

           Arguments:
             train_X {pandas.DataFrame} -- training set of features
             test_X {pandas.DataFrame} -- test set of features
             train_Y {pandas.DataFrame} -- training set of labels
             test_Y {pandas.DataFrame} -- test set of lab
         Returns:
            eval_set {list} - a list containing two tuples for evaluation set
      """
        eval_set = [(train_X, train_Y), (test_X, test_Y)]
        return eval_set


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
