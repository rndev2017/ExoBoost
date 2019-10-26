import os
import math
import re
import time
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef


data = pd.read_csv(
    os.getcwd()+"\\final_data\\orb_params.csv").drop(columns=["Unnamed: 0"])


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


predictions = model.predict(test_X)
test_y_arr = np.array(test_Y)

correctpred, total = 0, 0

for index in range(len(predictions)):
    if(predictions[index] == test_y_arr[index]):
        correctpred = correctpred + 1
    total = total+1

print('accuracy='+str((correctpred*100)/total))

time.sleep(5)

cm = confusion_matrix(test_y_arr, predictions)
print(cm)

time.sleep(5)

print(pd.crosstab(index=test_y_arr, columns=np.round(
    predictions), rownames=['actual'], colnames=['predictions']))

time.sleep(5)

test_preds = predictions
test_labels = test_y_arr

recall_per_class, classes = [], []
label_map = ['0', '1', '2']


for target_label in np.unique(test_labels):
    recall_numerator = np.logical_and(
        test_preds == target_label, test_labels == target_label).sum()
    recall_denominator = (test_labels == target_label).sum()
    recall_per_class.append(recall_numerator / recall_denominator)
    classes.append(label_map[target_label])
recall = pd.DataFrame({'recall': recall_per_class, 'class_label': classes})
recall.sort_values('class_label', ascending=False, inplace=True)

print(recall)

time.sleep(5)

# accuracy for entire dataset
correctpred_all, total_all = 0, 0

predictions_all = model.predict(data.iloc[:, 0:4])

all_y = np.array(data.iloc[:, -1])

for index in range(len(predictions_all)):
    if(predictions_all[index] == all_y[index]):
        correctpred_all = correctpred_all + 1
    total_all = total_all+1

print('accuracy='+str((correctpred_all*100)/total_all))

time.sleep(5)

print(pd.crosstab(index=all_y, columns=np.round(predictions_all),
                  rownames=['actual'], colnames=['predictions']))

time.sleep(5)

test_preds_all = predictions_all
test_labels_all = all_y

recall_per_class_all, classes_all = [], []


for target_label in np.unique(test_labels_all):
    recall_numerator_all = np.logical_and(
        test_preds_all == target_label, test_labels_all == target_label).sum()
    recall_denominator_all = (test_labels_all == target_label).sum()
    recall_per_class_all.append(recall_numerator_all / recall_denominator_all)
    classes_all.append(label_map[target_label])
recall_all = pd.DataFrame(
    {'recall': recall_per_class_all, 'class_label': classes_all})
recall_all.sort_values('class_label', ascending=False, inplace=True)
print(recall_all)

time.sleep(10)

print("Test Set: \n" + classification_report(test_labels, test_preds))
print("All Data: \n" + classification_report(test_labels_all, test_preds_all))

time.sleep(5)

print(matthews_corrcoef(test_labels_all, test_preds_all))
