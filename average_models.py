#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from datetime import date
from utils import make_submission_file
__author__ = 'thiebaut'
__date__ = '16/11/15'

def average_models(model_file_a, model_file_b):
    df = pd.read_csv(model_file_a)
    df_right = pd.read_csv(model_file_b)
    df = pd.merge(df, df_right, how='inner', on=['id'])
    df['genus'] = (df['genus_x']+df['genus_y'])/2.
    df = df.drop(['genus_x', 'genus_y'], axis=1)
    return df

df = pd.read_csv('models/training_VGG11-maxout_2015-11-17.csv')
y = pd.read_csv('data/train_labels.csv')
df = pd.merge(df, y, on=['id'])
y_pred = np.array(df['genus_x'], dtype=np.float64)
y_true = np.array(df['genus_y'], dtype=np.float64)
print "Model A AUC-ROC: ", roc_auc_score(y_true, y_pred)

"""
df = pd.read_csv('models/training_VGG11-maxout_2015-11-15.csv')
y = pd.read_csv('data/train_labels.csv')
df = pd.merge(df, y, on=['id'])
y_pred = np.array(df['genus_x'], dtype=np.float64)
y_true = np.array(df['genus_y'], dtype=np.float64)
print "Model B AUC-ROC: ", roc_auc_score(y_true, y_pred)
"""
average = average_models('models/training_VGG11_2015-11-13.csv', 'models/training_VGG11-maxout_2015-11-17.csv')
y_avg = np.array(average['genus'], dtype=np.float64)
print "Average models AUC-ROC: ", roc_auc_score(y_true, y_avg)

test_labels = 'SubmissionFormat.csv'
y_test = pd.read_csv("data/" + test_labels, index_col=0)
images_id_test = np.array(y_test.index.tolist())
name = '_' + str(date.today())

def make_submission_file(predictions, images_id, output_filepath="submission_" + str(date.today()) + ".csv"):
    predictions_df = pd.DataFrame(predictions, index=images_id, columns=['genus'])
    predictions_df.index.names = ['id']
    predictions_df.to_csv(output_filepath)

predictions = average_models('submissions/submission_VGG11_2015-11-13.csv',
                             'submissions/submission_VGG11-maxout_2015-11-17.csv')['genus'].values

make_submission_file(predictions, images_id_test, output_filepath='submissions/avg_submission'+name+'.csv')
"""
y_bis = []
for pred in y_pred:
    p = 1. if pred >= 0.5 else 0.
    y_bis.append(p)
"""

#print "Model accuracy: ", accuracy_score(y_true, y_bis)


