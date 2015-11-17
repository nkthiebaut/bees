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


def average_models(df, df_right):
    df = pd.merge(df, df_right, how='inner', on=['id'])
    df['genus'] = (df['genus_x']+df['genus_y'])/2.
    df = df.drop(['genus_x', 'genus_y'], axis=1)
    return df


def compute_auc_roc_and_accuracy(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    y = pd.read_csv('data/train_labels.csv')
    df = pd.merge(df, y, on=['id'])
    y_pred = np.array(df['genus_x'], dtype=np.float64)
    y_true = np.array(df['genus_y'], dtype=np.float64)
    print "Model AUC-ROC: ", roc_auc_score(y_true, y_pred)
    y_bis = []
    for pred in y_pred:
        p = 1. if pred >= 0.5 else 0.
        y_bis.append(p)
    print "Model accuracy: ", accuracy_score(y_true, y_bis)

def make_submission_file(predictions, output_filepath="submission_" + str(date.today()) + ".csv"):
    test_labels = 'SubmissionFormat.csv'
    y_test = pd.read_csv("data/" + test_labels, index_col=0)
    images_id_test = np.array(y_test.index.tolist())
    predictions_df = pd.DataFrame(predictions, index=images_id_test, columns=['genus'])
    predictions_df.index.names = ['id']
    predictions_df.to_csv(output_filepath)

model_a = 'models/training_VGG11_2015-11-13.csv'
model_b = 'models/training_VGG11-maxout_2015-11-17.csv'

compute_auc_roc_and_accuracy(model_a)
compute_auc_roc_and_accuracy(model_b)

df = pd.read_csv(model_a)
df_right = pd.read_csv(model_b)

predictions = average_models(df, df_right)

compute_auc_roc_and_accuracy(predictions)

# ------------ Make prediction ---------------------
test_a = 'submissions/submission_VGG11_2015-11-13.csv'
test_b = 'submissions/submission_VGG11-maxout_2015-11-17.csv'
df = pd.read_csv(test_a)
df_right = pd.read_csv(test_b)
predictions = average_models(df, df_right)

outfile = 'submissions/avg_submission_' + str(date.today()) +'.csv'
make_submission_file(predictions['genus'], output_filepath=outfile)

