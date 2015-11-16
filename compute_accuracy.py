#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

__author__ = 'thiebaut'
__date__ = '16/11/15'

def average_models(model_file_a, model_file_b):
    df = pd.read_csv(model_file_a)
    df_right = pd.read_csv(model_file_b)
    df = pd.merge(df, df_right, how='inner', on=['id'])
    df['genus'] = (df['genus_x']+df['genus_y'])/2.
    df = df.drop(['genus_x', 'genus_y'], axis=1)
    return df

df = pd.read_csv('models/training_VGG11_2015-11-13.csv')
y = pd.read_csv('data/train_labels.csv')
df = pd.merge(df, y, on=['id'])
y_pred = np.array(df['genus_x'], dtype=np.float64)
y_true = np.array(df['genus_y'], dtype=np.float64)
print "Model A AUC-ROC: ", roc_auc_score(y_true, y_pred)

df = pd.read_csv('models/training_VGG11-maxout_2015-11-15.csv')
y = pd.read_csv('data/train_labels.csv')
df = pd.merge(df, y, on=['id'])
y_pred = np.array(df['genus_x'], dtype=np.float64)
y_true = np.array(df['genus_y'], dtype=np.float64)
print "Model B AUC-ROC: ", roc_auc_score(y_true, y_pred)

average = average_models('models/training_VGG11_2015-11-13.csv', 'models/training_VGG11-maxout_2015-11-15.csv')
y_avg = np.array(average['genus'], dtype=np.float64)
print "Average models AUC-ROC: ", roc_auc_score(y_true, y_avg)


"""
y_bis = []
for pred in y_pred:
    p = 1. if pred >= 0.5 else 0.
    y_bis.append(p)
"""

#print "Model accuracy: ", accuracy_score(y_true, y_bis)


