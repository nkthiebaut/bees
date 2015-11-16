#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import date
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from utils import make_submission_file

__author__ = 'thiebaut'
__date__ = '16/11/15'


train_files = ['models/training_VGG11_2015-11-13.csv',
               'models/training_VGG11-maxout_2015-11-15.csv',
               'data/train_meta_infos.csv']

test_files = ['submissions/submission_VGG11_2015-11-13.csv',
              'submissions/submission_VGG11-maxout_2015-11-15.csv',
              'data/test_meta_infos.csv']

def average_models(model_file_a, model_file_b):
    df = pd.read_csv(model_file_a)
    df_right = pd.read_csv(model_file_b)
    df = pd.merge(df, df_right, how='inner', on=['id'])
    df['genus'] = (df['genus_x']+df['genus_y'])/2.
    df = df.drop(['genus_x', 'genus_y'], axis=1)
    return df

def merge_csv(files):
    df = average_models(files[0], files[1])
    for f in files[2:]:
        df_right = pd.read_csv(f)
        df = pd.merge(df, df_right, how='inner', on=['id'])
    return df


def auc_roc(y_true, y_prob, average=None):
    return roc_auc_score(y_true, y_prob[:,1], average=average)
#pca = PCA(whiten=True).fit(X)
clf_log = Pipeline([('pca', PCA(whiten=True)), ('log-reg', LogisticRegression(C=1000.))])
clf_test = LogisticRegression(C=1000., class_weight='balanced')
#clf_svm = svm.LinearSVC()
clf_neighbors = KNeighborsClassifier()
clf_RF = RandomForestClassifier(n_estimators=100, min_samples_split=10)
clf = clf_log

X = merge_csv(train_files)


train_labels = 'train_labels.csv'
y = pd.read_csv("data/"+train_labels)#, index_col=0)
y = pd.merge(y, X, on=['id'], suffixes=('','_y'))
X = X.drop('id', axis=1).values
y = np.array(y['genus']).astype(np.int32)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf.fit(X, y)
print 'Classifier accuracy (train):', clf.score(X, y)
print 'Classifier accuracy (test):', clf.score(X_test, y_test)
y_pred = clf.predict_proba(X_test)
print 'Classifier AUC-ROC (test):', roc_auc_score(y_test, y_pred[:, 0])

predictions = clf.predict(X_test)
print predictions

test_labels = 'SubmissionFormat.csv'
y_test = pd.read_csv("data/" + test_labels, index_col=0)
images_id_test = np.array(y_test.index.tolist())

X_test = merge_csv(test_files).drop('id', axis=1).values
merge_csv(test_files)
#pca.transform(X_test)
predictions = clf.predict_proba(X_test)
name = '_' + str(date.today())
make_submission_file(predictions, images_id_test, output_filepath='submissions/meta_submission'+name+'.csv')
