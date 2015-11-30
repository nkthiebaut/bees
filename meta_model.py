#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import date
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from utils import make_submission_file

__author__ = 'thiebaut'
__date__ = '16/11/15'


models = ['VGG11-maxout_2015-11-23', 'VGG11_2015-11-23',
          'VGG13_2015-11-23', 'VGG13-maxout_2015-11-23',
          'AlexNet_2015-11-22', #'reformed-gamblers_2015-11-23',
          'VGG11-maxout_2015-11-28_pretrain', #'VGG13-maxout_2015-11-28_pretrain',
          'MyNet_2015-11-27', 'AlexNet_2015-11-27']


#train_files = ['models/training_'+m+'.csv' for m in models]
train_files = ['train.csv']
train_files += ['models/raw_hog_daisy_rbf_train.csv']
train_files += ['data/train_meta_infos.csv']

#test_files = ['submissions/submission_'+m+'.csv' for m in models]
test_files = ['submissions/avg_submission_2015-11-30.csv']
test_files += ['submissions/raw_hog_daisy_rbf.csv']
test_files += ['data/test_meta_infos.csv']


def average_models(df, df_right):
    df = pd.merge(df, df_right, how='inner', on=['id'])
    df['genus'] = (df['genus_x']+df['genus_y'])/2.
    df = df.drop(['genus_x', 'genus_y'], axis=1)
    return df

def merge_csv(files):
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df_right = pd.read_csv(f)
        df = pd.merge(df, df_right, how='inner', on=['id'])
    return df


def scorer(est, X, y):
    y_pred = est.predict_proba(X)
    return roc_auc_score(y, y_pred[:, 1])

clf_lg =  LogisticRegressionCV()
#Pipeline([('pca', PCA(whiten=True)),
         #       ('log-reg', LogisticRegressionCV())])


clf_GBM = GradientBoostingClassifier(max_depth=25)
clf_RF = RandomForestClassifier()
#clf_GBM.set_params(gbm__max_depth=5)
#parameters = {'learning_rate': [0.2, 0.15, 0.1, 0.08]}
parameters = {'n_estimators': [50, 100, 200, 500, 1000]}
clf = GridSearchCV(clf_RF, param_grid=parameters, verbose=1, scoring=scorer)
#clf = clf_RF
X = merge_csv(train_files)

train_labels = 'train_labels.csv'
y = pd.read_csv("data/"+train_labels)  # , index_col=0)
y = pd.merge(y, X, on=['id'], suffixes=('', '_test'))
X = y.drop(['id','genus'], axis=1).values
y = y['genus'].values.astype(np.int64)
y = np.array([(i+1)%2 for i in y],dtype=np.int64)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_samples = len(X)

X_train = X[:.8 * n_samples]
y_train = y[:.8 * n_samples]
X_test = X[.8 * n_samples:]
y_test = y[.8 * n_samples:]

clf.fit(X_train, y_train)

#print cross_val_score(clf, X, y)
print 'Classifier accuracy (train):', clf.score(X_train, y_train)
print 'Classifier accuracy (validation):', clf.score(X_test, y_test)
y_pred = clf.predict_proba(X_test)
print 'Classifier AUC-ROC (validation):', roc_auc_score(y_test, y_pred[:, 1])

"""
test_labels = 'SubmissionFormat.csv'
y_test = pd.read_csv("data/" + test_labels, index_col=0)
images_id_test = np.array(y_test.index.tolist())
"""

df_test = merge_csv(test_files)
ids = df_test['id'].values
X_test = df_test.drop('id', axis=1).values
predictions = clf.predict_proba(X_test)
name = '_' + str(date.today())
output_filepath='submissions/meta_submission'+name+'.csv'
#make_submission_file(predictions, images_id_test, output_filepath='submissions/meta_submission'+name+'.csv')
predictions_df = pd.DataFrame(predictions[:, 0], index=ids, columns=['genus'])
predictions_df.index.names = ['id']
predictions_df.to_csv(output_filepath)

