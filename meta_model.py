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

train_files = ['models/training_VGG11_2015-11-13.csv',
               'models/training_VGG11-maxout_2015-11-15.csv',
               'data/train_meta_infos.csv']

test_files = ['submissions/submission_VGG11_2015-11-13.csv',
              'submissions/submission_VGG11-maxout_2015-11-15.csv',
              'data/test_meta_infos.csv']

def merge_csv(files):
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df_right = pd.read_csv(f)
        df = pd.merge(df, df_right, how='inner', on=['id'])
    return df

def scorer(est, X, y):
    y_pred = est.predict_proba(X)
    return roc_auc_score(y, y_pred[:, 1])

clf_log = Pipeline([('pca', PCA(whiten=True)), ('log-reg', LogisticRegression(C=1000.))])
clf_test = LogisticRegression(C=1000., class_weight='balanced')
clf_neighbors = KNeighborsClassifier()
clf_RF = RandomForestClassifier(n_estimators=100, min_samples_split=10)
#clf_GBM = Pipeline([('ss', StandardScaler()), ('gbm', GradientBoostingClassifier())])
clf_GBM = GradientBoostingClassifier(max_depth=25)
#clf_GBM.set_params(gbm__max_depth=5)
parameters = {'gbm__max_depth': [3, 5, 7, 10]}


#clf = GridSearchCV(clf_GBM, param_grid=parameters, verbose=2, scoring=scorer)
clf = clf_GBM
X = merge_csv(train_files)

train_labels = 'train_labels.csv'
y = pd.read_csv("data/"+train_labels)#, index_col=0)
y = pd.merge(y, X, on=['id'], suffixes=('', '_y'))
X = X.drop('id', axis=1).values
y = y['genus'].values.astype(np.int64)

#X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
n_samples = len(X)

X_train = X[:.9 * n_samples]
y_train = y[:.9 * n_samples]
X_test = X[.9 * n_samples:]
y_test = y[.9 * n_samples:]

#print roc_auc_score(y_test, y_model[:, 1])
clf.fit(X_train, y_train)

print 'Classifier accuracy (train):', clf.score(X_train, y_train)
print 'Classifier accuracy (test):', clf.score(X_test, y_test)
y_pred = clf.predict_proba(X_test)
print 'Classifier AUC-ROC (test):', roc_auc_score(y_test, y_pred[:, 1])
#print clf.best_params_
#predictions = clf.feature_importances_
#print predictions

exit()
test_labels = 'SubmissionFormat.csv'
y_test = pd.read_csv("data/" + test_labels, index_col=0)
images_id_test = np.array(y_test.index.tolist())

X_test = merge_csv(test_files).drop('id', axis=1).values
predictions = clf.predict_proba(X_test)

name = '_' + str(date.today())
make_submission_file(predictions, images_id_test, output_filepath='submissions/meta_submission'+name+'.csv')
predictions_df = pd.DataFrame(predictions[:, 1], index=images_id, columns=['genus'])
predictions_df.index.names = ['id']
predictions_df.to_csv(output_filepath)
