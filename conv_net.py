#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
import numpy as np
import theano
from datetime import date

from sklearn.metrics import roc_auc_score

from lasagne.updates import nesterov_momentum, sgd, adam
from lasagne.nonlinearities import LeakyRectify

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import PrintLayerInfo

from utils import make_submission_file
from utils import regularization_objective
from utils import load_numpy_arrays
from utils import float32
from utils import plot_loss

from adaptative_learning import AdjustVariable
from adaptative_learning import EarlyStopping
from data_augmentation import DataAugmentationBatchIterator
from data_augmentation import FlipBatchIterator

from layers import *
sys.setrecursionlimit(10000)

# ----- Parameters -----
batch_size = 56
nb_channels = 3
crop_size = 150
activation_function = LeakyRectify(0.1)
init_learning_rate = 0.005
# ----------------------

X, y, images_id = load_numpy_arrays('train.pkl')
sample_size = y.shape[0] - y.shape[0] % batch_size
X = X[:sample_size]
y = y[:sample_size]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape
print "y value counts: ", np.unique(y, return_counts=True)

conv_net = NeuralNet(
    layers_team_oO,

    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(init_learning_rate)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=init_learning_rate, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=5),
        ],

    #batch_iterator_train=BatchIterator(batch_size=batch_size),
    batch_iterator_train=DataAugmentationBatchIterator(batch_size=batch_size, crop_size=crop_size),
    batch_iterator_test=DataAugmentationBatchIterator(batch_size=31, crop_size=crop_size),

    objective=regularization_objective,
    objective_lambda2=0.0005,

    #train_split=TrainSplit(eval_size=0.25, stratify=True),
    max_epochs=50,
    verbose=3,
    )

conv_net.fit(X, y)

with open('conv_net'+str(date.today)+'.pkl', 'wb') as f:
    cPickle.dump(conv_net, f, -1)

# ----- Train set ----
train_predictions = conv_net.predict_proba(X)
make_submission_file(train_predictions, images_id, output_filepath='submissions/training_'+str(date.today)+'.csv')
plot_loss(conv_net,"submissions/loss_"+str(date.today())+".png", show=False)
print "Train set AUC ROC: ", roc_auc_score(y, train_predictions[:, 1])

# ----- Test set ----
X_test, _, images_id_test = load_numpy_arrays('test.pkl')
print "Test:"
print "X_test.shape:", X_test.shape
predictions = conv_net.predict_proba(X_test)
make_submission_file(predictions, images_id_test, output_filepath='submissions/submission_'+str(date.today)+'.csv')
