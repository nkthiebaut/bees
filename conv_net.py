#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
sys.setrecursionlimit(10000)
import numpy as np

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, sgd, adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit

from utils import make_submission_file
from utils import regularization_objective



X, y, images_id = cPickle.load(open('train.pkl', 'rb'))

X = np.array(X).astype(np.float32)[:-1]
y = np.array(y).astype(np.int32)[:-1]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape

layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    # first stage of our convolutional layers
#    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
#    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
#    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 500}),
#    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 500}),

    # the output layer
    (DenseLayer, {'num_units': 1, 'nonlinearity': softmax}),
]

nouri_net = NeuralNet(
    layers0,

    update=adam,
    update_learning_rate=0.0001,
    #update_momentum=0.9,

    #batch_iterator_train=BatchIterator(batch_size=64),
    #batch_iterator_test=BatchIterator(batch_size=62),

    objective=regularization_objective,
    objective_lambda2=0.0025,

    train_split=TrainSplit(eval_size=0.25, stratify=False),
    #regression=True,
    max_epochs=10,
    verbose=3,
    )
"""
from nolearn.lasagne import PrintLayerInfo
nouri_net.verbose = 3
nouri_net.initialize()
layer_info = PrintLayerInfo()
layer_info(nouri_net)
#exit(0)
"""
nouri_net.fit(X, y)

with open('nouri_net.pkl', 'wb') as f:
    cPickle.dump(nouri_net, f, -1)

X_test, _, images_id = cPickle.load(open('test.pkl', 'rb'))

print "Test:"
print "X_test.shape:", X_test.shape
print "y.shape:", y.shape
predictions = nouri_net.predict_proba(X_test)

values, counts = np.unique(predictions, return_counts=True)
for v, c in zip(values,counts):
    print 'Number of {}: {}'.format(v,c)

make_submission_file(predictions, images_id)
