#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
sys.setrecursionlimit(10000)

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit

from utils import make_submission_file
from utils import regularization_objective



X, y, images_id = cPickle.load(open('train.pkl', 'rb'))

import numpy as np
#X = X[:-1]
#y = y.reshape(-1,1).astype(np.float32)

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
#    (DenseLayer, {'num_units': 64}),
#    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 1, 'nonlinearity': softmax}),
]

nouri_net = NeuralNet(
    layers0,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    #batch_iterator_train=BatchIterator(batch_size=64),
    #batch_iterator_test=BatchIterator(batch_size=62),

    objective=regularization_objective,
    objective_lambda2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    #regression=True,
    max_epochs=10,
    verbose=3,
    )

from nolearn.lasagne import PrintLayerInfo
nouri_net.verbose = 3
nouri_net.initialize()
layer_info = PrintLayerInfo()
layer_info(nouri_net)

#exit(0)
nouri_net.fit(X, y)

with open('nouri_net.pkl', 'wb') as f:
    cPickle.dump(nouri_net, f, -1)

X_test, _, images_id = cPickle.load(open('test.pkl', 'rb'))

print "Test:"
print "X_test.shape:", X_test.shape
print "y.shape:", y.shape
predictions = nouri_net.predict_proba(X_test)

print predictions

make_submission_file(predictions, images_id)
