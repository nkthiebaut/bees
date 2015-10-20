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
from utils import load_numpy_arrays

batch_size = 48

X, y, images_id = load_numpy_arrays('train.pkl')
#X, y, images_id = cPickle.load(open('train.pkl', 'rb'))

sample_size = y.shape[0] - y.shape[0] % batch_size
X = X[:sample_size]
y = y[:sample_size]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape
print "y value counts: ", np.unique(y, return_counts=True)

layers4_mnist = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

nouri_net = NeuralNet(
    layers4_mnist,

    update=adam,
    update_learning_rate=0.0002,
    #update_momentum=0.9,

    batch_iterator_train=BatchIterator(batch_size=32),
    #batch_iterator_test=BatchIterator(batch_size=31),

    #objective=regularization_objective,
    #objective_lambda2=0.0025,

    #train_split=TrainSplit(eval_size=0.25, stratify=True),
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

X_test, y, images_id = load_numpy_arrays('test.pkl')
#X_test, _, images_id = cPickle.load(open('test.pkl', 'rb'))

print "Test:"
print "X_test.shape:", X_test.shape
print "y.shape:", y.shape
predictions = nouri_net.predict_proba(X_test)

values, counts = np.unique(predictions, return_counts=True)
for v, c in zip(values,counts):
    print 'Number of {}: {}'.format(v,c)

make_submission_file(predictions, images_id)
