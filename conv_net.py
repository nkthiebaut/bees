#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
import numpy as np
import theano

from sklearn.metrics import roc_auc_score
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import leaky_rectify
from lasagne.updates import nesterov_momentum, sgd, adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import PrintLayerInfo

from utils import make_submission_file
from utils import regularization_objective
from utils import load_numpy_arrays
from utils import float32
from utils import print_predictions

from adaptative_learning import AdjustVariable
from adaptative_learning import EarlyStopping
from data_augmentation import DataAugmentationBatchIterator
sys.setrecursionlimit(10000)


batch_size = 64
crop_size = 150

X, y, images_id = load_numpy_arrays('train.pkl')

sample_size = y.shape[0] - y.shape[0] % batch_size
X = X[:sample_size]
y = y[:sample_size]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape
print "y value counts: ", np.unique(y, return_counts=True)

layersA_simonyan = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 4096}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 4096}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

layersA_simonyan_leaky = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':leaky_rectify}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 4096, 'nonlinearity':leaky_rectify}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 4096, 'nonlinearity':leaky_rectify}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]


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

conv_net = NeuralNet(
    layersA_simonyan,

    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=10),
        ],

    #batch_iterator_train=BatchIterator(batch_size=batch_size),
    batch_iterator_train=DataAugmentationBatchIterator(batch_size=batch_size, crop_size=crop_size),
    #batch_iterator_test=BatchIterator(batch_size=31),

    objective=regularization_objective,
    objective_lambda2=0.0005,

    #train_split=TrainSplit(eval_size=0.25, stratify=True),
    max_epochs=10,
    verbose=3,
    )

conv_net.fit(X, y)

with open('conv_net.pkl', 'wb') as f:
    cPickle.dump(conv_net, f, -1)

X_test, y, images_id = load_numpy_arrays('test.pkl')

print "Test:"
print "X_test.shape:", X_test.shape
print "y.shape:", y.shape
predictions = conv_net.predict_proba(X_test)
make_submission_file(predictions, images_id)

# print_predictions(predictions)

# train_predictions = conv_net.predict_proba(X)
# print "AUC ROC: ", roc_auc_score(y, train_predictions[:, 1])