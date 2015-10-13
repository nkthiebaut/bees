#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
sys.setrecursionlimit(10000)

from lasagne import layers
import numpy as np


from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
from lasagne.nonlinearities import softmax
from utils import make_submission_file

layers4 = [
    (InputLayer, {'shape': (64,3, 200, 200)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

nouri_net = NeuralNet(
    layers4,
    update_learning_rate=0.01,
    update_momentum=0.9,
    batch_iterator_train=BatchIterator(batch_size=64),
	train_split=TrainSplit(eval_size=0.25),

    max_epochs=10,
    verbose=1,
    )
#
#from nolearn.lasagne import PrintLayerInfo
#nouri_net.verbose = 3
#nouri_net.initialize()
#layer_info = PrintLayerInfo()
#layer_info(nouri_net)
#
X, y, images_id = cPickle.load(open('train.pkl', 'rb'))
X=X[:-1]
#y=y.reshape(-1,1)[:-1]
y=y[:-1].astype(np.int32)
print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape

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

predictions = nouri_net.predict_proba(X_test)

make_submission_file(predictions, images_id)
