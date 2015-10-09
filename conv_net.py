#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from DataManager import DataManager

nouri_net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 200, 200),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=1, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
    )

DM = DataManager()
DM.normalize()
DM.save_to_lasagne_format(filename='data.pkl')

X, y = cPickle.load(open('data.pkl', 'rb'))
nouri_net.fit(X, y)

with open('nouri_net.pkl', 'wb') as f:
    cPickle.dump(nouri_net, f, -1)

DM_test = DataManager(test=True)
DM_test.normalize()
X_test = DM_test.get_reshaped_features()
predictions = nouri_net.predict_proba(X)