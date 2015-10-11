#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle

from lasagne import layers
from nolearn.lasagne import NeuralNet

from utils import make_submission_file

nouri_net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 200, 200),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    hidden2_num_units=500,
    output_num_units=1, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
    )

X, y, images_id = cPickle.load(open('train.pkl', 'rb'))
nouri_net.fit(X, y)

with open('nouri_net.pkl', 'wb') as f:
    cPickle.dump(nouri_net, f, -1)

X_test, _, images_id = cPickle.load(open('test.pkl', 'rb'))

predictions = nouri_net.predict_proba(X_test)

make_submission_file(predictions, images_id)
