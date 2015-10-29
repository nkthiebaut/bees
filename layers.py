#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '25/10/15'

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax

from conv_net import crop_size
from conv_net import nb_channels
from conv_net import activation_function

layers_test = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

layers_mnist = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

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

layers_simonyan = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

layers_team_oO = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

    (Conv2DLayer, {'num_filters': 16, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 16, 'filter_size': (1, 1), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 1), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 1), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': (1, 1), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': (1, 1), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]
