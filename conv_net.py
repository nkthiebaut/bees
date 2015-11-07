#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '02/10/15'

import cPickle
import sys
import numpy as np
import theano

from sklearn.metrics import roc_auc_score

from lasagne.updates import nesterov_momentum
from lasagne.updates import sgd
from lasagne.updates import adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import PrintLayerInfo

from utils import make_submission_file
from utils import regularization_objective
from utils import load_numpy_arrays
from utils import float32
from utils import plot_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
#from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
#from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import rectify

from adaptative_learning import AdjustVariable
from adaptative_learning import EarlyStopping
from data_augmentation import DataAugmentationBatchIterator
from data_augmentation import FlipBatchIterator

sys.setrecursionlimit(10000)


def build_layers(name='VGG16', nb_channels=3, crop_size=200, activation_function=rectify):
    """

    :rtype : list
    :param nb_channels: Number of channels per pixels (1 for black and white, 3 for RGB pictures
    :param crop_size: image width and height  after batch data augmentation
    :param activation_function: neurons activation function (same for all)
    :return: model_zoo
    """

    assert isinstance(name, str)

    zoo = {}

    zoo['layers_test'] = [
        (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

        (Conv2DLayer, {'num_filters': 16, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (DenseLayer, {'num_units': 16}),

        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    zoo['layers_mnist'] = [
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

    zoo['VGG16'] = [
        (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),

        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    zoo['VGG19'] = [
        (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),

        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]


    zoo['team_oO'] = [
        (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

        (Conv2DLayer, {'num_filters': 32, 'filter_size': (5, 5), 'stride': 2, 'pad': 2, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 64, 'filter_size': (5, 5), 'stride': 2, 'pad': 2, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 512, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),

        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    zoo['AlexNet'] = [
        (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

        (Conv2DLayer, {'num_filters': 96, 'filter_size': (11, 11), 'stride': 4, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (3, 3), 'stride': 2}),

        (Conv2DLayer, {'num_filters': 256, 'filter_size': (5, 5), 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (3, 3), 'stride': 2}),

        (Conv2DLayer, {'num_filters': 384, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 384, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
        (MaxPool2DLayer, {'pool_size': (3, 3), 'stride': 2}),

        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 4096, 'nonlinearity':activation_function}),
        (DropoutLayer, {}),

        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    try:
        layers = zoo[name]
    except KeyError:
        print(name+' not found in available model zoo.')

    return layers


def auc_roc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob[:,1])


def build_network(network_name, data_augmentation='full', lambda2=0.0005, max_epochs=50, nb_channels=3, crop_size=200,
                  activation_function=rectify, batch_size=48, init_learning_rate=0.01):
    """Build nolearn neural network and returns it

    :param network: pre-defined network name
    :param data_augmentation: type of batch data aug. ('no', 'flip' or 'full')
    :return: NeuralNet nolearn object
    """
    if data_augmentation == 'no':
        batch_iterator_train = BatchIterator(batch_size=batch_size)
    elif data_augmentation == 'flip':
        batch_iterator_train = FlipBatchIterator(batch_size=batch_size)
    elif data_augmentation == 'full':
        batch_iterator_train = DataAugmentationBatchIterator(batch_size=batch_size, crop_size=crop_size)
    else:
        raise ValueError(data_augmentation+' is an unknown data augmentation strategy.')

    layers = build_layers(network_name, nb_channels=nb_channels, crop_size=crop_size,
                          activation_function=activation_function)

    conv_net = NeuralNet(
        layers,

        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(init_learning_rate)),
        update_momentum=theano.shared(float32(0.9)),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=init_learning_rate, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=5),
            ],

        batch_iterator_train = batch_iterator_train,
        # batch_iterator_test=DataAugmentationBatchIterator(batch_size=31, crop_size=crop_size),

        objective=regularization_objective,
        objective_lambda2=lambda2,

        train_split=TrainSplit(eval_size=0.1, stratify=True),
        custom_score=('AUC-ROC', auc_roc),
        max_epochs=max_epochs,
        verbose=3,
        )
    return conv_net