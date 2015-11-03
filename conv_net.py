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

# ----- Parameters -----
batch_size = 64
nb_channels = 3
crop_size = 200
init_learning_rate = 0.01
activation_function = rectify
lambda2=0.0005
max_epochs=100
exp_name=sys.argv[1]
# ----------------------

X, y, images_id = load_numpy_arrays('train.npz')
sample_size = y.shape[0] - y.shape[0] % batch_size
X = X[:sample_size]
y = y[:sample_size]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape
print "y value counts: ", np.unique(y, return_counts=True)

layers_test = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

    (Conv2DLayer, {'num_filters': 16, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),


    (DenseLayer, {'num_units': 16}),

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

layers_A = [
    (InputLayer, {'shape': (None, nb_channels, crop_size, crop_size)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (5, 5), 'stride': 2,'pad': 2, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3),'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (5, 5), 'stride':2, 'pad': 2, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (5, 5), 'stride':2, 'pad': 2, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (5, 5), 'stride':2, 'pad': 2, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1, 'nonlinearity':activation_function}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

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

VGGNet = [
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

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]


layers_team_oO = [
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

    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity':activation_function}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

layers_krizhevsky = [
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

def auc_roc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob[:,1])

conv_net = NeuralNet(
    layers_simonyan,
    update=adam,
    update_learning_rate=0.0002,

    #update=nesterov_momentum,
    #update_learning_rate=theano.shared(float32(init_learning_rate)),
    #update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        #AdjustVariable('update_learning_rate', start=init_learning_rate, stop=0.0001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=5),
        ],

    #batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
    batch_iterator_train=DataAugmentationBatchIterator(batch_size=batch_size, crop_size=crop_size),
    #batch_iterator_test=DataAugmentationBatchIterator(batch_size=31, crop_size=crop_size),

    objective=regularization_objective,
    objective_lambda2=lambda2,

    #train_split=TrainSplit(eval_size=0.25, stratify=True)
    custom_score=('AUC-ROC', auc_roc),
    max_epochs=max_epochs,
    verbose=3,
    )


conv_net.fit(X, y)

name = exp_name + '_'+ str(date.today())
with open('models/conv_net_'+name+'.pkl', 'wb') as f:
    cPickle.dump(conv_net, f, -1)

# ----- Train set ----
train_predictions = conv_net.predict_proba(X)
make_submission_file(train_predictions[:sample_size], images_id[:sample_size], output_filepath='submissions/training_'+name+'.csv')
plot_loss(conv_net,"submissions/loss_"+name+".png", show=False)

# ----- Test set ----
X_test, _, images_id_test = load_numpy_arrays('test.npz')
print "Test:"
print "X_test.shape:", X_test.shape
predictions = conv_net.predict_proba(X_test)
make_submission_file(predictions, images_id_test, output_filepath='submissions/submission_'+str(date.today())+'.csv')

