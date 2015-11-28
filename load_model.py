#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:01:06 2015

@author: thiebaut
"""

import cPickle
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from lasagne.nonlinearities import rectify, leaky_rectify, very_leaky_rectify

from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

from utils import GetOptions, load_numpy_arrays
from conv_net import build_network

args = GetOptions()

if args['activation'] == 'rectify':
    activation_function = rectify
elif args['activation'] == 'leaky_rectify':
    activation_function = leaky_rectify
elif args['activation'] == 'very_leaky_rectify':
    activation_function = very_leaky_rectify
else:
    raise ValueError('Unknown activation function')
args['activation_function'] = activation_function


X, y, images_id = load_numpy_arrays(args['train_file'])
sample_size = y.shape[0] - y.shape[0] % args['batch_size']
X = X[:sample_size]
y = y[:sample_size]

print "Train:"
print "X.shape:", X.shape
print "y.shape:", y.shape
y_counts = np.unique(y, return_counts=True)[1]
print "y value counts: ", y_counts
print "pictures size: ", sqrt(X.shape[1]/3.)


args['dataset_ratio'] = 3.8
args['network'] = 'AlexNet'
args['batch_size'] = 141
exp_name = args['network']

print "Input arguments:", args
conv_net = build_network(**args)

if args['load']:
    with open(args['load'], 'rb') as f:
        loaded_net = cPickle.load(f)
    conv_net.load_params_from(loaded_net)


plot_conv_weights(conv_net.layers_[1], figsize=(4, 4))
plt.savefig('weights.png')

plot_conv_activity(conv_net.layers_[1], X[0:1])
plt.savefig('activity.png')

plot_occlusion(conv_net, X[:5], y[:5])
plt.savefig('occlusion.png')
