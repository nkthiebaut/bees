#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:28:59 2015

@author: thiebaut
"""

import pickle
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer


def build_vgg_cnn_s(size):
    net = {}
    net['input'] = InputLayer((None, 3, size, size))
    net['conv1'] = Conv2DLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = MaxPool2DLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = Conv2DLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = Conv2DLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
    net['conv4'] = Conv2DLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
    net['conv5'] = Conv2DLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

    #  !wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl

    model = pickle.load(open('vgg_cnn_s.pkl'))

    lasagne.layers.set_all_param_values(net['fc7'], model['values'][:14])

    return net



# !wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
def build_vgg_16(size=224):
    net = {}
    net['input'] = InputLayer((None, 3, size, size))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=2,
                            nonlinearity=lasagne.nonlinearities.softmax)

    model = pickle.load(open('vgg_16.pkl'))

    lasagne.layers.set_all_param_values(net['fc7'], model['values'][:-1])

    return net
