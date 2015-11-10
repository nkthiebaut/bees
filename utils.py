#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '01/10/15'

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import pandas as pd

from PIL import Image
from nolearn.lasagne import objective
from lasagne.layers import get_all_params
from skimage.transform import AffineTransform
from skimage.transform import warp
from skimage.util import pad
from math import pi


def float32(k):
    return np.cast['float32'](k)


def show_img(img_id, root='data/images', test=False):
    dir = 'test' if test else 'train'
    path = os.path.join(root, dir, '{}.jpg'.format(img_id))
    im = Image.open(path)
    im_np = np.array(im)
    plt.imshow(im_np)


def print_predictions(predictions):
    values, counts = np.unique(predictions, return_counts=True)
    for v, c in zip(values, counts):
        print 'Number of {}: {}'.format(v, c)


def get_image(path, img_id, n_channels=3, width=None):
    """
    Get pixels values from image id
    :param img_id: image id (int)
    :return: numpy flattened array with integer pixels values (np.uint8 array)
    """
    filename = "{}.jpg".format(img_id)
    filepath = os.path.join(path, filename)
    pixels = np.array(Image.open(filepath), dtype=np.uint8)
    if pixels.shape[2] > n_channels:
        # raise Warning('Warning: Image ' + str(img_id) + ' is RGBA (alpha), converting to RGB.')
        pixels = pixels[:, :, :n_channels]
    if width is not None:
        pixels = pixels[:width, :width, :]
    return pixels.flatten()


def make_submission_file(predictions, images_id, output_filepath="submission_" + str(date.today()) + ".csv"):
    predictions_df = pd.DataFrame(predictions[:, 1], index=images_id, columns=['genus'])
    predictions_df.index.names = ['id']
    predictions_df.to_csv(output_filepath)


def load_numpy_arrays(filename):
    f = open(filename, 'rb')
    data = np.load(f)
    return np.array(data['arr_0']).astype(np.float32), np.array(data['arr_1']).astype(np.int32), data['arr_2']


def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses


def plot_loss(net, filename="submissions/loss_" + str(date.today()) + ".png", show=False):
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.savefig(filename)
    if show:
        plt.show()


def data_augmentation_test(img_id=1, crop_size=200, pad_size=100):
    Xb = np.array(Image.open('data/images/train/' + str(img_id) + '.jpg'), dtype=np.uint8) / np.float32(255.)

    im_size = Xb.shape[0]
    frame_size = im_size + 2 * pad_size
    print "X shape ", Xb.shape
    padded = np.zeros((3, frame_size, frame_size))
    for i in range(3):
        padded[i] = pad(np.swapaxes(Xb, 0, 2)[i], (pad_size, pad_size), 'reflect')
    padded = np.swapaxes(padded, 0, 2)
    print "Padded shape ", padded.shape

    lower_cut = (im_size - crop_size) / 2 + pad_size
    upper_cut = (im_size + crop_size) / 2 + pad_size
    shift_x = frame_size / 2
    shift_y = shift_x
    tf_shift = AffineTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = AffineTransform(translation=[shift_x, shift_y])

    scaling_factor = 0.2 * np.random.random() + 0.9
    angle = 2 * pi * (np.random.random() - 0.5)
    trans_x = np.random.randint(-5, 5)
    trans_y = np.random.randint(-5, 5)

    tf = AffineTransform(scale=(scaling_factor, scaling_factor), rotation=angle, shear=None,
                         translation=(trans_x, trans_y))
    padded = warp(padded, (tf_shift + (tf + tf_shift_inv)).inverse)
    print "Padded shape after transform ", padded.shape

    # Crop to desired size
    tmp = padded[lower_cut:upper_cut, lower_cut:upper_cut, :]
    print "Finally, cuts and shape: ", lower_cut, upper_cut, padded.shape
    plt.imshow(tmp)


import argparse


def GetOptions():
    """ Retrieve options from standard input """
    p = argparse.ArgumentParser(description='Neural net. training',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('network', metavar="network", type=str, default='VGG16',
                   help='Network name (should be defined in the model zoo).')
    p.add_argument('-b', '--batch-size', metavar="batch_size", type=int, default=48,
                   help='Batch size')
    p.add_argument('--max-epochs', metavar="max_epochs", type=int, default=50,
                   help='Minimum distance value')
    p.add_argument('--channels', metavar="channel", type=int, default=3,
                   help='Number of color channels (3 for RGB)')
    p.add_argument('--crop-size', metavar="crop_size", type=int, default=200,
                   help='Pictures batch data augmentation crop size.')
    p.add_argument('-d', '--data-aug', metavar="data_aug_type", type=str, default='full',
                   help='Batch data augmentation type')
    p.add_argument('--activation', metavar="activation", type=str, default='rectify',
                   help='Activation function (rectify, leaky_rectify, very_leaky_rectify')
    p.add_argument('--learning_rate', metavar="learning_rate", type=float, default=0.01,
                   help='Initial learning rate of Nesterov momentum method')
    p.add_argument('--lambda2', metavar="lambda2", type=float, default=0.0005,
                   help='Lambda2 regularization term')

    p.add_argument('--excited', action="store_true", dest="excited", help='')
    args = vars(p.parse_args())
    return args
