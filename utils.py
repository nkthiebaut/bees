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


def get_image(path, img_id, n_channels=3):
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

def plot_loss(net, filename="submissions/loss_"+str(date.today())+".png", show=False):
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.savefig(filename)
    if show:
        plt.show()
