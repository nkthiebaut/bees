#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '01/10/15'

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import pandas as pd

from tqdm import tqdm
from PIL import Image
from nolearn.lasagne import objective
from lasagne.layers import get_all_params


def show_img(img_id, root='data/images', test=False):
    dir = 'test' if test else 'train'
    path = os.path.join(root, dir, '{}.jpg'.format(img_id))
    im = Image.open(path)
    im_np = np.array(im)
    plt.imshow(im_np)


def get_image(path, img_id):
    """
    Get pixels values from image id
    :param img_id: image id (int)
    :return: numpy flattened array with integer pixels values (np.uint8 array)
    """
    filename = "{}.jpg".format(img_id)
    filepath = os.path.join(path, filename)
    pixels = np.array(Image.open(filepath), dtype=np.uint8)
    if pixels.shape[2] == 4:
        # raise Warning('Warning: Image ' + str(img_id) + ' is RGBA (alpha), converting to RGB.')
        pixels = pixels[:, :, :3]
    return pixels.flatten()

def make_submission_file(predictions ,images_id, output_filepath="submission_"+str(date.today())+".csv"):
    predictions_df = pd.DataFrame(predictions, index=images_id, columns=['genus'])
    predictions_df.index.names = ['id']
    predictions_df.to_csv(output_filepath)


def equalize_classes(X, y, images_id, random=False):
    """ Copy underepresented class until equality is reached """
    print "Equalizing classes."
    # Get classes occurrences difference
    delta = abs(len(np.where(y==0)[0]) - len(np.where(y==1)[0]))
    n_images= X.shape[0]
    width = X.shape[2]
    n_features = width*width
    
    X_append = np.zeros((delta, 3, width, width))
    y_append = np.zeros(delta)
    images_id_append = np.zeros(delta)
    j = 0
    for i in tqdm(range(delta)):
        while True:
            if random:
                j = np.random.randint(0, n_images)
            else:
                j = (j+1)%n_images
            if y[j] == np.int32(0):
                break
#           X_append[i, :] = X[j]
        images_id_append[i] = images_id[j]
    X = np.append(X, X_append, axis=0)
    y = np.append(y, y_append)
    images_id = np.append(images_id, images_id_append)

    print "Feature matrix shape after equalization: {}".format(X.shape)
    #map(lambda x: np.append(x, x[j]), [X, y, images_id])
    return X, y, images_id


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

