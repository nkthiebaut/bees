#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '29/09/15'

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cPickle
import sys
sys.setrecursionlimit(1000000)

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from skimage.util import pad

from utils import get_image


class DataManager(object):
    """ DataManager class that imports and pre-treat bees pictures """
    def __init__(self, root='data/images', test=False, rgb=True, width=200):

        self.test = test
        directory = 'test' if test else 'train'
        labels_file = 'SubmissionFormat.csv' if test else 'train_labels.csv'
        self.path = os.path.join(root, directory)
        self.n_channels = 3 if rgb else 1
        self.width = width
        self.X = None
        self.y = pd.read_csv("data/"+labels_file, index_col=0)
        self.n_classes = self.y['genus'].value_counts()
        self.images_id = np.array(self.y.index.tolist())
        self.y = np.array(self.y['genus']).astype(np.int32)

        self.n_images = self.y.shape[0]
        self.n_features = None

        self.prepare_data()

    def prepare_data(self):
        """ Treat the pictures """
        tmp = 'test' if self.test else 'train'
        print " ---- Loading "+tmp+" set ----"
        for i, img_id in tqdm(enumerate(self.images_id)):
            features = get_image(self.path, img_id, width=self.width)
            if self.X is None:
                self.n_features = features.shape[0]
                self.X = np.zeros((self.n_images, self.n_features),
                                  dtype=np.uint8)

            if not features.shape[0] == self.n_features:
                print "Error on image {}".format(img_id)

            self.X[i, :] = features
        print "Feature_matrix shape: {}".format(self.X.shape)

    def shuffle(self):
        """ Shuffle the features, labels and ids"""
        print "{} shuffles.".format(type(self).__name__)
        self.X, self.y, self.images_id = shuffle(self.X, self.y, self.images_id)

    def normalize(self):
        """  Normalize all RGB channels separately, accross the training set """
        print "{} is normalizing per RGB channel.".format(type(self).__name__)
        rgb = self.X.reshape(self.n_images, self.width*self.width, self.n_channels).astype(np.float32)

        if self.test:
            with open('std_scaler.pkl', 'rb') as f:
                self.std_scaler = cPickle.load(f)
        else:
            self.std_scaler = []
        # Normalize over each RGB channel separately
        normalized = []
        for i in tqdm(xrange(self.n_channels)):
            if not self.test:
                ss = StandardScaler().fit(rgb[:, :, i])
                self.std_scaler.append(ss)
            normalized.append(self.std_scaler[i].transform(rgb[:, :, i]))
        normalized = np.array(normalized)

        self.X = np.zeros((self.n_images, self.n_features),
                                     dtype=np.float32)

        for i in xrange(self.n_images):
            self.X[i, :] = normalized[:, i, :].flatten(order="F")

        if not self.test:
            with open('std_scaler.pkl', 'wb') as f:
                cPickle.dump(self.std_scaler, f)

    def pad(self, new_width):
        """ Pad pictures with reflections
        :param new_width: desired pictures width (and height)
        :return:
        """
        print "{} is padding pictures.".format(type(self).__name__)
        pad_size = new_width - self.width
        new_X = np.zeros((self.n_images, new_width * new_width * self.n_channels))
        pre_pad = pad_size // 2 + pad_size % 2
        post_pad = pad_size // 2
        if pad_size < 0:
            raise ValueError('new_width should not be smaller than the original one.')

        for i in tqdm(xrange(self.n_images)):
            padded = np.zeros((3, new_width, new_width))
            pic = self.X[i].reshape(self.width, self.width, self.n_channels)
            for j in xrange(self.n_channels):
                padded[j] = pad(np.swapaxes(pic, 0, 2)[j], (pre_pad, post_pad), 'reflect')
            padded = np.swapaxes(padded, 0, 2)
            new_X[i] = padded.flatten()
        self.width = new_width
        self.X = new_X

    def save_to_lasagne_format(self, filename=None):
        """  Save reshaped feature matrix, labels and ids in a cPickle file
        :param filename: file where datas are saved
        """
        print "{} saving datas.".format(type(self).__name__)
        if filename is None:
            filename = 'test.npz' if self.test else 'train.npz'
        with open(filename, 'wb') as f:
            np.savez(f, np.swapaxes(self.X.reshape(-1, self.width, self.width, self.n_channels), 1, 3),
                     self.y, self.images_id)

    def get_in_lasagne_format(self):
        return (self.get_reshaped_features(), self.y, self.images_id)

    def get_reshaped_features(self):
        """ Get numpy matrix with shape compatible with lasagne
        :return: reshaped matrix
        """
        return np.swapaxes(self.X.reshape(-1, self.width, self.width, self.n_channels), 1, 3)

    def show(self, img_id):
        """ Show image with id number img_id
        :param img_id: image id (int)
        """
        loc = np.where(self.images_id == img_id)[0]
        if len(loc) != 1:
            raise IndexError('Image with id'+str(img_id)+'cannot be found.')
        plt.imshow(self.X[loc].reshape(self.width, self.width, self.n_channels))

    def equalize_classes(self, random=False):
        """ Copy underepresented class until equality is reached """
        print "{} is equalizing classes.".format(type(self).__name__)

        # Get classes occurrences difference
        delta = int(reduce(lambda x, y: x-y, self.n_classes))

        X_append = np.zeros((delta, self.n_features), dtype=self.X.dtype)
        y_append = np.zeros(delta)
        images_id_append = np.zeros(delta)
        j = 0
        for i in tqdm(range(delta)):
            while True:
                if random:
                    j = np.random.randint(0, self.n_images)
                else:
                    j = (j+1)%self.n_images
                if self.y[j] == np.int32(0):
                    break
            X_append[i, :] = self.X[j]
            y_append[i] = self.y[j]
            images_id_append[i] = self.images_id[j]
        self.X = np.append(self.X, X_append, axis=0)
        self.y = np.append(self.y, y_append)
        self.images_id = np.append(self.images_id, images_id_append)

        print "Feature matrix shape after equalization: {}".format(self.X.shape)
        self.n_images += delta

    def __getitem__(self, index):
        """ Overload the [] operator """
        return self.X[index].reshape(self.width, self.width, self.n_channels)
