#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '29/09/15'

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cPickle

from tqdm import tqdm
from PIL import Image
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


class DataManager(object):
    """ DataManager class that imports and pre-treat bees pictures """
    def __init__(self, root='data/images', test=False):
        directory = 'test' if test else 'train'
        self.path = os.path.join(root, directory)

        self.X = None
        self.y = pd.read_csv("data/train_labels.csv", index_col=0)
        self.images_id = self.y.index.tolist()
        self.y = np.array(self.y, dtype=np.float32)

        self.n_images = self.y.shape[0]
        self.n_features = None

        self.submission_format = pd.read_csv("data/SubmissionFormat.csv",
                                             index_col=0)

    def prepare_data(self):
        """ Treat the pictures """
        for i, img_id in tqdm(enumerate(self.images_id)):
            features = self.get_image(img_id)
            if self.X is None:
                self.n_features = features.shape[0]
                self.X = np.zeros((self.n_images, self.n_features),
                                  dtype=np.uint8)

            if not features.shape[0] == self.n_features:
                print "Error on image {}".format(img_id)

            self.X[i, :] = features

    def get_image(self, img_id):
        """
        Get pixels values from image id
        :param img_id: image id (int)
        :return: numpy flattened array with integer pixels values (np.uint8 array)
        """
        filename = "{}.jpg".format(img_id)
        filepath = os.path.join(self.path, filename)
        pixels = np.array(Image.open(filepath), dtype=np.uint8)
        if pixels.shape[2] == 4:
            print 'Image ' + str(img_id) + ' is RGBA (alpha), converting to RGB.'
            pixels = pixels[:, :, :3]
        return pixels.flatten()

    def shuffle(self):
        """ Shuffle the features, labels and ids"""
        self.X, self.y, self.images_id = shuffle(self.X, self.y, self.images_id,
                                              random_state=42)

    def normalize(self):
        """  Normalize all RGB channels separately, accross the training set """
        ss = StandardScaler()
        rgb = self.X.reshape(self.n_images, 200*200, 3).astype(np.float32)

        # Normalize over each RGB channel separately
        normalized = []
        for i in range(3):
            normalized.append(ss.fit_transform(rgb[:, :, i]))
        normalized = np.array(normalized)

        self.X = np.zeros((self.n_images, self.n_features),
                                     dtype=np.float32)

        for i in range(self.n_images):
            self.X[i, :] = normalized[:, i, :].flatten(order="F")

    def save(self, filename='data.pkl'):
        """  Save reshaped feature matrix, labels and ids in a pickle file
        :param filename: file where datas are saved
        """
        cPickle.dump((self.get_reshaped(), self.y, self.images_id,
                            open(filename, 'wb'))

    def reshape(self):
        """ Change feature_matrix shape for compatibility w. lasagne """
        self.X = self.X.reshape((self.n_images, 200, 200, 3))

    def get_reshaped(self):
        """ Get numpy matrix with shape compatible with lasagne
        :return: reshaped matrix
        """
        return self.X.reshape((self.n_images, 200, 200, 3))

    def show(self, img_id):
        """
        Show image with id number img_id
        :param img_id: image id (int)
        :return: matplotlib.pyplot imshow object
        """
        loc = np.where(self.images_id == img_id)
        plt.imshow(self.X[loc].reshape(200, 200, 3))
