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
sys.setrecursionlimit(10000)

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from utils import get_image


class DataManager(object):
    """ DataManager class that imports and pre-treat bees pictures """
    def __init__(self, root='data/images', test=False):

        self.test = test
        directory = 'test' if test else 'train'
        labels_file = 'SubmissionFormat.csv' if test else 'train_labels.csv'
        self.path = os.path.join(root, directory)

        self.X = None
        self.y = pd.read_csv("data/"+labels_file, index_col=0)
        self.n_classes = self.y['genus'].value_counts()
        self.images_id = np.array(self.y.index.tolist())
        self.y = np.array(self.y['genus']).astype(np.float32)

        self.n_images = self.y.shape[0]
        self.n_features = None

        self.prepare_data()

    def prepare_data(self):
        """ Treat the pictures """
        for i, img_id in tqdm(enumerate(self.images_id)):
            features = get_image(self.path, img_id)
            if self.X is None:
                self.n_features = features.shape[0]
                self.X = np.zeros((self.n_images, self.n_features),
                                  dtype=np.uint8)

            if not features.shape[0] == self.n_features:
                print "Error on image {}".format(img_id)

            self.X[i, :] = features


    def shuffle(self):
        """ Shuffle the features, labels and ids"""
        self.X, self.y, self.images_id = shuffle(self.X, self.y, self.images_id,
                                                 random_state=42)

    def normalize(self):
        """  Normalize all RGB channels separately, accross the training set """
        #self.ss = StandardScaler()
        rgb = self.X.reshape(self.n_images, 200*200, 3).astype(np.float32)

        if self.test:
            with  open('std_scaler.pkl', 'rb') as f:
                self.std_scaler = cPickle.load(f)
        else:
            self.std_scaler = []
        # Normalize over each RGB channel separately
        normalized = []
        for i in tqdm(range(3)):
            if not self.test:
                ss = StandardScaler().fit(rgb[:, :, i])
                self.std_scaler.append(ss)
            normalized.append(self.std_scaler[i].transform(rgb[:, :, i]))
        normalized = np.array(normalized)

        self.X = np.zeros((self.n_images, self.n_features),
                                     dtype=np.float32)

        for i in range(self.n_images):
            self.X[i, :] = normalized[:, i, :].flatten(order="F")

        if not self.test:
            with open('std_scaler.pkl', 'wb') as f:
                cPickle.dump(self.std_scaler, f)

    def save_to_lasagne_format(self, filename=None):
        """  Save reshaped feature matrix, labels and ids in a pickle file
        :param filename: file where datas are saved
        """
        if filename is None:
            filename = 'test.pkl' if self.test else 'train.pkl'
        with open(filename, 'wb') as f:
            cPickle.dump(self.get_in_lasagne_format(), f, -1)

    def get_in_lasagne_format(self):
        return (self.get_reshaped_features(), self.y, self.images_id)

    def get_reshaped_features(self):
        """ Get numpy matrix with shape compatible with lasagne
        :return: reshaped matrix
        """
        # return self.X.reshape(self.n_images, 200, 200, 3)
        return self.X.reshape(-1, 3, 200, 200)

    def show(self, img_id):
        """ Show image with id number img_id
        :param img_id: image id (int)
        """
        loc = np.where(self.images_id == img_id)[0]
        if len(loc) != 1:
            raise IndexError('Image with id'+str(img_id)+'cannot be found.')
        plt.imshow(self.X[loc].reshape(200, 200, 3))

    def equalize_classes(self, random=False):
        """ Copy underepresented class until equality is reached """
        # Get classes occurrences difference
        delta = int(reduce(lambda x, y: y-x, self.n_classes))

        j = 0
        for i in range(delta):
            while True:
                if random:
                    j = np.random.randint(0, self.n_images)
                else:
                    j = (j+1)%self.n_images
                if self.y[j] == 0.:
                    print 'ok'
                    break
            map(lambda x: np.append(x, x[j]), [self.X, self.y, self.images_id])

    def __getitem__(self, index):
        """ Overload the [] operator """
        return self.X[index].reshape(200, 200, 3)
