#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '29/09/15'

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

class DataManager(object):

    def __init__(self, root='data/images', test=False):
        dir = 'test' if test else 'train'
        self.path = os.path.join(root, dir)

        # Get training labels
        self.y = pd.read_csv("data/train_labels.csv", index_col=0)
        self.n_imgs = self.y.shape[0]

        self.submission_format = pd.read_csv("data/SubmissionFormat.csv",
                                             index_col=0)

    def prepare_data(self):
        self.X = None

        for i, img_id in tqdm(enumerate(self.y.index)):
            features = self.get_image(img_id)
            if self.X is None:
                self.n_features = features.shape[0]
                self.X = np.zeros((self.n_imgs, self.n_features),
                                  dtype=np.float32)

            if not features.shape[0] == self.n_features:
                print "Error on image {}".format(img_id)

            self.X[i, :] = features


    def get_image(self, img_id):
        filename = "{}.jpg".format(img_id)
        filepath = os.path.join(self.path, filename)
        pixels = np.array(Image.open(filepath), dtype=np.int32)
        if pixels.shape[2] > 3:
            print 'Image: '+str(img_id)+' is RGBA, converting to RGB.'
        pixels = pixels[:,:,:3]
        return pixels.flatten()

    def shuffle(self):
        self.X, self.y = shuffle(self.X, self.y, random_state=42)

    def normalize(self):
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)

    def get_params(self):
        return self.X, self.y

    def get_reshaped(self):
        return self.X.reshape((3969,3,200,200))

    def show(self, img_id):
        loc = self.y.index.get_loc(img_id)
        plt.imshow(self.X[loc].reshape(200,200,3))