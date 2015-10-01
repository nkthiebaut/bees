#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '01/10/15'

import os

import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def load(root='/data/images', test=False):

   n_imgs = label_dataframe.shape[0]

    # initialized after first call to
    feature_matrix = None

    for i, img_id in tqdm(enumerate(label_dataframe.index)):
        features = preprocess(get_image(img_id))

        # initialize the results matrix if we need to
        # this is so n_features can change as preprocess changes
        if feature_matrix is None:
            n_features = features.shape[0]
            feature_matrix = np.zeros((n_imgs, n_features), dtype=np.float32)

        if not features.shape[0] == n_features:
            print "Error on image {}".format(img_id)
            features = features[:n_features]

        feature_matrix[i, :] = features

    def get_image(path):
        return np.array(Image.open(path), dtype=np.int32)

        dir = 'test' if test else 'train'
        self.path = os.path.join(root, dir)

        # Get training labels
        self.y = pd.read_csv("data/train_labels.csv", index_col=0)

        self.submission_format = pd.read_csv("data/SubmissionFormat.csv",
                                index_col=0)
        for i, img_id in tqdm(enumerate(y.index)):
            X = preprocess_picture(img_id)
         X = np.vstack(df['Image'].values) / 255.




    def shuffle(self):
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.y = self.y.astype(np.float32)

    def normalize(self):
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)

    def get_params(self):
        return self.X, self.y


def flatten_pic(pic):
    return pic.flatten()

def expand_pic(pic):
    return pic.reshape((200,200,3))