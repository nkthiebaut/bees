#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '01/10/15'

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from PIL import Image

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
        print 'Warning: Image ' + str(img_id) + ' is RGBA (alpha), converting to RGB.'
        pixels = pixels[:, :, :3]
    return pixels.flatten()

def make_predictions(output_filepath="submission"+str(date.today())+".csv"):
    prediction_df = submission_format_df.copy()

    # create features
    test_features = create_feature_matrix(submission_format_df)
    test_features_stand = ss.transform(test_features)
    test_features_pca = pca.transform(test_features_stand)

    # predict with the best estimator from the grid search
    preds = gs.best_estimator_.predict_proba(test_features_pca)

    # copy the predictions to the dataframe with the ids and
    # write it out to a file
    prediction_df.genus = preds[:, 1]
    prediction_df.to_csv(output_filepath)

    return prediction_df