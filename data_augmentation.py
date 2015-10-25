#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '13/10/15'

from nolearn.lasagne.base import BatchIterator
import numpy as np
from math import pi
from skimage.transform import SimilarityTransform
from skimage.transform import warp


class DataAugmentationBatchIterator(BatchIterator):
    def __init__(self, crop_size=128):
        super(DataAugmentationBatchIterator, self).__init__()
        self.crop_size = crop_size

    def transform(self, Xb):
        Xb = super(DataAugmentationBatchIterator, self).transform(Xb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        Xb = np.swapaxes(Xb, 1, 3)
        im_size = Xb.shape[1]
        lower_cut = (im_size - self.crop_size)/2
        upper_cut = (im_size + self.crop_size)/2
        shift_x, shift_y = np.array(im_size)/2
        tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
        for i in range(bs):
            # Apply similarity transform to zoom, rotate and translate
            scaling_factor = 0.2 * np.random.random() + 0.9
            angle = pi * (np.random.random()-1.)/8
            trans_x = np.random.randint(-5, 5)
            trans_y = np.random.randint(-5, 5)
            tf = SimilarityTransform(scale=scaling_factor, rotation=angle, translation=(trans_x, trans_y))
            Xb[i] = warp(Xb, (tf_shift + (tf + tf_shift_inv)).inverse)

            # Crop to desired size
            Xb[i] = Xb[i, :, lower_cut:upper_cut, lower_cut:upper_cut]
        Xb = np.swapaxes(Xb, 1, 3)
        return Xb
