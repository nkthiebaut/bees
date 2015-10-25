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
        super( FileInfo, self ).__init__()
        self.crop_size = crop_size

    def transform(self, Xb):
        Xb = super(FlipBatchIterator, self).transform(Xb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        Xb = np.swapaxes(Xb, 1, 3)
        im_size = Xb.shape[1]
        lower_cut = (im_size - self.crop_size)/2
        upper_cut = (im_size + self.crop_size)/2
        for i in range(bs):
            # Apply similarity transform to zoom, rotate and translate
            scaling_factor = 0.2 * np.random.random() + 0.9
            angle = 2 * pi * np.random.random()
            tf = SimilarityTransform(scale=scaling_factor, rotation=angle, translation=(im_size, im_size))
            Xb = warp(Xb, tf)

            # Crop to desired size
            Xb[i] = Xb[i,:,lower_cut:upper_cut, lower_cut:upper_cut]
        Xb = np.swapaxes(Xb, 1, 3)
        return Xb
