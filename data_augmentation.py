#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '13/10/15'

import sys
import numpy as np
from math import pi

from skimage.transform import SimilarityTransform
from skimage.transform import AffineTransform
from skimage.transform import warp
from skimage.util import pad
from nolearn.lasagne.base import BatchIterator


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb


class DataAugmentationBatchIterator(BatchIterator):
    def __init__(self, batch_size, crop_size=200, pad_size=100, nb_channels=3, scale_delta=0.2, max_trans=5,
                 angle_factor=1., shear=None):
        """
        Transforms pictures randomly at each batch iteration with horizontal flips, rotations, translations, zooms

        :param batch_size: size of the batch (int)
        :param crop_size: width bof output pictures (int)
        :param pad_size: number of pixels to add on both sides of pictures to avoid black regions (int)
        :param nb_channels: number of channels (=3 for RGB pictures)
        :param scale_delta: delta for zoom scale (scale is in [1-scale_delta, 1+scale_delta]
        :param max_trans: maximum translation (int)
        :param angle_factor: ratio of full rotation to explore (0 for no rotation, 1 for 2\pi)
        :param shear: shear angle in counter-clockwise direction as radians.
        :return: transformed batch
        """
        super(DataAugmentationBatchIterator, self).__init__(batch_size)
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.nb_channels = nb_channels
        self.scale_delta = scale_delta
        self.max_trans = max_trans
        self.angle_factor = angle_factor
        self.shear = shear

    def transform(self, Xb, yb):
        Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        #  Divide pixels values by 255 to make it fit in [-1;1], for SimilarityTransform compatibility
        Xb /= np.float32(255.)

        # Change shape from [Batch_size, nb_channels, width, height] to [Batch_size, width, height, nb_channels]
        Xb = np.swapaxes(Xb, 1, 3)

        # Define relevant shapes
        im_size = Xb.shape[1]
        frame_size = im_size + 2 * self.pad_size
        lower_cut = (frame_size - self.crop_size) / 2
        upper_cut = (frame_size + self.crop_size) / 2
        shift_x = frame_size / 2
        shift_y = shift_x

        # Necessary shifts to allow rotation around center
        tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

        for i in xrange(bs):
            pic = Xb[i]  # Picture as a [width, height, nb_channels] np.array

            # Pad image to avoid black regions after zoom/rotation/translation
            padded = np.zeros((self.nb_channels, frame_size, frame_size))
            for j in xrange(self.nb_channels):
                padded[j] = pad(np.swapaxes(pic, 0, 2)[j], (self.pad_size, self.pad_size), 'reflect')
            padded = np.swapaxes(padded, 0, 2)

            #  Pick random values
            scaling_factor = 2 * np.random.random() * self.scale_delta + (1. - self.scale_delta)
            angle = 2 * pi * (np.random.random() - 0.5) * self.angle_factor
            trans_x = np.random.randint(-self.max_trans, self.max_trans)
            trans_y = np.random.randint(-self.max_trans, self.max_trans)

            # Apply similarity transform to zoom, rotate and translate
            tf = SimilarityTransform(scale=scaling_factor, rotation=angle, translation=(trans_x, trans_y))
            padded = warp(padded, (tf_shift + (tf + tf_shift_inv)).inverse)

            # Crop to desired size
            Xb[i] = padded[lower_cut:upper_cut, lower_cut:upper_cut, :]

        Xb = np.swapaxes(Xb, 1, 3)
        Xb *= np.float32(255.)
        return Xb, yb


class ResamplingBatchIterator(DataAugmentationBatchIterator):
    """
    Batch iterators that initially equalize classes and gradualy decreases balance to reach sample imbalance
    Adapted from https://github.com/sveitser/kaggle_diabetic/blob/master/iterator.py
    """

    def __init__(self, batch_size, max_epochs, dataset_ratio, final_ratio, crop_size=200, 
                 pad_size=100, nb_channels=3, scale_delta=0.2, max_trans=5,
                 angle_factor=1., shear=None, output=sys.stdout):
        super(ResamplingBatchIterator, self).__init__(batch_size, crop_size, pad_size, 
nb_channels, scale_delta, max_trans, angle_factor, shear)
        self.max_epochs = max_epochs
        self.dataset_ratio = dataset_ratio
        self.count = 0
        self.final_ratio = final_ratio
	self.output=output

    def __call__(self, X, y=None, transform=None):
        if y is not None:
            #  Ratio changes gradually from dataset_ratio to 1
            self.count += 1
            ratio = self.dataset_ratio * (self.max_epochs - self.count) + self.final_ratio * (self.count-1)
	    ratio /= float(self.max_epochs-1)
            p = np.zeros(len(y))
            weights = (ratio, 1)
            for i, weight in enumerate(weights):
                p[y == i] = weight
            indices = np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                                       p=np.array(p) / p.sum())
            X = X[indices]
            y = y[indices]
        self.tf = transform
        self.X, self.y = X, y
        #print >> self.output, "Epoch "+str(self.count)+", labels counts: "+str(np.unique(y, return_counts=True)[1])
        return self


class ResamplingFlipBatchIterator(FlipBatchIterator):
    """
    Batch iterators that initially equalize classes and gradualy decreases balance to reach sample imbalance
    Adapted from https://github.com/sveitser/kaggle_diabetic/blob/master/iterator.py
    """

    def __init__(self, batch_size, max_epochs, dataset_ratio, final_ratio, output=sys.stdout):
        super(ResamplingFlipBatchIterator, self).__init__(batch_size)
        self.max_epochs = max_epochs
        self.dataset_ratio = dataset_ratio
        self.count = 0
        self.final_ratio = final_ratio
	self.output=output

    def __call__(self, X, y=None, transform=None):
        if y is not None:
            #  Ratio changes gradually from dataset_ratio to 1
            self.count += 1
            ratio = self.dataset_ratio * (self.max_epochs - self.count) + self.final_ratio * (self.count-1)
	    ratio /= float(self.max_epochs-1)
            p = np.zeros(len(y))
            weights = (ratio, 1)
            for i, weight in enumerate(weights):
                p[y == i] = weight
            indices = np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                                       p=np.array(p) / p.sum())

            X = X[indices]
            y = y[indices]
        self.tf = transform
        self.X, self.y = X, y
        #print >> self.output, "Epoch "+str(self.count)+", labels counts: "+str(np.unique(y, return_counts=True)[1])
        return self
