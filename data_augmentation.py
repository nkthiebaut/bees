#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nolearn.lasagne.base import BatchIterator
from skimage.transform import rotate
from skimage.transform import rescale
import numpy as np

__author__ = 'thiebaut'
__date__ = '13/10/15'

import numpy as np

class DataAugmentationBatchIterator(BatchIterator):
    def transform(self, Xb):
        Xb = super(FlipBatchIterator, self).transform(Xb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        Xb = np.swapaxes(Xb, 1, 3)
        for i in range(bs):
            # Zoom and crop
            scale_factor = 0.5 * np.random.random() + 1.
            Xb[i] = rescale(Xb[i], scale_factor)
            center = (1.*np.random.random()

            # Translate

            # Rotate
            angle = np.random.random_integers(0, 359)
            Xb[i] = rotate(Xb[i], angle, mode='nearest')
        Xb = np.swapaxes(Xb, 1, 3)



        return Xb
