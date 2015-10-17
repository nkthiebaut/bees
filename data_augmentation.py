#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nolearn.lasagne.base import BatchIterator

__author__ = 'thiebaut'
__date__ = '13/10/15'

import numpy as np

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb):
        Xb = super(FlipBatchIterator, self).transform(Xb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb

class RotateBatchIterator(BatchIterator):
    def transform(self, Xb):
        pass

class ZoomBatchIterator(BatchIterator):
    def transform(self, Xb):
        pass

class TranslateBatchIterator(BatchIterator):
    def transform(selfs,Xb):
        pass
