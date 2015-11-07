#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '09/10/15'

from DataManager import DataManager


def prepare_data(name='', width = 224, balanced=True):
    DM = DataManager()
    if balanced:
        DM.equalize_classes()
    DM.shuffle()
    DM.normalize()
    if width > 200:
        DM.pad(224)
    DM.save_to_lasagne_format(filename='train'+name+'.npz')

    DM = DataManager(test=True)
    DM.normalize()
    if width > 200:
        DM.pad(224)
    DM.save_to_lasagne_format(filename='test'+name+'.npz')

if __name__ == __main__:
    prepare_data(name='_width_224_balanced', width=224, balanced=True)
    prepare_data(name='_width_224_unbalanced', width=224, balanced=False)
    prepare_data(name='_width_200_balanced', width=200, balanced=True)
    prepare_data(name='_width_200_unbalanced', width=200, balanced=False)