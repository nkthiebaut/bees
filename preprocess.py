#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'thiebaut'
__date__ = '09/10/15'

import matplotlib
matplotlib.use('Agg')
from DataManager import DataManager

import argparse

def GetOptions():
    """ Retrieve options from standard input """
    p = argparse.ArgumentParser(description='Neural net. training',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-n', '--name', metavar="name", type=str, default='',
                   help='Dataset name.')
    p.add_argument('-w', '--width', metavar="width", type=int, default=200,
                   help='Images width (blank for no padding)')
    p.add_argument('-b', '--balanced', action="store_true", dest="balanced", 
help='Whether to over-sample to equalize classes or no')
    args = vars(p.parse_args())
    return args

def prepare_data(name='', width=200, balanced=True):
    DM = DataManager()
    if balanced:
        DM.equalize_classes()
    DM.shuffle()
    DM.normalize()
    if width > 200:
        DM.pad(width)
    DM.save_to_lasagne_format(filename='train'+name+'.npz')

    DM = DataManager(test=True)
    DM.normalize()
    if width > 200:
        DM.pad(width)
    DM.save_to_lasagne_format(filename='test'+name+'.npz')

if __name__ == '__main__':
    args = GetOptions()
    print args
    prepare_data(name=args['name'], width=args['width'], balanced=args['balanced'])
"""
    prepare_data(name='_width_224_balanced', width=224, balanced=True)
    prepare_data(name='_width_224_unbalanced', width=224, balanced=False)
    prepare_data(name='_width_200_balanced', width=200, balanced=True)
    prepare_data(name='_width_200_unbalanced', width=200, balanced=False)
"""
