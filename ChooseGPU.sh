#!/usr/bin/env bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu$1,floatX=float32 python $2
