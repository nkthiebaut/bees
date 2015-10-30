#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import make_submission_file
from utils import load_numpy_arrays
from datetime import date
import cPickle
import sys

conv_net = cPickle.load(open(str(sys.argv[1]),'rb'))

# ----- Test set ----
X_test, _, images_id_test = load_numpy_arrays('test.npz')
print "Test:"
print "X_test.shape:", X_test.shape
predictions = conv_net.predict_proba(X_test)
make_submission_file(predictions, images_id_test, output_filepath='submissions/submission_'+str(date.today)+'.csv')

