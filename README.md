# bees
Driven data bees classifier competition codes [http://www.drivendata.org/competitions/8/]. 

## Competition description
All pictures are 200x200, RGB, with two classes: Bumble bee or honey bee.
* Train set: 3969 pictures (3142 class 1, 827 class 0)
* Test set: 992 pictures


## Code architecture
* DataManager class takes care of loading and pre-treatment (normalization, shuffle)

# Literature
* Daniel Nouri's [tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) on ConvNets w. lasagne and nolearn for Kaggle Face Keypoints detectors  
* Bee's [benchmark notebook](http://nbviewer.ipython.org/github/drivendata/benchmarks/blob/master/bees-benchmark.ipynb) on DrivenData.org 
* [Lasagne tutorials](http://lasagne.readthedocs.org/en/latest/user/tutorial.html)
* [NoLearn mnist tutorial](http://nbviewer.ipython.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb)
