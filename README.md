# bees
Driven data bees classifier competition codes [http://www.drivendata.org/competitions/8/]. 

## Installation
<code> git clone https://github.com/wajsbrot/bees </code>
Run setup.sh <code> sudo bash setup.sh </code>
Optional: Install [CudNN](https://developer.nvidia.com/cudnn) for faster training (you have to register):
+ check Cuda version is > 7.0: <code> cvnn --version </code>
+ unpack: <code>tar zxvf cudnn-7.0-linux-x64-v3.0-prod.tgz</code>
+ put files in your cuda directory (/usr/local/cuda or /usr/local/cuda-7.0 on linux)

## Competition description
All pictures are 200x200, RGB, with two classes: Bumble bee or honey bee.
* Train set: 3969 pictures (3142 class 1, 827 class 0) (divisors: 1, 3, 7, 9, 21, 27, 49, 63, 81, 147, ...)
* Test set: 992 pictures


## Code architecture
* DataManager class takes care of loading and pre-treatment (normalization, shuffle)

# Literature
## Bees
* Bee's [benchmark notebook](http://nbviewer.ipython.org/github/drivendata/benchmarks/blob/master/bees-benchmark.ipynb) on DrivenData.org
 
## NoLearn and Lasagne 
* Daniel Nouri's [tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) on ConvNets w. lasagne and nolearn for Kaggle Face Keypoints detectors
* [NoLearn mnist tutorial](http://nbviewer.ipython.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb)
* [NoLearn Documentation](https://github.com/dnouri/nolearn)
* [Lasagne tutorials](http://lasagne.readthedocs.org/en/latest/user/tutorial.html)
* [Team o_O code](https://github.com/sveitser/kaggle_diabetic) for diabetic rethinopathy Kagggle competition (3rd place using lasagne)
