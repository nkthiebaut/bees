#!/usr/bin/env bash

pip uninstall -y Lasagne
pip uninstall -y nolearn
pip uninstall -y numpy
pip install numpy
#pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
pip install tqdm
cd data/images
bash download_data.sh
cd ../..