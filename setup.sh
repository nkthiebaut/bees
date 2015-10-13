#!/usr/bin/env bash

pip uninstall -y Lasagne
pip uninstall -y nolearn
pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
pip install tqdm
cd data/images
bash download_data.sh
cd ../..