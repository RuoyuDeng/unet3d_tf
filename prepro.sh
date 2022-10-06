#!/bin/bash

raw_npy_dir="/raid/data/imseg/29gb-npy"
tf_dir="/raid/data/imseg/29gb-tf"
python3 dataset/prepro.py --np-raw ${raw_npy_dir} --tf-data ${tf_dir} --do-crop --do-reshape