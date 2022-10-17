#!/bin/bash


tf_dir="/raid/data/imseg/29gb-tf"
raw_npy_dir="/raid/data/imseg/29gb-npy"
prep_npy_dir="${raw_npy_dir}-prep"


# if npy-prep directory exists, we skip crop and shapping, generate tf-files immediately
if [ ! -d "${prep_npy_dir}" ]
then
    sudo python3 dataset/prepro.py --np-raw ${raw_npy_dir} --tf-data ${tf_dir} --do-crop --do-reshape
else
    # echo $raw_npy_dir
    sudo python3 dataset/prepro.py --np-raw ${raw_npy_dir} --tf-data ${tf_dir}
fi
