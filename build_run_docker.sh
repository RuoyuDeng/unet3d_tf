#!/bin/bash

sudo docker build -t unet3d_tf:test .
mode=$1

result_path="/raid/data/imseg/results"
tf_data_path="/raid/data/imseg/29gb-tf"

if [[ "$mode" == "train" ]]
then
    sudo docker run --runtime=nvidia -it  --rm --ipc=host \
    -v ${tf_data_path}:/data -v ${result_path}:/results \
    unet3d_tf:test /bin/bash scripts/unet3d_train_single.sh 8 /data /results 2
elif [[ "$mode" == "container" ]]
then
    sudo docker run --runtime=nvidia -it  --rm --ipc=host \
    -v ${tf_data_path}:/data -v ${result_path}:/results \
    unet3d_tf:test /bin/bash
fi

