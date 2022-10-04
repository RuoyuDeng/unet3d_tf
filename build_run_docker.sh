#!/bin/bash

sudo docker build -t unet3d_tf:test_ry .
mode=$1

if [[ "$mode" == "train" ]]
then
    sudo docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm --ipc=host \
    -v ${PWD}/data:/data -v ${PWD}/results:/results -v /raid/data/imseg/preproc-data:/preproc-data \
    unet3d_tf:test_ry /bin/bash scripts/unet3d_train_single.sh 1 \
    ../DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical/data /results 2
elif [[ "$mode" == "container" ]]
then
    sudo docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm --ipc=host \
    -v ${PWD}/data:/data -v ${PWD}/results:/results -v /raid/data/unet/unet3d_tf/npy_data:/preproc-data \
    unet3d_tf:test_ry /bin/bash
fi

