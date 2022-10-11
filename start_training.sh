#!/bin/bash


result_path="/raid/data/imseg/results"
tf_data_path="/raid/data/imseg/29gb-tf"

DOCKER_CMD="${DOCKER_CMD} -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/source_data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d_tf:test /bin/bash scripts/unet3d_train_single.sh 1 $num_gpus $dataset_size"


# echo $DOCKER_CMD
exec $DOCKER_CMD






if [[ "$mode" == "train" ]]
then
    sudo docker run --runtime=nvidia -it  --rm --ipc=host \
    -v ${tf_data_path}:/data -v ${result_path}:/results \
    unet3d_tf:test /bin/bash scripts/unet3d_train_single.sh 8 /data /results 2