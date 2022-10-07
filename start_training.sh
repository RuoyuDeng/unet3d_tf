#!/bin/bash


DOCKER_CMD="${DOCKER_CMD} -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/source_data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus $dataset_size"


# echo $DOCKER_CMD
exec $DOCKER_CMD



