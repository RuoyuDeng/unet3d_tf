# Kill the tmux session from a previous run if it exists
tmux kill-session -t unet3d_tf 2>/dev/null
# Start a new tmux session from which we will run training
tmux new-session -d -s unet3d_tf
tmux send-keys -t dlio_unet "./build_run_docker.sh train" C-m