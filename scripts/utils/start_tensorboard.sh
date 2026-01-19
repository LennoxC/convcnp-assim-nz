#!/usr/bin/env bash

echo "Starting TensorBoard."
cd /home/crowelenn/dev/convcnp-assim-nz

# Usage: ./start_tensorboard.sh [subdir] [port] [log_dir]
SUBDIR=${1:-""}
PORT=${2:-6006}
LOG_DIR=${3:-"$PWD/.tensorboard"}

# ensure log directory exists
mkdir -p "$LOG_DIR/$SUBDIR"

if lsof -i :"$PORT" >/dev/null ; then
    echo "TensorBoard is already running on port $PORT."
    exit 0
fi

# start tensorboard and record the nohup process id
nohup pixi run tensorboard --logdir "$LOG_DIR/$SUBDIR" --port "$PORT" > tensorboard_nohup.out 2>&1 &

# return the PID of the tensorboard process
TB_PID=$!

echo "TensorBoard started on port $PORT with log dir $LOG_DIR/$SUBDIR. PID = $TB_PID"