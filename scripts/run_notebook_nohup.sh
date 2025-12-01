#!/usr/bin/env bash

# Usage: ./run_notebook_nohup.sh my_notebook.ipynb
# e.g. from the project root:
# ./scripts/run_notebook_nohup.sh notebooks/experiments/exp2_transfer_learning.ipynb
# or
# ./scripts/run_notebook_nohup.sh notebooks/experiments/exp2_transfer_learning.ipynb 6006 tb_logs 2
#
# This will run the notebook in the background using nohup and output
# the executed notebook to a subdirectory "executed" in the same directory.
# This is useful for long-running experiments on remote servers, while preserving 
# the notebook interface and outputs.

NOTEBOOK="$1"

# Optional: port for tensorboard (default: 6006)
TENSORBOARD_PORT=${2:-6006}

# TensorBoard directory defaults to ./tb_logs/notebook_name
TB_LOG_DIR=${3:-"$PWD/.tb_logs"}

if [ -f .env ]; then
    # Source the .env file
    source ./.env
fi

if [ -z "$NOTEBOOK" ]; then
  echo "Usage: $0 notebook.ipynb [tensorboard_port] [tb_log_dir] [nprocs]"
  exit 1
fi

if [ ! -f "$NOTEBOOK" ]; then
  echo "File not found: $NOTEBOOK"
  exit 1
fi

# Extract directory and base filename
NB_DIR=$(dirname "$NOTEBOOK")
NB_BASE=$(basename "$NOTEBOOK" .ipynb)

# ensure executed output dir exists (e.g. notebooks/experiments/executed)
OUT_DIR="${NB_DIR}/executed"
mkdir -p "$OUT_DIR"
OUTNOTEBOOK="${OUT_DIR}/${NB_BASE}_executed.ipynb"

echo "Running notebook in background using nohup..."
echo "Output notebook: $OUTNOTEBOOK"

# set notebook kernel log file (picked up by setup_logging via NOTEBOOK_LOG_FILE)
# create a per-run timestamped log file in ./logs/
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOG_DIR/${NB_BASE}_$TS.log"
export NOTEBOOK_LOG_FILE="$LOG_FILE"
echo "Notebook kernel logs: $LOG_FILE"
echo "To follow notebook kernel logs: tail -f $LOG_FILE"

# Optional: number of processes for DDP (if >1, will use torch.distributed.run)
# pass as fourth arg: ./run_notebook_nohup.sh notebook.ipynb 6006 .tb_logs 4
NPROCS=${4:-1}

if [ "$NPROCS" -gt 1 ]; then
  echo "Launching DDP notebook execution with $NPROCS processes (torchrun)..."
  nohup /home/crowelenn/niwa/convcnp-assim-nz/venv/bin/python -m torch.distributed.run --nproc_per_node=$NPROCS \
    scripts/torchrun_nb_exec.py --notebook "$NOTEBOOK" \
    > nohup.out 2>&1 &
else
  nohup /home/crowelenn/niwa/convcnp-assim-nz/venv/bin/python -m nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "${NB_BASE}_executed.ipynb" \
    --output-dir "$OUT_DIR" \
    --ExecutePreprocessor.timeout=-1 \
    > nohup.out 2>&1 &
fi

PID=$!
echo "Started background job with PID: $PID"
echo "Monitor progress using: tail -f nohup.out"

# start tensorboard in the background (if not already running)
if lsof -i :"$TENSORBOARD_PORT" >/dev/null ; then
    echo "TensorBoard is already running on port $TENSORBOARD_PORT."
else
    echo "Starting TensorBoard on port $TENSORBOARD_PORT, log dir: $TB_LOG_DIR"
    nohup /home/crowelenn/niwa/convcnp-assim-nz/venv/bin/tensorboard --logdir "$TB_LOG_DIR" --port $TENSORBOARD_PORT > tensorboard_nohup.out 2>&1 &
    TB_PID=$!
    echo "TensorBoard started with PID: $TB_PID"
    echo "To restart TensorBoard later, run:" 
    echo "/home/crowelenn/niwa/convcnp-assim-nz/venv/bin/tensorboard --logdir \"$TB_LOG_DIR\" --port $TENSORBOARD_PORT"
fi
