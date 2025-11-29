#!/usr/bin/env bash

# Usage: ./run_notebook_nohup.sh my_notebook.ipynb
# e.g. from the project root:
# ./scripts/run_notebook_nohup.sh notebooks/experiments/exp2_transfer_learning.ipynb
#
# This will run the notebook in the background using nohup and output
# the executed notebook to a subdirectory "executed" in the same directory.
# This is useful for long-running experiments on remote servers, while preserving 
# the notebook interface and outputs.

NOTEBOOK="$1"

if [ -z "$NOTEBOOK" ]; then
  echo "Usage: $0 notebook.ipynb"
  exit 1
fi

if [ ! -f "$NOTEBOOK" ]; then
  echo "File not found: $NOTEBOOK"
  exit 1
fi

# Extract directory and base filename
NB_DIR=$(dirname "$NOTEBOOK" )
NB_BASE=$(basename "$NOTEBOOK" .ipynb)

# ensure executed output dir exists (e.g. notebooks/experiments/executed)
OUT_DIR="${NB_DIR}/executed"
mkdir -p "$OUT_DIR"
OUTNOTEBOOK="${OUT_DIR}/${NB_BASE}_executed.ipynb"

echo "Running notebook in background using nohup..."
echo "Output notebook: $OUTNOTEBOOK"
echo "nohup log: nohup.out"

# set notebook kernel log file (picked up by setup_logging via NOTEBOOK_LOG_FILE)
# create a per-run timestamped log file in ./logs/
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOG_DIR/${NB_BASE}_$TS.log"
export NOTEBOOK_LOG_FILE="$LOG_FILE"
echo "Notebook kernel logs: $LOG_FILE"
echo "To follow notebook kernel logs: tail -f $LOG_FILE"
nohup /home/crowelenn/niwa/convcnp-assim-nz/venv/bin/python -m nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "${NB_BASE}_executed.ipynb" \
  --output-dir "$OUT_DIR" \
  --ExecutePreprocessor.timeout=-1 \
  > nohup.out 2>&1 &

PID=$!
echo "Started background job with PID: $PID"
echo "Monitor progress using: tail -f nohup.out"
