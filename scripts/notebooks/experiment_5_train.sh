#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=100gb:ngpus=1
#PBS -l walltime=72:00:00
#PBS -q a100q
#PBS -koed
#PBS -e /dev/null
#PBS -o /dev/null

# Redirect stdout and stderr to a log file
mkdir -p "${PBS_O_WORKDIR}/logs"
exec &> "${PBS_O_WORKDIR}/logs/${PBS_JOBID%.*}-${PBS_JOBNAME%.*}.log"

echo "Running experiment 5, training, GPU mode."
cd /home/crowelenn/dev/convcnp-assim-nz

export DEVELOPMENT_ENVIRONMENT="0"
export DATASET_GENERATION="0" # leave this as 0

export EXPERIMENT_NAME="experiment5_nzra_diff"

pixi run -e gpu python -m nbconvert \
  --to notebook \
  --execute "./notebooks/experiment5/experiment5_nzra_diff.ipynb" \
  --output "experiment5_nzra_diff_executed.ipynb" \
  --output-dir "./notebooks/experiment5/executed" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=convcnp-gpu