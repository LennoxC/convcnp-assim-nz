#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=1500gb
#PBS -l walltime=12:00:00
#PBS -q shortq
#PBS -koed
#PBS -e /dev/null
#PBS -o /dev/null

# Redirect stdout and stderr to a log file
mkdir -p "${PBS_O_WORKDIR}/logs"
exec &> "${PBS_O_WORKDIR}/logs/${PBS_JOBID%.*}-${PBS_JOBNAME%.*}.log"

echo "Running experiment 5, data generation, CPU mode."
cd /home/crowelenn/dev/convcnp-assim-nz

export DEVELOPMENT_ENVIRONMENT="0"
export DATASET_GENERATION="1" # leave this as 1
export PROCESS_YEAR="2016" # (optional) set to a year (e.g., 2013) to only process that year. Comment out or set to 0 if not using.
export EXPERIMENT_NAME="experiment5_nzra_diff_coord_norm_landonly" # experiment name. Used in file paths for checkpointing model + data processor (normalization)

pixi run -e default python -m nbconvert \
  --to notebook \
  --execute "./notebooks/experiment5/experiment5_nzra_diff.ipynb" \
  --output "experiment5_nzra_diff_executed.ipynb" \
  --output-dir "./notebooks/experiment5/executed" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=convcnp-cpu