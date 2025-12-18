#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q a100_devq
#PBS -koed
#PBS -e /dev/null
#PBS -o /dev/null

# Redirect stdout and stderr to a log file
mkdir -p "${PBS_O_WORKDIR}/logs"
exec &> "${PBS_O_WORKDIR}/logs/${PBS_JOBID%.*}-${PBS_JOBNAME%.*}.log"

# check the output directory works (needs fixing I believe)
echo "Starting ERA5 processing job."
cd /home/crowelenn/dev/convcnp-assim-nz
pixi run python -m nbconvert \
  --to notebook \
  --execute "./notebooks/experiment4/experiment4_nzra_target.ipynb" \
  --output "experiment4_nzra_target_executed.ipynb" \
  --output-dir "./notebooks/experiment4/executed/" \
  --ExecutePreprocessor.timeout=-1