#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=01:00:00
#PBS -q shortq
#PBS -koed
#PBS -e /dev/null
#PBS -o /dev/null

# Redirect stdout and stderr to a log file
mkdir -p "${PBS_O_WORKDIR}/logs"
exec &> "${PBS_O_WORKDIR}/logs/${PBS_JOBID%.*}-${PBS_JOBNAME%.*}.log"

echo "Starting ERA5 processing job."
cd /home/crowelenn/dev/convcnp-assim-nz
pixi run python ./src/convcnp_assim_nz/utils/era5_subset.py