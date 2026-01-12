#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=00:02:00
#PBS -q a100_devq
#PBS -koed
#PBS -e /dev/null
#PBS -o /dev/null

# Redirect stdout and stderr to a log file
mkdir -p "${PBS_O_WORKDIR}/logs"
exec &> "${PBS_O_WORKDIR}/logs/${PBS_JOBID%.*}-${PBS_JOBNAME%.*}.log"

nvidia-smi

cd /home/crowelenn/dev/convcnp-assim-nz
pixi run -e gpu python ./notebooks/experiment4/gpu_test.py

