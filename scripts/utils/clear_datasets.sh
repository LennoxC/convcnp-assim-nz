#!/usr/bin/env bash

DIR_BASE=${1:-"/esi/project/niwa00004/crowelenn/data/pickle/"}

# check the user really wants to delete these files
read -p "Are you sure you want to delete all datasets in ${DIR_BASE}? This action cannot be undone. (y/n) " -n 1 -r
echo    # move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting dataset clearing."
    exit 0
fi

# delete all files in the dataset directory
rm "${DIR_BASE}"/experiment5_nzra_target_train_tasks/*
rm "${DIR_BASE}"/experiment5_nzra_target_val_tasks/*

echo "Datasets cleared from ${DIR_BASE}."