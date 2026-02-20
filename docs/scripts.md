# Scripts

A collection of bash scripts to automate various tasks. Here we walk through what each script is for.

## Notebooks
These are scripts for running the experiments as jobs on the HPC as PBS scripts. All of these are similar: there are different scripts for each of experiment 4 and experiment 5, both of which can be run in datagen (dataset generation) or train mode. Use the corresponding scripts depending on what you're trying to do!

The only parts of these scripts you will need to modify are:
- The PBS parameters at the top which specify the queue, walltime, compute units etc.
- The `DEVELOPMENT_ENVIRONMENT` variable. You may want to set this to "1" (true) occasionally, when debugging on the PBS queue - especially when a GPU is required.
- The `EXPERIMENT_NAME` variable, which specifies where the normalization parameters are saved, relative to the `MODEL_DIR` variable in the .env file.

When an experiment has compelted successfully, the exectued notebook is saved into the notebooks directory, with all the outputs included.

## Utils/*

- `clear_datasets.sh` is a helper script to remove the contents of a datasets directory. I found while developing the dataset saving/reading logic, I was clearing the contents of this directory often, so wrote this script to speed up that process. Now I would reccomend just using `rm -r <PATH>` to clear the dataset. You can remove the whole folder rather than just removing the contents, like this helper script does.

- `run_era5_processor.sh` triggers the `era5_subset.py` script which converts multi-file ERA5 data into a zarr.

- `run_himawari8_proessor.sh` does the same as `run_era5_processor.sh`, except it triggers the `himawari8_subset.py` script.

- `start_tensorboard.sh` starts tensorboard with nohup. For more information, see the `tensorboard.md` documentation. If you're here just looking for the command to view my previous experiments, then here it is: `./scripts/utils/start_tensorboard.sh runs 6006 /home/crowelenn/dev/convcnp-assim-nz/.tensorboard/experiment4`

- `run_notebook_nohup.sh` was used when running jobs on the university GPU servers. This is no longer needed when running experiments on the HPC.