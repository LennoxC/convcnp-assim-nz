# Running notebooks

Experiments - i.e. modifications to the model & data, were conducted within notebooks. While debugging, the notebooks can be run against interactive compute. Some notebooks will have different settings when run in "development" mode. Development mode is set by an environmental variable. This defaults to "true" (or, more precisely, the notebooks set a python variable to true if there is no development environmental variable present). When running via a pbs script, you may want to set this environmental variable to false. Typically, running in development mode reduces the dataset size and model complexity, which is desireable when running on interactive compute. How dev mode affects dataset size is set up at the top of each notebook. Inside the `if development_mode:` function, override any hyperparameters you see fit.

### Experiments 1, 2, and 3

These experiments were conducted on the victoria university GPU servers, with limited datasets (which were downloaded locally). I did attempt to make these backwards compatible so they can also run on the NIWA HPC, but some minor code changes might be required. See the `data_api.md` documentation for more details, especially around the `use_absolute_filepaths()` function which was created for backwards compatibility purposes.

In experiment 1, some basic testing was done on small ConvNP's, mainly to familiarise myself with deepsensor. Looking at this code could be a useful starting point to understand the deepsensor library, but nothing groundbreaking was achieved here! 

Experiment 2 was performed to understand if transfer learning could be used along with the deepsensor library. The results are documented in `experiment2/exp2_adding_channels.ipynb`.

Experiment 3 was a quick investigation into what ramifications missing data in one channel has on the rest of the context set. The problem + workaround are explained in the noteook.

### Experiments 4 & 5

These experiments were conducted on the NIWA HPC. Within each experiment, multiple configurations of the model & dataset were tested. While lots of hyperparameter tuning took place, this was largely ineffective. There were a few key architectual milestones which were the major indicators of progress. These are detailed in `experiments.md`.

Experiment 4 largely revolved around the initial set-up of the ConvNP for temperature prediction. Multi-variate prediction (e.g. wind_u and wind_v) is also supported. In this experiment, we found that edge effects (likely due to zero-padding in the convolution kernels) causes instabilities, and a negative feedback loop where the majority of the loss is at the edges of the prediction grid. This led to the introduction of the 'loss masking' around the edges of the grid, which required modifications to deepsensor. This made for stable prediction, which collapsed to the mean. All these iterations (including plenty of hyperparameter tuning) were conducted through modifications to the experiment 4 notebook. Most changes are backwards compatible, meaning that experiments can be reproduced by modifying hyperparameters (listed at the top of the notebook) or changing dataset structures. 

Experiment 5 shifted from temperature prediction, to prediction of the ERA5-NZRA temperature residual. At first when predicting this residual, the model collapsed to the mean bias - i.e. it found the system errors in ERA5, but didn't improve its prediction from here (`experiment5_nzra_diff.ipynb`). This is comparable to the collapse to mean observed in experiment 4. Next, the per-coordinate normalized ERA5-NZRA residual was predicted. The model responded by just predicting 0 everywhere - a collapse to mean, considering the dataset had already been normalized per coordinate. This led to a seperate investigation, about weather the station observations were closely correlated to the NZRA temperature. More details can be found at the top of the `experiment5_stations_target.ipynb` notebook. This analysis was performed in `experiment5_stations_nzra_eda.ipynb`. After determining that station temperatures have a weak relationship with NZRA temperature (after per-coordinate normalization of the ERA5 residual), an additional experiment to substitute NZRA temperature for station temperature while learning was set up in `experiment5_stations_target.ipynb`. This experiment is ongoing.

#### Running Experiments 4 & 5

There are two important filepaths to set when running the notebooks. These are found in the .env file. I organized the paths like this:

```
DATASET_PICKLE_PATH = /esi/project/niwa00004/<USER>/data/pickle/<descriptive_name>/
MODEL_DIR = /esi/project/niwa00004/<USER>/data/model/<descriptive_model_name>/
```

`DATASET_PICKLE_PATH` determines where the pickled dataset is saved to. As you might want to do multiple experiments (i.e. train multiple models) on the same dataset, this is seperate to the `MODEL_DIR` path. Note that pickled datasets can get huge - hundreds of gigabytes for multi-year hourly datasets. Be careful!

`MODEL_DIR` controls where the model parameters are saved, when model checkpointing occurs. This has been of limited concern so far, as no models have been successful enough to warrent saving... however this was tested in experiment 2 (transfer learning). More importantly, an additional path is formed by appending the environmental variable `EXPERIMENT_NAME` (which is set in the experiment_5_datagen scripts). Inside this path, the normalization parameters are saved between runs. While running in a notebook against interactive compute, `EXPERIMENT_NAME` is set to "default_experiment". Then in the datagen/train scripts, you set `EXPERIMENT_NAME` to something different. As the dataset is often filtered to a subset of datetimes in dev mode, you want to store the normalization parameters in different places, hence this extra path. If you want to use previously defined normalization parameters, make sure that the `MODEL_DIR` and `EXPERIMENT_NAME` haven't changed since you fit the data processors (both deepsensor and custom).

Notebooks can be run in 'dataset generation' or 'model training' mode. **Detailed instructions on how to run these notebooks can be found at the top of `experiment5_nzra_diff.ipynb`**, which explains how the python variables for development mode and dataset generation mode work. As there are too many tasks to store in an in-memory python array, tasks are pickled and saved to the disk in dataset generation mode. Then in model training mode, dataset generation is skipped, and tasks are read from the disk and used to train the model. The mode is controlled via an environmental variable, and all the code for both modes sits in the notebooks. You can debug in dataset generation mode or training mode, but model training with interactive compute will likely crash the kernel.