# The dataset API

This documentation relates to things you may find in the `src/convcnp_assim_nz` directory. 

At a high-level there are four directories:
- config: manages the dotenv file and logging.
- data_processing: the API to all the datasets, including converters for unit conversion, the custom per-coordinate normalizer, and generation of auxiliary variables (sun stuff).
- learning: model training related things, like training loops, batching etc.
- utils: random things like the era5 loader, variable names, cartopy.

I will explore each of these directories below. There are also comments in the code to help.

### config
**env_loader.py**: When this file is imported, the dotenv file is loaded. Thus it is important to import this regardless of weather or not you want to use the `get_env_var` helper function or not. `get_env_var` is a pretty standard helper function to return a variable from the dotenv file (or environmental variable) also allowing you to specify a default, and a flag on weather or not the default was used. `use_absolute_filepaths` is also defined in here. This mainly exists for backwards compatibility purposes. To use the data api on the HPC, you will need to call `use_absolute_filepaths`. Just calling the function is all you need to do - this function sets an environmental variable which signals to the rest of the data api that you should use absolute filepaths to access the datasets. When running on the university GPU servers, all the data was kept nice and tidy in a single directory, so could be accessed with relative filepaths. Obviously, this is not the case on the HPC. It is unlikely that we will return to training on the university GPU servers, so this could (and should) be removed.

**logging_config.py**: Sets up the logger. call `setup_logging()` at the top of a notebook. Then you can call `logger = logging.getLogger(__name__)` to get a logger object. By default, this writes to `./src/.logs.out`.

### data_processing

This is a big module. In general, there is one class per dataset. Each class follows the same principles of design and usage. For example, `era5_processor.py` defines the `ProcessERA5` class. Each class initializes a 'file loader' behind the scenes. These file loaders are kept in the `file_loaders` subdirectory. The idea is that if a new source for the same dataset is created, you can just create a new file loader which allows the processor to read in the files, without having to re-invent the entire data processor. Each file loader class is a bit different due to the different file formats etc.

I am not going to step through the ins and outs of each of these classes, as most of them are largely the same. Examples of how to use them are concisely displayed in the experiment4 & experiment5 notebooks. It is useful to have an idea of what is done within each data processor though. In general the same process is followed within each processor:

- find the files and load them into memory. The files might be lazy-loaded (netcdf). This will give a raw xarray/pandas object.
- filter to a subset of years (if asked for by the user).
- convert the variables and coordinates to the standardized format. This uses hard-coded variable names from the source, and maps them to the string variables defined in `convcnp_assim_nz.utils.variables.var_names` (and `coord_names`). You may need to edit this if you do some dataset surgery, like adding new variables etc.
- optinally define some helper methods.
- return the dataset.

To use, create a processor, then call to load the dataset. For example:

```
nzra_processor = ProcessNZRA()
nzra_ds = nzra_processor.load_ds(years=years)

# now nzra_ds is a netcdf dataset for you to process further.
# all the required filesystem logic is hidden away.
```

Here, a list of years was passed into the load call, to automatically filter the dataset to a subset of time. This is useful when loading in a smaller subset of time in dev mode.

The biggest outlier in this pattern is the stations loader. This is a difficult one as on the university servers, I was using a multi-file netcdf dataset. On the HPC, stations are stored in a .csv file. Hence, you have to specify: `station_processor = ProcessStations(mode="csv")`. You also need to specify the csv file for the variable you are accessing. For example: 

```
if TEMPERATURE in target_variable:
    csv_file = '2013-2018_temps.csv'

stations_df = station_processor.load_df(target_variable, csv_file=csv_file, year_start=min(years), year_end=max(years))
```

Note that the filesystem paths etc are defined in the .env file. These are required by the file loaders. Please read the **env_loader.py** documentation above as this is important.

#### `normalize/coordinate_normalizer.py` 
The deepsensor normalizer doesn't do per-coordinate normalization. I got onto this near the end of my fixed-term, so didn't want to invest too much time in updating the deepsensor normalizer to include this, so made my own module instead. However, my module works near identically to deepsensor, in terms of the way it is called, fit etc. So hopefully merging the two one day isn't too difficult! To use it, create and fit a coordinate normalizer as follows:

```
normalizer = CoordinateNormalizer()

if normalizer.try_load(custom_norm_dir):
    normalizer.load(custom_norm_dir)

stations_norm, nzra_ds_stations_norm, nzra_ds_norm = normalizer(
        [stations_era5[['residual', TIME, LATITUDE, LONGITUDE]], 
        nzra_ds_stations[[f"{var}_station" for var in target_variable_diff]], 
        nzra_ds[target_variable_diff]]
    )

normalizer.save_if_not_loaded(custom_norm_dir)
```

`try_load` will attempt to load the normalization parameters, and return True if successful, False if not. It is important to fit the normalization parameters once, and use these same parameters at test/prediction time. This is especially important when climate change is increasing the mean temperature over time...

In the background, this normalizer saves .nc files for the mean/std of each variable it is fit to, and does the same for .csv files with a lat/lon column. It is quite interesting to plot the normalization parameters to see what the biases are! If the normalizer is not initialized with parameters from a previous fit, then the mean/std are calculated in the normalizer(...) call. Finding the parameters is skipped in the normalizer(...) call if mean/std parameters are already loaded.

**caution**: I encountered a confusing bug in this module where I tried to fit two different datasets which had the same variable name. This caused issues as under-the-hood, this module forms a dictionary of variable name to normalization parameters. Hence, multiple datasets with the same parameter name cause it to fail. I got around it by renaming the variable to something else, but this should be fixed in the future, because this could happen often. It is not uncommon to normalize 'temperature' on two different datasets!

### learning

**model_diagnostics.py**: There are some basic functions to print out the model structure, count parameters etc. Useful for diagnostics and debugging, but not required for learning. Be aware that rather than just passing your deepsensor `model` into these functions, you may need to pass in `model.model` - which accesses the neuralprocesses model within the deepsensor model.

**nps_unet.py**: This was created for transfer learning. `copy_unet_weights_except_first` is designed to copy UNet weights from one model to another, excluding the first layer, as this layer's dimensionality is determined by the task structure (i.e. number of channels). When training a model where you have performed transfer learning via weight copying, it may be desireable to first only train the 'untrained' layer first, before beginning to train the rest of the unet. That is what `freeze_unet_except_first` is for.

More details are available in `experiments.md`: Experiment 2 (transfer learning).

**model_training.py** contains a few miscellaneous functions for model training and evaluation. A few of these functions are probably good candidates to be built into the deepsensor fork. When designing a new experiment, most of the work is normally done in this file - as lots of the 'heavy lifting' deep learning stuff is done in here.

A note on **pickling**: this was done because most of the time when dealing with massive datasets, they can't fit in memory (like deepsensor is designed to do). Hence the tasks are pickled to the disk, and only loaded **within the training or validation loop**. Hence, the training and validation functions need to be updated to support this.

- `compute_val_loss()`: copied across from deepsensor. This local copy is not used as far as I'm aware.
- `compute_val_loss_pickled()`: Computes the validation loss from pickled tasks. Validation tasks are kept in a different directory to training tasks to avoid data leakage. This resembles the training function closely, just doesn't back-propogate!

- `train_epoch_pickled()`: The main training function. This trains for an entire epoch. There is a function nested inside this one to train for one batch. Note the following parameters:
    - `model`: the model to be updated.
    - `train_task_dir`: the directory containing all the tasks which you want to train on.
    - `opt`: the optimizer. I have been using Adam with learning rate scheduling (learning rate scheduling is handled in the notebook).
    - `batch_size`: batch size integer.
    - `epoch`: to keep track of what epoch we are on. Used for generating random seeds to shuffle the dataset.
    - `use_grad_clip`: we were having model instability issues early on in development. But after masking edge loss, we aren't needing gradient clipping anymore. I reccomend leaving this off (False).
    - `grad_clip_value`: the value for gradient clipping if using. I used 1.0.
    - `grad_accum_steps`: how many batches to accumulate gradients over. Used to simulate a larger batch size if you can't fit an adequate batch size in vram.
    - `fix_noise`: when faced with LinAlgError: negative cholesky... you can try to fix the noise to combat this. It was caused by negative eigenvalues or something, but fix_noise adds a small positive value to the diagonal to prevent this (I think). I believe I used 1e-3 as a ballpark value, but I think a better fix is reducing the learning rate, using loss-masking around the edges of the prediction grid, and using a sample of points rather than all the points.
    - `n_subsample_targets`: how many points to randomly subsample in the target set. This may be good regularization, and was used in lots of my experiments. Leaving this `None` disables target subsampling.

There are a few additional features that I added, but this function operates similarly to the deepsensor implementation. Note that batching only works when there are the same number of targets in each target set. This was an issue when training on a set of station observations, but is manageable when predicting on a grid. Target subsampling via `n_subsample_targets` helps as it guarentees the same number of target points in each task. The function `batch_data_by_num_stations()` can be used to group data into batches based on the number of targets, if required. This wasn't used in experiment 4 or experiment 5.

`concat_tasks_custom()`: Concatenate a list of tasks into a single task containing multiple batches. A custom function was required only because of this line: `merged_task["Y_t_aux"] = B.concat(*[t["Y_t_aux"] for t in tasks])` which was added because the deepsensor function didn't include the auxiliary set in each task. This could be moved to the deepsensor fork.

`return_sample_predictions()`: This is probably the most problematic function in the whole repository. Each experiment demands slightly different visualization of tasks, so this function needs to be redesigned bascially every time. Furthermore, there are many deepsensor objects which are required to perform inference tasks, so all of these need to be passed in, along with all the datasets, creating many arguments. The function has to perform inference + plotting, so there is a lot going on. In `experiment5_stations_target.ipynb` I included some commented out code for testing this function. It is usually too complex to test and get right within the training loop, so I would reccomend debugging it before an experiment. Within the training loop, this function is typically called twice, once for training data and once for validation data. The resulting images are logged to tensorboard, and are very useful for evaluating model performance. Due to the nature of this task, the function is not designed to be backwards-compatible with other experiments - it is simply too complicated. Other apporaches such as just defining this function in the experiment notebook could be considered. I'm not sure if it makes sense to define this function in the API section of the repo considering it changes each experiment, but it is also very large and clunky to have in a notebook. If you need to find previous versions of this function, the github commit history is probably your best bet. At one point I created `stations_sample_predictions()` which does a similar task - I think I was running a couple of different experiments and needed both for a while. I have left them both as two different examples. Both might be useful to copy/paste code snippets from.

`subsample_targets()`: Used to subsample a random set of values from the target set. This is called within the training loop if `n_subsample_targets` is set. This could be moved into the deepsensor repository as it is quite useful + might provide really good regularisation.

### utils

`/era5_loader/*` contains a python module which downloads ERA5 data from the climate data store API. This is saved as NETCDF files. This was useful when using the university GPU servers, but isn't needed on the HPC with ERA5 data already downloaded. I wrote this based off a script which Emily provided me when starting work on this project. More information about using this is available in `era5_loader.md`.

`era5_subset.py` is a helper script to take a multi-file ERA5 dataset, and convert it to a zarr. I used this while prototyping a model, but now read directly from the multi-file ERA5 dataset. This was a one-off task but you can modify the script as required if you need to do something similar.

`himawari8_subset.py` is another helper script to take multi-file netcdf file and convert it to a zarr. 

`install_cartopy_coastlines.py` is a helper script which you might need to use to execute experiment 4 and experiment 5. Cartopy is a python library for plotting coastlines. Usually, cartopy will install the required coastlines when it runs (over the network). This doesn't work on HPC nodes when there is no network connection. Therefore you need to run this script on the login node to cache the required cartopy data to `$CARTOPY_FILES_DIR`. From memory, this script takes a couple of minutes to run. You need to install this for the receptive field plotting in the experiment 4 & 5 notebooks. It seems cartopy is a requirement to use this deepsensor function.

