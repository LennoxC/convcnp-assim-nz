# ConvCNP for Data Assimilation | Earth Sciences NZ x Victoria University of Wellington

Data Assimilation in weather forecasting involves combining a model's previous forecast with new, real-time sensor observations, to create a more accurate picture of the current state of the atmosphere. This is then used for the next numerical weather prediction model run, improving the accuracy of future observations.

In this repository, we experiment with using Convolutional Conditional Neural Processes (ConvCNP) for data assimilation.

### Notebooks
Experiments are done in the notebooks folder. These are generally commited with results. Running the notebook should re-produce these results. At times, the notebooks folder may contain `debug` or `eda` folders, containing notebooks for these respective purposes.

### Deepsensor
Some modifications have been made to the deepsensor repository. There is a [fork of deepsensor here](https://github.com/LennoxC/deepsensor/tree/main). Pip is configured to import the main branch of this fork in `requirements.txt`.

We anticipate that we may need to fork neuralprocesses in the future too.

### Using this repository

Notebooks can be run with `scripts/run_notebook_nohup.sh`. This automatically launches tensorboard, runs the notebook with nohup (which ignores the disconnect signal so you can power off your laptop while it runs on a GPU server), outputs logs, and saves the output notebook to the `executed` directory at the notebook location. Of course, while developing, you can interactively run notebooks too. This script just allows you to run notebooks in a more production-like environment.

There are also some (or *one* currently) command like utilties:
- `src.utils.era5_loader.main`: This module defines a command line tool for downloading ERA5 data from the [Climate Data Store](https://cds.climate.copernicus.eu/). 

*There are more entry points to come, maybe.*

Each entry point has corresponding documentation in the `docs` folder.

### Environmental Variables

Python's `dotenv` package is used for managing environmental variables. A sample file `.env_template` is included in this repository for your convenience. After cloning this repo, you should rename this file to `.env`, and enter appropriate values for each variable.
