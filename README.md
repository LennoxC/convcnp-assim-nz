# ConvCNP for Data Assimilation | Earth Sciences NZ x Victoria University of Wellington

Data Assimilation in weather forecasting involves combining a model's previous forecast with new, real-time sensor observations, to create a more accurate picture of the current state of the atmosphere. This is then used for the next numerical weather prediction model run, improving the accuracy of future observations.

In this repository, we experiment with using Convolutional Conditional Neural Processes (ConvCNP) for data assimilation.

### Using this repository

The entry points for this respository are as follows:
- `src.utils.era5.main`: This module defines a command line tool for downloading ERA5 data from the [Climate Data Store](https://cds.climate.copernicus.eu/). 

*There are more entry points to come.*

Each entry point has corresponding documentation in the `docs` folder.

### Environmental Variables

Python's `dotenv` package is used for managing environmental variables. A sample file `.env_template` is included in this repository for your convenience. After cloning this repo, you should rename this file to `.env`, and enter appropriate values for each variable.

#### Required variables:
- `DATA_HOME`: specify the base directory where model training data will be stored.

**Variables required for other entry points are outlined in their corresponding documentation (`/docs`)**
