# ConvCNP for Data Assimilation | Earth Sciences NZ x Victoria University of Wellington

Data Assimilation in weather forecasting involves combining a model's previous forecast with new, real-time sensor observations, to create a more accurate picture of the current state of the atmosphere. This is then used for the next numerical weather prediction model run, improving the accuracy of future observations.

In this repository, we experiment with using Convolutional Conditional Neural Processes (ConvCNP) for data assimilation. If a ConvCNP model can speed up data assimilation, it will allow for near-real-time nowcasting, updating continuously as sensor readings arrive. This will be useful for the renewable energy industry, emergency management, and any other industry which requires a live view of current weather conditions across the country.

### Documentation

I'd reccomend reading the documentation in the `./docs` folder. Start with `update-feb-2026` and go from there. 

The following documentation is available:
- **data_api.md**: Explains the codebase which the notebooks leverage heavily
- **era5_loader.md**: Explains how to use the ERA5 loader utility. This isn't very relevant when on the HPC with ERA5 data already downloaded.
- **experiments.md**: Explains the purpose of each notebook in the notebooks directory.
- **notebooks.md**: Explains the notebooks directory, where experiments were being run.
- **pixi.md**: Useful notes on the pixi environment, running with interactive compute in VSCode, submitting jobs to the HPC etc. I'd reccomend reading this one.
- **tensorboard.md**: Starting tensorboard and viewing the experiments we have completed already.
- **scripts.md**: The scripts directory contains various bash scripts. This documentation explains what each script is for.

### Environmental Variables

Python's `dotenv` package is used for managing environmental variables. A sample file `.env_template` is included in this repository for your convenience. After cloning this repo, you should rename this file to `.env`, and enter appropriate values for each variable.
