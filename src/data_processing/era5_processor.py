import os
from typing import Literal, List
import glob
import dask
import pandas as pd
import xarray as xr
from datetime import datetime

from src.utils.variables.var_names import *
from src.utils.variables.coord_names import *
from src.data_processing.file_loaders.era5_fileloader import ERA5FileLoader
from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var

class ProcessERA5(DataProcess):
    file_loader: ERA5FileLoader = None

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = ERA5FileLoader()

    # loading from the file system is abstracted to the file loader
    # different file structures can be handled there without changing this class (changes would be made only in the file loader)

    def load_ds(self, 
                mode: Literal['surface', 'pressure'],
                years: List=None,
                standardise_var_names: bool=True,
                standardise_coord_names: bool=True,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            years (list): specific years, retrieves all if set to None
        """

        # load with the file loader
        ds = self.file_loader.load_era5_dataset(mode, years)

        # rename variables and coordinates to standard names if specified (default is True)
        if standardise_var_names:
            ds = self.rename_variables(ds) # standardise variable names

        if standardise_coord_names:
            ds = self.rename_coords(ds) # standardise coordinate names

        return ds
    
    # DataSet (ds) to DataArray (da) where the dataarray is a single variable (specified by var)
    def get_variable(self, ds, var) -> xr.DataArray:
        """
        Filter ERA5 data to a single variable, or list of variables.
        Preserves all other dimensions (time, level, lat, lon).
        """
        return ds[var]
    
    def rename_variables(self, ds) -> xr.Dataset:
        """
        Rename variables in the dataset to standardised names.
        The standard names are defined as strings in src/utils/variables/var_names.py
        """
        rename_dict = {}
        for var in ds.data_vars:
            if var == 't2m' or var == 't':
                rename_dict[var] = TEMPERATURE
            elif var == 'u10' or var == 'u':
                rename_dict[var] = WIND_U
            elif var == 'v10' or var == 'v':
                rename_dict[var] = WIND_V
        ds = ds.rename(rename_dict)
        return ds
    
    def rename_coords(self, ds) -> xr.Dataset:
        """
        Rename coordinates in the dataset to standardised names.
        The standard names are defined as strings in src/utils/variables/var_names.py
        """
        rename_dict = {}
        for coord in ds.coords:
            if coord == 'valid_time':
                rename_dict[coord] = TIME
            if coord == 'latitude':
                rename_dict[coord] = LATITUDE
            if coord == 'longitude':
                rename_dict[coord] = LONGITUDE
        ds = ds.rename(rename_dict)
        return ds