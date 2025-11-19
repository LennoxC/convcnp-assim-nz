import os
from typing import Literal, List
import glob
import dask
import pandas as pd
import xarray as xr
from datetime import datetime

from src.utils.variables.var_names import *
from src.data_processing.file_loaders.era5_fileloader import ERA5FileLoader
from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var

"""
Currently supported variables from ERA5:
- temperature (2m or pressure levels)
- wind_u_component (10m or pressure levels)
- wind_v_component (10m or pressure levels)
"""

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
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            years (list): specific years, retrieves all if set to None
        """

        ds = self.file_loader.load_era5_dataset(mode, years)

        # rename the variables to standardised names
        if mode == 'surface':

            # u10: wind u component at 10m
            # v10: wind v component at 10m
            # d2m: dew point temperature at 2m
            # t2m: temperature at 2m
            # lsm: land sea mask
            # msl: mean sea level pressure

            rename_dict = {
                't2m': TEMPERATURE,
                'u10': WIND_U,
                'v10': WIND_V,
            }
            ds = ds.rename(rename_dict)
        elif mode == 'pressure':
            rename_dict = {
                't': TEMPERATURE,
                'u': WIND_U,
                'v': WIND_V,
            }
            ds = ds.rename(rename_dict)

        return ds
    
    def get_variable(self, ds, var) -> xr.DataArray:
        """
        Filter ERA5 data to a single variable, or list of variables.
        Preserves all other dimensions (time, level, lat, lon).
        """
        return ds.sel(variable=var)