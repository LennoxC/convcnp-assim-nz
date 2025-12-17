import os
from typing import Literal, List
import glob
import pandas as pd
import xarray as xr
from datetime import datetime

from convcnp_assim_nz.data_processing.utils_processor import DataProcess
from convcnp_assim_nz.config.env_loader import get_env_var

class ERA5FileLoader:

    def __init__(self):
        pass

    def get_filenames(self, mode, years
                      #mode: Literal['surface', 'pressure'],
                      #years: List=None,
                      ) -> List[str]:
        """ Get list of ERA5 filenames for variable and list of years (if specified) """ 

        # we must have either 'surface' or 'pressure' mode as those are the two ERA5 data types
        if mode not in ["surface", "pressure"]:
            raise ValueError(f'mode must be "surface" or "pressure", not {mode}')

        # find the home path for the specified mode
        home_path = os.path.join(get_env_var("DATA_HOME"), get_env_var("ERA5_SUFFIX", default="era5"), mode)
   
        # search all .nc files in home path if no years specified
        if years is None:
            filenames = glob.glob(f'{home_path}/*/*/*.nc')
        else: # search for all .nc files in specified years
            filenames = []
            for year in years:
                filenames_year = glob.glob(f'{home_path}/{year}/*.nc')
                if len(filenames_year) == 0:
                    filenames_year = glob.glob(f'{home_path}/{year}/*/*.nc')
                filenames = filenames + filenames_year

        return filenames
    
    def load_era5_dataset(self, 
                mode: Literal['surface', 'pressure'],
                years: List=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            years (list): specific years, retrieves all if set to None
        """

        if mode not in ['surface', 'pressure'] and mode is not None:
            raise ValueError(f'mode must be "surface" or "pressure", not {mode}')

        # convert year to the correct format
        if type(years) == int:
            years = [years]
        elif type(years) == str:
            years = [int(years)]
        elif type(years) == list:
            years = [int(year) for year in years]
            if len(set(years)) == 1:
                years = [years[0]]
        else:
            ValueError (f'Years should be int, str or list, not {type(years)}')

        if os.getenv("USE_ABSOLUTE_FILEPATHS") == "0":
            if mode is None:
                raise ValueError("mode must be specified when USE_ABSOLUTE_FILEPATHS is False")
            
            filenames = self.get_filenames(mode, years)
            return xr.open_mfdataset(filenames)
        else:
            zarr_file = get_env_var("ERA5_PATH")
            return xr.open_zarr(zarr_file)