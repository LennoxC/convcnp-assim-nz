import xarray as xr
import os
from convcnp_assim_nz.data_processing.file_loaders.himawari8_fileloader import Himawari8FileLoader
from convcnp_assim_nz.utils.variables.var_names import *
from convcnp_assim_nz.utils.variables.coord_names import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ProcessHimawari8:
    
    def __init__(self):
        self.file_loader = Himawari8FileLoader()
        pass

    def load_ds(self, 
                years: list,
                standardise_var_names: bool=True,
                standardise_coord_names: bool=True) -> xr.Dataset:
        """ Load Himawari-8 dataset from file """
        ds = self.file_loader.load_himawari8_dataset()
        
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
            raise ValueError (f'Years should be int, str or list, not {type(years)}')
        
        #ds = ds.sel(time=ds.time.dt.year.isin(years))

        if standardise_var_names:
            ds = self.rename_variables(ds) # standardise variable names

        if standardise_coord_names:
            ds = self.rename_coords(ds) # standardise coordinate names
        
        return ds
    
    def rename_coords(self, ds) -> xr.Dataset:
        """
        Rename coordinates in the dataset to standardised names.
        The standard names are defined as strings in src/utils/variables/coord_names.py

        time       (time) datetime64[ns] 70kB 2017-01-01T00:10:00 ... 2017-12-31T...
        latitude   (latitude) float32 740B -37.49 -37.47 -37.46 ... -35.02 -35.0
        longitude  (longitude) float32 740B 173.8 173.8 173.8 ... 176.3 176.3 176.3

        """
        
        rename_dict = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'time': TIME,
        }

        ds = ds.rename(rename_dict)
        return ds
    
    def rename_variables(self, ds) -> xr.Dataset:
        """
        Rename variables in the dataset to standardised names.
        The standard names are defined as strings in src/utils/variables/var_names.py

        Data variables:
        B03      (time, latitude, longitude) float64 2GB dask.array<chunksize=(1, 185, 185), meta=np.ndarray>
        B09      (time, latitude, longitude) float64 2GB dask.array<chunksize=(1, 185, 185), meta=np.ndarray>
        B13      (time, latitude, longitude) float64 2GB dask.array<chunksize=(1, 185, 185), meta=np.ndarray>
        """
        rename_dict = {
            'B03': BAND_3,
            'B09': BAND_9,
            'B13': BAND_13
        }

        ds = ds.rename(rename_dict)
        return ds
    
    def get_valid_timesteps_vectorised(self, ds, band, tol = 1e-8, timerange=None, offset=timedelta(hours=1)):
        """
        Return pandas.DatetimeIndex of timesteps between timerange (inclusive) at interval `offset`
        for which the specified frequency band is not entirely zeros.
        """

        # set timerange to the full width of the dataset if not specified
        if timerange is None:
            timerange = (ds[TIME].values.min(), ds[TIME].values.max())

        # coerce endpoints and step to pandas types
        start_ts = pd.to_datetime(timerange[0])
        end_ts = pd.to_datetime(timerange[1])
        step = pd.Timedelta(offset)

        # build the target times we want to test
        target_times = pd.date_range(start=start_ts, end=end_ts, freq=step)
        ds_index = pd.DatetimeIndex(pd.to_datetime(ds[TIME].values))

        nonzero_per_time = (np.abs(ds[band].values) > tol).any(axis=(1, 2))

        # find nearest dataset indices for each target time
        nearest_idx = ds_index.get_indexer(target_times, method="nearest")

        # filter out any unmatched indices (get_indexer returns -1 if no match)
        valid_mask = nearest_idx >= 0
        nearest_idx = nearest_idx[valid_mask]

        # keep only those target times where the nearest dataset time is non-zero
        keep_mask = nonzero_per_time[nearest_idx]
        valid_ds_times = ds_index[nearest_idx[keep_mask]]

        return pd.DatetimeIndex(valid_ds_times)