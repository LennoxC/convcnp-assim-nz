import xarray as xr
import os
import numpy as np
import scipy

from convcnp_assim_nz.data_processing.utils_processor import DataProcess
from convcnp_assim_nz.config.env_loader import get_env_var
from convcnp_assim_nz.data_processing.file_loaders.topography_fileloader import TopographyFileLoader
from convcnp_assim_nz.utils.variables.var_names import *
from convcnp_assim_nz.utils.variables.coord_names import *

class ProcessTopography(DataProcess):
    file_loader: TopographyFileLoader = None

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = TopographyFileLoader()
    
    def load_ds(self, 
                standardise_var_names: bool=True, 
                standardise_coord_names: bool=True) -> xr.Dataset:
        # loading from the file system is abstracted to the file loader
        # different file structures can be handled there without changing this class (changes would be made only in the file loader)
        
        ds = self.file_loader.load_topography_file()
        
        if standardise_var_names:
            ds = self.rename_variables(ds)
        
        if standardise_coord_names:
            ds = self.rename_coords(ds)

        return ds
    
    def rename_variables(self, ds) -> xr.Dataset:
        rename_dict = {}
        for var in ds.data_vars:
            if var == 'elevation':
                rename_dict[var] = ELEVATION

        ds = ds.rename(rename_dict)
        return ds

    def rename_coords(self, ds) -> xr.Dataset:
        rename_dict = {}
        for coord in ds.coords:
            if coord == 'lat':
                rename_dict[coord] = LATITUDE
            elif coord == 'lon':
                rename_dict[coord] = LONGITUDE

        ds = ds.rename(rename_dict)
        return ds

    def compute_tpi(self, ds, window_sizes=[0.1, 0.05, 0.025]) -> xr.Dataset:
        
        for window_size in window_sizes:
            smoothed_elevations = ds[ELEVATION].copy(deep=True)

            # compute the smoothing using a gaussian filter
            scales = window_size / self._find_resolutions(ds)

            smoothed_elevations.data = scipy.ndimage.gaussian_filter(ds[ELEVATION].data, sigma=scales, mode='nearest')

            tpi_values = ds[ELEVATION] - smoothed_elevations
            # output strings will look something like 'tpi_ws0_1' for window size (ws) = 0.1 and TOPOGRAPHIC_POSITION_INDEX = 'tpi'
            ds[f'{TOPOGRAPHIC_POSITION_INDEX}_ws{str(window_size).replace(".", "_")}'] = tpi_values

        return ds


    def _find_resolutions(self, ds) -> np.array:
        coord_names = list(ds.dims)
        resolutions = np.array([np.abs(np.diff(ds.coords[coord].values)[0]) for coord in coord_names])
        return resolutions


    """
    def ds_to_da(self, 
                 ds: xr.Dataset, 
                 var: str='elevation',
                 ) -> xr.DataArray:
        return super().ds_to_da(ds, var)
    
    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)
    """