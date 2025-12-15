import xarray as xr
import os

from convcnp_assim_nz.config.env_loader import get_env_var
from convcnp_assim_nz.data_processing.file_loaders.nzra_fileloader import NZRAFileLoader
from convcnp_assim_nz.data_processing.utils_processor import DataProcess
from convcnp_assim_nz.utils.variables.var_names import *
from convcnp_assim_nz.utils.variables.coord_names import *

class ProcessNZRA(DataProcess):
    file_loader: NZRAFileLoader = None

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = NZRAFileLoader()

    def load_ds(self,
                years: list,
                standardise_var_names: bool=True,
                standardise_coord_names: bool=True):
        
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

        # load from files
        ds = self.file_loader.load_nzra_dataset()
        ds = ds.sel(time=ds.time.dt.year.isin(years))

        ds = self.avg_temperature(ds)

        if standardise_var_names:
            ds = self.rename_variables(ds) # standardise variable names
        
        if standardise_coord_names:
            ds = self.rename_coords(ds) # standardise coordinate names

        return ds

    def rename_coords(self, ds) -> xr.Dataset:
        """
        Rename coordinates in the dataset to standardised names.
        The standard names are defined as strings in src/utils/variables/coord_names.py
        
        COORDS:
        latitude   (latitude) float64 1kB -37.49 -37.48 -37.46 ... -35.02 -35.0
        longitude  (longitude) float64 1kB 173.8 173.8 173.8 ... 176.3 176.3 176.3
        time       datetime64[ns] 8B 2017-01-01T03:00:00
        
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

        DATA VARIABLES:
        max_sfc_temp         (latitude, longitude) float32 137kB dask.array<chunksize=(185, 185), meta=np.ndarray>
        mean_sfc_dw_sw_flux  (latitude, longitude) float32 137kB dask.array<chunksize=(185, 185), meta=np.ndarray>
        min_sfc_temp         (latitude, longitude) float32 137kB dask.array<chunksize=(185, 185), meta=np.ndarray>
        sfc_merid_wind       (latitude, longitude) float32 137kB dask.array<chunksize=(185, 185), meta=np.ndarray>
        sfc_zonal_wind       (latitude, longitude) float32 137kB dask.array<chunksize=(185, 185), meta=np.ndarray>
        """
        rename_dict = {
            'temperature_avg': TEMPERATURE,
            'mean_sfc_dw_sw_flux': SHORTWAVE_FLUX_DOWN,  # mean downward shortwave flux at the surface
            'sfc_merid_wind': WIND_U,                    # true northward (meridional) wind at 10m
            'sfc_zonal_wind': WIND_V                     # true eastward (zonal) wind at 10m
        }

        ds = ds.rename(rename_dict)

        return ds

    def avg_temperature(self, ds: xr.Dataset) -> xr.Dataset:
        """
        By default NZRA provides min and max surface temperature.
        Apparently within ESNZ it is convention to use the average of these two
        as a proxy for surface air temperature.
        """

        ds['temperature_avg'] = (ds['max_sfc_temp'] + ds['min_sfc_temp']) / 2
        ds = ds.drop_vars(['max_sfc_temp', 'min_sfc_temp'])

        return ds
