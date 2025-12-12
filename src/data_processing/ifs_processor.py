import xarray as xr
import os

from src.config.env_loader import get_env_var
from src.data_processing.file_loaders.nzra_fileloader import NZRAFileLoader
from src.data_processing.utils_processor import DataProcess
from src.utils.variables.var_names import *
from src.utils.variables.coord_names import *

class ProcessIFS(DataProcess):
    def __init__(self) -> None:
        super().__init__()

    def load_ds(self,
                years: list,
                standardise_var_names: bool=True,
                standardise_coord_names: bool=True):
        
        ifs_path = get_env_var('IFS_DATA_PATH')

        ifs_ds = xr.open_zarr(ifs_path)

        if standardise_var_names:
            ifs_ds = self.standardise_var_names(ifs_ds)

        if standardise_coord_names:
            ifs_ds = self.standardise_coord_names(ifs_ds)

        ifs_ds_years = ifs_ds.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-12-31'))

        return ifs_ds_years
    
    def standardise_var_names(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Data Variables from IFS dataset before standardisation:
        10m_u_component_of_wind  (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        10m_v_component_of_wind  (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        10m_wind_speed           (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        2m_temperature           (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        geopotential             (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        mean_sea_level_pressure  (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        specific_humidity        (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        surface_pressure         (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        temperature              (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        total_precipitation_6hr  (time, latitude, longitude) float32 526kB dask.array<chunksize=(1, 9, 10), meta=np.ndarray>
        u_component_of_wind      (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        v_component_of_wind      (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        vertical_velocity        (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        wind_speed               (time, level, latitude, longitude) float32 7MB dask.array<chunksize=(1, 13, 9, 10), meta=np.ndarray>
        """

        vars_rename_map = {
            '2m_temperature': TEMPERATURE,
            'total_precipitation_6hr': PRECIPITATION,
            'surface_pressure': SURFACE_PRESSURE,
            'specific_humidity': HUMIDITY,
            '10m_wind_speed': WIND_SPEED,
            '10m_u_component_of_wind': WIND_U,
            '10m_v_component_of_wind': WIND_V,
            'temperature': 'air_temperature',     # to avoid conflicts
            'wind_speed': 'surface_wind_speed',   # to avoid conflicts
        }
    
        ds_renamed = ds.rename(vars_rename_map)
    
        return ds_renamed

    def standardise_coord_names(self, ds: xr.Dataset) -> xr.Dataset:

        coords_rename_map = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'time': TIME
        }

        ds_renamed = ds.rename(coords_rename_map)

        return ds_renamed