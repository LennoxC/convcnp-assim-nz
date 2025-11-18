import xarray as xr
import os

from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var

class ProcessTopography(DataProcess):

    def __init__(self) -> None:
        super().__init__()

    def load_ds(self) -> xr.Dataset:
        
        filename = os.path.join(get_env_var('DATA_HOME'),
                   get_env_var('TOPOGRAPHY_SUFFIX'),
                   get_env_var('TOPOGRAPHY_FILE'))
        
        return self.open_ds(file=filename)
    
    def ds_to_da(self, 
                 ds: xr.Dataset, 
                 var: str='elevation',
                 ) -> xr.DataArray:
        return super().ds_to_da(ds, var)
    
    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)
