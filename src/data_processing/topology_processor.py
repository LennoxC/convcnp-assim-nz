import xarray as xr
import os

from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var
from src.data_processing.file_loaders.topology_fileloader import TopologyFileLoader

class ProcessTopography(DataProcess):
    file_loader: TopologyFileLoader = None

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = TopologyFileLoader()

    def load_ds(self) -> xr.Dataset:
        # loading from the file system is abstracted to the file loader
        # different file structures can be handled there without changing this class (changes would be made only in the file loader)
        return self.file_loader.load_topology_file()
    
    def ds_to_da(self, 
                 ds: xr.Dataset, 
                 var: str='elevation',
                 ) -> xr.DataArray:
        return super().ds_to_da(ds, var)
    
    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)
