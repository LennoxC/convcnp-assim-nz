import xarray as xr
import os

from src.config.env_loader import get_env_var
class NZRAFileLoader:
    
    def __init__(self, *args, **kwds):
        pass
    
    def load_nzra_dataset(self, *args, **kwds):
        
        # home path for NZRA data zarr store. Environmental variable NZRA_SUFFIX should include '.zarr' if required
        home_path = os.path.join(get_env_var("DATA_HOME"), get_env_var("NZRA_SUFFIX", default="nzra"))

        ds = xr.open_zarr(home_path)

        return ds