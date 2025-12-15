import xarray as xr
import os

from convcnp_assim_nz.config.env_loader import get_env_var

class Himawari8FileLoader:
    
    def __init__(self):
        pass
    
    def load_himawari8_dataset(self):
        
        if os.getenv("USE_ABSOLUTE_FILEPATHS") == "1":
            home_path = get_env_var("HIMAWARI8_PATH")

            # if the filepath contains a * then use open_mfdataset
            if "*" in home_path:
                ds = xr.open_mfdataset(home_path, parallel=True)

        else:
            # home path for Himawari-8 data zarr store. Environmental variable HIMAWARI8_SUFFIX should include '.zarr' if required
            home_path = os.path.join(get_env_var("DATA_HOME"), get_env_var("HIMAWARI8_SUFFIX", default="himawari8.zarr"))

        ds = xr.open_zarr(home_path)

        return ds