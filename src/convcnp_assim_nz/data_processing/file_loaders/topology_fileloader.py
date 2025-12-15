import os
from convcnp_assim_nz.config.env_loader import get_env_var
import pandas as pd
import xarray as xr

class TopologyFileLoader():
    """
    Class to load the topology file.
    Handles all the interaction with the file system.
    Assumes that:
        - topology file is stored at DATA_HOME/TOPOGRAPHY_SUFFIX/TOPOGRAPHY_FILE
    """

    def __init__(self):
        pass

    def load_topology_file(self) -> xr.Dataset:
        
        if os.getenv("USE_ABSOLUTE_FILEPATHS") == "1":
            filename = get_env_var('TOPOGRAPHY_PATH')
        else:
            filename = os.path.join(get_env_var('DATA_HOME'),
                    get_env_var('TOPOGRAPHY_SUFFIX'),
                    get_env_var('TOPOGRAPHY_FILE'))
            
        return xr.open_dataset(filename)