import os
from src.config.env_loader import get_env_var
import pandas as pd
import xarray as xr


class StationFileLoader():
    """
    Class to load station metadata and data files.
    Handles all the interaction with the file system.
    Assumes that:
        - files are stored at DATA_HOME/STATION_SUFFIX/
        - metadata file is named as per STATION_METADATA_FILE
        - stations are stored as .nc files in the format {station_id}.nc
    """

    def __init__(self):
        pass

    def load_station_metadata(self):
        
        metadata_filepath = os.path.join(get_env_var('DATA_HOME'),
                                         get_env_var('STATION_SUFFIX'),
                                         get_env_var('STATION_METADATA_FILE'))
        
        return pd.read_csv(metadata_filepath)

    def stations_from_files(self) -> xr.Dataset:
        
        stations_filenames = self.get_station_filenames()

        for fname in stations_filenames:
            pass

    def get_station_filenames(self, returnId=False) -> list[str]:
        all_files = os.listdir(os.path.join(get_env_var('DATA_HOME'),
                                       get_env_var('STATION_SUFFIX')))
    
        station_files = [f for f in all_files if f.endswith('.nc')]
        
        if returnId:
            station_ids = [os.path.splitext(f)[0] for f in station_files]
            return station_files, station_ids
        else: # by default just return filenames
            return station_files