
from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var
import os
import pandas as pd

# Important functions to copy over:
# get_metadata_df()
# load_station_df()

class ProcessStations(DataProcess):
    stations_metadata: pd.DataFrame = None
    
    def __init__(self) -> None:
        super().__init__()

        self.stations_metadata = self.load_station_metadata()

    def load_station_metadata(self):
        
        metadata_filepath = os.path.join(get_env_var('DATA_HOME'),
                                         get_env_var('STATION_SUFFIX'),
                                         get_env_var('STATION_METADATA_FILE'))
        
        return pd.read_csv(metadata_filepath)
    
    def get_station_info(self, station_no: int):
        station_info = self.stations_metadata[self.stations_metadata['station_no'] == station_no]
        if station_info.empty:
            raise ValueError(f"Station number {station_no} not found in metadata.")
        return station_info.iloc[0].to_dict()

    def stations_from_files(self) -> xr.Dataset:
        
        stations_filenames = self.get_station_filenames()

        for fname in stations_filenames:
            pass

    def get_station_filenames(self) -> list[str]:
        all_files = os.listdir(os.path.join(get_env_var('DATA_HOME'),
                                       get_env_var('STATION_SUFFIX')))
    
        station_files = [f for f in all_files if f.endswith('.nc')]
        
        return station_files