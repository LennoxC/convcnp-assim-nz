
from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var
from src.data_processing.file_loaders.station_fileloader import StationFileLoader
import os
import pandas as pd

# Important functions to copy over:
# get_metadata_df()
# load_station_df()

class ProcessStations(DataProcess):
    stations_metadata: pd.DataFrame = None
    file_loader: StationFileLoader = None
    #stations_by_variable: dict[str, list[int]] = None

    # loading from the file system is abstracted to the file loader
    # different file structures can be handled there without changing this class (changes would be made only in the file loader)

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = StationFileLoader()

        self.stations_metadata = self.file_loader.load_station_metadata()
        #self.stations_by_variable = self.paths_of_stations_with_variable()
    
    # create a dictionary mapping stations to station ids for each measured variable
    # variables could be things like 'temperature', 'humidity', etc.
    def paths_of_stations_with_variable(self, var: str) -> dict[str, list[id]]:
        stations_by_var = {}

        for file, id in self.file_loader.get_station_filenames(returnId=True):
            ds = self.file_loader.load_station_file(file)

            for var in ds.data_vars:
                if var not in stations_by_var:
                    stations_by_var[var] = []
                stations_by_var[var].append(id)

        return stations_by_var

    # ============= Station Info Retrieval =============

    # this differs from get_station_info in deepsensorNZ which gets station by variable (I think?)
    def get_station_info_by_id(self, station_no: int):
        station_info = self.stations_metadata[self.stations_metadata['station_no'] == station_no]
        if station_info.empty:
            raise ValueError(f"Station number {station_no} not found in metadata.")
        return station_info.iloc[0].to_dict()
    
    #def load_stations_time(self, )