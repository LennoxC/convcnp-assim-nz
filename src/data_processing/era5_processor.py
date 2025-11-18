import os
from typing import Literal, List
import glob
import pandas as pd
import xarray as xr
from datetime import datetime

from src.data_processing.file_loaders.era5_fileloader import ERA5FileLoader
from src.data_processing.utils_processor import DataProcess
from src.config.env_loader import get_env_var

class ProcessERA5(DataProcess):
    file_loader: ERA5FileLoader = None

    def __init__(self) -> None:
        super().__init__()
        self.file_loader = ERA5FileLoader()

    # loading from the file system is abstracted to the file loader
    # different file structures can be handled there without changing this class (changes would be made only in the file loader)

    def load_ds(self, 
                mode: Literal['surface', 'pressure'],
                years: List=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            years (list): specific years, retrieves all if set to None
        """

        return self.file_loader.load_era5_dataset(mode, years)