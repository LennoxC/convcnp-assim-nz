# This script is intended to download ERA5 data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
# Currently only implemented for downloads of hourly data, grouped by day.

# After signing up for an API key, you will need to accept the license agreement. 
# See https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences
# You will be prompted to do this the first time you attempt to use the API.

from src.config.env_loader import *
from src.config.logging_config import setup_logging
from src.utils.era5.cds_client import download_era5_data
from argparse import ArgumentParser
import cdsapi
import os

def get_commandline_args():
    """
    Parse command line arguments, return parser object.
    """
    parser = ArgumentParser(description="Download ERA5 data from the Copernicus Climate Data Store (CDS).")
    parser.add_argument(
        "-c", "--config", type=str, dest="config", default=None,
        help="Path to the configuration file containing the API key.",
    )
    parser.add_argument(
        "-s", "--start_date", type=str, dest="start_date", default=None,
        help="Start date for the data download in YYYYMMDD format.",
    )
    parser.add_argument(
        "-e", "--end_date", type=str, dest="end_date", default=None,
        help="End date for the data download in YYYYMMDD format.",
    )
    parser.add_argument(
        "-o", "--output", type=str, dest="output", default=os.path.join(os.getenv("DATA_HOME"), "era5"), # default output path is DATA_HOME/era5
        help="Output directory for the downloaded data. Default is DATA_HOME/era5.",
    )
    parser.add_argument(
        "-p", "--parallel", action='store_true', dest="parallel", default=True,
        help="Use parallel downloading.",
    )
    return parser.parse_args()

def main():
    setup_logging()
    args = get_commandline_args()
    download_era5_data(args)
    
if __name__ == "__main__":
    main()