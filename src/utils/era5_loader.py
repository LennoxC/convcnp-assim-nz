# This script is intended to download ERA5 data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
# Currently only implemented for downloads of hourly data, grouped by day.

from src.config.env_loader import *
from argparse import ArgumentParser
import cdsapi
import os
import logging



# Set up logging
def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Clear all handlers attached to the 'cdsapi' logger
    cdsapi_logger = logging.getLogger("cdsapi")
    cdsapi_logger.handlers.clear()
    
    # Prevent the cdsapi logger from propagating its logs to the root logger
    cdsapi_logger.propagate = False
    
    logger = logging.getLogger(__name__)
    return logger

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
    print(os.getenv("DATA_HOME"))
    pass

if __name__ == "__main__":
    main()