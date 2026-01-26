import glob
import xarray as xr
from datetime import datetime
from dask.diagnostics import ProgressBar
from convcnp_assim_nz.data_processing.nzra_processor import ProcessNZRA
from convcnp_assim_nz.config.env_loader import get_env_var, use_absolute_filepaths
from convcnp_assim_nz.utils.variables.coord_names import LATITUDE, LONGITUDE

def main():
    # read in himawari8 data for 2017
    start_year = 2013
    end_year = 2017
    
    print(f"Reading in ERA5 data for {start_year} to {end_year+1}...")
    
    # use absolute filepaths for this script - i.e. data is distributed across filesystem, not relative to DATA_HOME
    use_absolute_filepaths(True)
    
    filepath_base = "/esi/project/niwa00004/riom/data/era5_nz/"

    files = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_str = f"{month:02d}"
            files.append(f"{filepath_base}{year}-{month_str}.nc")
    
    era5_ds = xr.open_mfdataset(files, engine="h5netcdf")

    nzra_processor = ProcessNZRA()
    nzra_ds = nzra_processor.load_ds([2017])

    lat_min = nzra_ds[LATITUDE].min().item()
    lat_max = nzra_ds[LATITUDE].max().item()
    lon_min = nzra_ds[LONGITUDE].min().item()
    lon_max = nzra_ds[LONGITUDE].max().item()

    era5_ds_croptonzra = era5_ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )

    rechunked = era5_ds_croptonzra.chunk({
        'time': 24,
        'latitude': 161,
        'longitude': 281
    })

    print("Saving ERA5 data to zarr...")
    save_location = "/esi/project/niwa00004/crowelenn/data/era5_nz_temperature_2013_2017/era5_nz_temperature_2013_2017.zarr"
    rechunked.to_zarr(save_location)

if __name__ == "__main__":
    main()