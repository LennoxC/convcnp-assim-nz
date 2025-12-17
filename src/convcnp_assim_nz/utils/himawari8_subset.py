import glob
import xarray as xr
from datetime import datetime
from dask.diagnostics import ProgressBar
from convcnp_assim_nz.data_processing.nzra_processor import ProcessNZRA
from convcnp_assim_nz.config.env_loader import get_env_var, use_absolute_filepaths
from convcnp_assim_nz.utils.variables.coord_names import LATITUDE, LONGITUDE

def get_coords():
    nzra_processor = ProcessNZRA()
    dset_nzra = nzra_processor.load_ds([2017])
    return dset_nzra

def preprocess(ds: xr.Dataset) -> xr.Dataset:
    dset_nzra = get_coords()
    latitude = dset_nzra[LATITUDE].values
    longitude = dset_nzra[LONGITUDE].values

    file = ds.encoding['source']
    time_str = file[-21:-9]
    time = datetime(
        int(time_str[:4]), int(time_str[4:6]), int(time_str[6:8]),
        int(time_str[8:10]), int(time_str[10:12])
    )
    ds = ds.assign_coords({'time': time}).expand_dims('time')
    ds = ds.sel(latitude=latitude, longitude=longitude)
    return ds

def main():
    # read in himawari8 data for 2017
    print("Reading in Himawari-8 data for 2017...")
    
    # use absolute filepaths for this script - i.e. data is distributed across filesystem, not relative to DATA_HOME
    use_absolute_filepaths(True)
    
    pattern = (
        "/esi/project/niwa00004/meyerstj/data/ml-datasets/himawari8/"
        "regridded/2017*/AHI_L1B_2017*10_NZCSM.nc"
    )

    # where to save the zarr
    save_location = "/esi/project/niwa00004/crowelenn/data/himawari8_2017/himawari8.zarr"

    h8_files = glob.glob(pattern)

    with ProgressBar():
        ds_h8 = xr.open_mfdataset(
            h8_files,
            engine="h5netcdf",
            preprocess=preprocess,
            parallel=True,
        )

    print("Saving Himawari-8 data to zarr...")
    ds_h8.to_zarr(save_location)

if __name__ == "__main__":
    main()