import xarray as xr
import numpy as np

class DataProcess:
    def __init__(self) -> None:
        pass


    def open_ds(self,
                file: str,
                ) -> xr.Dataset:
        """ Open file as xarray dataset """
        return xr.open_dataset(file)


    def open_da(self, 
                da_file: str,
                ) -> xr.DataArray:
        """ Open as dataarray """
        #return rioxarray.open_rasterio(da_file)
        return xr.open_dataarray(da_file)
    

    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: str,
                 ) -> xr.DataArray:
        """ Get data array from dataset """
        return ds[var]

    
    def mask_da(self, 
                da: xr.DataArray, 
                mask_value: float=-1e30,
                ) -> xr.DataArray:
        """ 
        Set to None "no data" points below mask_value (e.g. -1e30) 
        """
        return da.where(da > mask_value).squeeze()


    def coarsen_da(self, 
                    da: xr.DataArray, 
                    coarsen_by: int, 
                    boundary: str = 'trim',
                    ):
        """
        Reduce resolution of data array by factor coarsen_by. e.g. coarsen_by=4 will reduce 25m resolution to 100m resolution.
        https://stackoverflow.com/questions/53886153/resample-xarray-object-to-lower-resolution-spatially
        """
        #return da.coarsen(longitude=coarsen_by, boundary=boundary).mean().coarsen(latitude=coarsen_by, boundary=boundary).mean().squeeze()
        if coarsen_by == 1:
            return da
        else:
            return da.coarsen(latitude=coarsen_by, longitude=coarsen_by, boundary=boundary).mean()


    def rename_xarray_coords(self,
                             da,
                             rename_dict: dict,
                             ):
        """ Rename coordinates """
        return da.rename(rename_dict)


    def save_nc(self,
                da, 
                name: str,
                ) -> None:
        """ Save as .nc netcdf to name e.g. 'path/file.nc' """
        da.to_netcdf(name)


    def resolution(self,
                   ds: xr.Dataset,
                   coord: str,
                   ) -> float:
        """ Calculate resolution of coodinate in dataset """
        return np.round(np.abs(np.diff(ds.coords[coord].values)[0]), 5)