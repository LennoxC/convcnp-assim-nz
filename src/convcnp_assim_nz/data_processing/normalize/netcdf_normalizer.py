import xarray as xr
from typing import Union
import pandas as pd
from convcnp_assim_nz.utils.variables.coord_names import TIME, LATITUDE, LONGITUDE
import os

"""
I have tried to construct this so that it operates in a similar pattern to the deepsensor normalizer.
For now, this is its own class which saves data in a different way, but in future we may want to refactor
this so that it is integrated into deepsensor.
"""

class NetCDFNormalizer:
    def __init__(self):
        
        self.avg_over = [TIME]
        self.average_per = [LATITUDE, LONGITUDE]

        self.params = {}
        
    def __call__(self, data: Union[xr.Dataset, pd.DataFrame], **kwds):
        
        if isinstance(data, list):
            return tuple([self.fit(d) for d in data])
        else:
            return self.fit(data)

    def fit(self, data: xr.Dataset | pd.DataFrame):
        if isinstance(data, xr.Dataset):
            ds = xr.Dataset()
            for variable in data.data_vars:
                ds[f"{variable}_norm"] = self.fit_xr(data, variable)
            return ds
        if isinstance(data, pd.DataFrame):
            df = pd.DataFrame(index=data.set_index([TIME, LATITUDE, LONGITUDE]).index)
            for column in [col for col in data.columns if col not in [TIME, LATITUDE, LONGITUDE]]:
                df[f"{column}_norm"] = self.fit_pd(data, column)
            return df
        else:
            raise TypeError("Input data must be an xarray.Dataset or pandas.DataFrame")

    def fit_xr(self, data: xr.Dataset, variable: str):
        if self.params.get(variable) is None:
            ds_params = {}
        
            ds_params['mean'] = data[variable].mean(dim=self.avg_over)
            ds_params['std'] = data[variable].std(dim=self.avg_over)

            self.params[variable] = ds_params

        out_variable = f"{variable}_norm"

        data[out_variable] = (data[variable] - self.params[variable]['mean']) / self.params[variable]['std']

        return data[out_variable]

    
    def fit_pd(self, data: pd.DataFrame, column: str):
        if self.params.get(column) is None:
            df_params = {}
            new_norm_variables = data.groupby(self.average_per)[column].agg(['mean', 'std']).reset_index()
            df_params['mean'] = new_norm_variables.set_index(self.average_per)['mean']
            df_params['std'] = new_norm_variables.set_index(self.average_per)['std']

            self.params[column] = df_params

        # fetch the normalization parameters for this variable from the params dictionary
        norm_variables = pd.concat([self.params[column]['mean'], self.params[column]['std']], axis=1)

        # join norm_per_station back to stations_era5
        data = data.merge(norm_variables, how='left', on=self.average_per, suffixes=('', '_station_norm'))

        data[f"{column}_norm"] = (data[column] - data['mean']) / data['std']
        data = data.set_index([TIME, LATITUDE, LONGITUDE])

        # return the normalized column along with the average_per columns
        return data[f"{column}_norm"]
    

    def unnormalize_xr(self, data: xr.Dataset, variable: str):
        out_variable = variable.replace("_norm", "")
        
        data[out_variable] = data[variable] * self.params[out_variable]['std'] + self.params[out_variable]['mean']
        
        return data[out_variable]
    
    def save(self, filepath: str):
        # directories within the filepath will need to be created for each variable
        for variable, var_params in self.params.items():
            var_dir = os.path.join(filepath, variable)
            os.makedirs(var_dir, exist_ok=True)

            # if the parameters are xarray objects, we can save them as netcdf files
            if isinstance(var_params['mean'], xr.DataArray) and isinstance(var_params['std'], xr.DataArray):
                var_params['mean'].to_netcdf(os.path.join(var_dir, 'mean.nc'))
                var_params['std'].to_netcdf(os.path.join(var_dir, 'std.nc'))

            # if the parameters are pandas objects, we can save them as csv files
            elif isinstance(var_params['mean'], pd.Series) and isinstance(var_params['std'], pd.Series):
                var_params['mean'].to_csv(os.path.join(var_dir, 'mean.csv'))
                var_params['std'].to_csv(os.path.join(var_dir, 'std.csv'))
            
            else:
                raise TypeError(f"Normalization parameters must be either xarray.DataArray or pandas.DataFrame. Found {type(var_params['mean'])} and {type(var_params['std'])} for variable {variable}")

        print(f"Normalization parameters saved to {filepath}")

    def load(self, filepath: str):
        # discover the variables by looking at the directories in the filepath
        for variable in os.listdir(filepath):
            var_dir = os.path.join(filepath, variable)
            if os.path.isdir(var_dir):
                var_params = {}
                mean_path_nc = os.path.join(var_dir, 'mean.nc')
                std_path_nc = os.path.join(var_dir, 'std.nc')
                mean_path_csv = os.path.join(var_dir, 'mean.csv')
                std_path_csv = os.path.join(var_dir, 'std.csv')

                if os.path.exists(mean_path_nc) and os.path.exists(std_path_nc):
                    var_params['mean'] = xr.load_dataarray(mean_path_nc)
                    var_params['std'] = xr.load_dataarray(std_path_nc)

                elif os.path.exists(mean_path_csv) and os.path.exists(std_path_csv):
                    var_params['mean'] = pd.read_csv(mean_path_csv).set_index(self.average_per)
                    var_params['std'] = pd.read_csv(std_path_csv).set_index(self.average_per)

                else:
                    raise FileNotFoundError(f"Normalization parameters for variable {variable} not found in {var_dir}")

                self.params[variable] = var_params

        print(f"Normalization parameters loaded from {filepath}")
        
