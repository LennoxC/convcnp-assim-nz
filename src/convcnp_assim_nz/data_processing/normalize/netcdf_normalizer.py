import xarray as xr
from typing import Union
import pandas as pd
from convcnp_assim_nz.utils.variables.coord_names import TIME, LATITUDE, LONGITUDE

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
        ds_params = {}
        
        ds_params['mean'] = data[variable].mean(dim=self.avg_over)
        ds_params['std'] = data[variable].std(dim=self.avg_over)

        self.params[variable] = ds_params

        out_variable = f"{variable}_norm"

        data[out_variable] = (data[variable] - self.params[variable]['mean']) / self.params[variable]['std']

        return data[out_variable]

    
    def fit_pd(self, data: pd.DataFrame, column: str):
        
        df_params = {}
        norm_variables = data.groupby(self.average_per)[column].agg(['mean', 'std']).reset_index()

        # join norm_per_station back to stations_era5
        data = data.merge(norm_variables, how='left', on=self.average_per, suffixes=('', '_station_norm'))

        df_params['mean'] = norm_variables.set_index(self.average_per)['mean']
        df_params['std'] = norm_variables.set_index(self.average_per)['std']

        data[f"{column}_norm"] = (data[column] - data['mean']) / data['std']
        data = data.set_index([TIME, LATITUDE, LONGITUDE])

        # return the normalized column along with the average_per columns
        return data[f"{column}_norm"]
    

    def unnormalize_xr(self, data: xr.Dataset, variable: str):
        out_variable = variable.replace("_norm", "")
        
        data[out_variable] = data[variable] * self.params[out_variable]['std'] + self.params[out_variable]['mean']
        
        return data[out_variable]
    
    def save(self):
        raise NotImplementedError("Saving normalization parameters is not implemented yet.")

    def load(self):
        raise NotImplementedError("Loading normalization parameters is not implemented yet.")
