import math
import xarray as xr

# Scalar conversion functions

def kelvin_to_celsius(da):
    """Convert temperature from Kelvin to Celsius."""
    da_converted = da - 273.15
    da_converted.attrs['units'] = 'Celsius'
    return da_converted

def celsius_to_kelvin(da):
    """Convert temperature from Celsius to Kelvin."""
    da_converted = da + 273.15
    da_converted.attrs['units'] = 'Kelvin'
    return da_converted