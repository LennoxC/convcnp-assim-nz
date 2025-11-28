

# pass in a set of xarray indices (lat, lon, datetime) and get back the sun incidence angle at those points

import xarray as xr
from src.utils.variables.coord_names import *
from src.utils.variables.var_names import *
import numpy as np

def get_sun_culmination(ds) -> xr.DataArray:
    times = ds[TIME]
    lats = ds[LATITUDE]
    lons = ds[LONGITUDE]

    # convert times to day of year
    day_of_year = xr.DataArray(
        times.dt.dayofyear,
        coords={TIME: times},
        dims=[TIME]
    )

    # find solar declination angle
    # dec = 23.44 degrees * sin( 2pi * (284 + N) / 365 )
    decl_deg = xr.apply_ufunc(
        lambda n: 23.44 * np.sin(2*np.pi*(284 + n)/365),
        day_of_year
    )
    decl = np.deg2rad(decl_deg)

    lat = np.deg2rad(lats)

    # incidence angle formula
    # returns: 'sine pf the solar elevation angle' at solar culmination
    # incidence = sin(lat) * sin(decl) + cos(lat) * cos(decl)
    incidence_2d = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl)
    )

    incidence, _lat, _lon = xr.broadcast(incidence_2d, lats, lons)

    incidence = incidence.transpose(TIME, LATITUDE, LONGITUDE)

    incidence = incidence.rename(SUN_CULMINATION)

    incidence.attrs["units"] = "cos(zenith)"

    return incidence
