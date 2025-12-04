

# pass in a set of xarray indices (lat, lon, datetime) and get back the sun incidence angle at those points

import xarray as xr
from src.utils.variables.coord_names import *
from src.utils.variables.var_names import *
import numpy as np

def get_declination_angle(day_of_year: xr.DataArray) -> xr.DataArray:
    """
    Given a DataArray of day_of_year values, compute the solar declination angle
    in radians for each day.

    Formula:
        decl = 23.44 degrees * sin( 2pi * (284 + N) / 365 )
    where N is the day of the year (1 to 365).
    """

    decl_deg = xr.apply_ufunc(
        # modified formula from 23.44 * np.sin(2*np.pi*(284 + n)/365)
        lambda n: 23.44 * np.sin(2*np.pi*(284 + n)/365),
        #lambda n: ((23.44 * np.pi)/180) * np.sin(2*np.pi*(284 + n)/365),
        day_of_year
    )
    decl = np.deg2rad(decl_deg)
    return decl

def get_sun_culmination(ds) -> xr.DataArray:
    """
    This is the cosine of the zenith angle of the sun at solar culmination (solar noon).
    I.e. the highest the sun gets in the sky that day at that latitude.
    This is used in-place of 'day of year' encoding.
    """

    times = ds[TIME]
    lats = ds[LATITUDE]
    lons = ds[LONGITUDE]

    decl = get_declination_angle(
        xr.DataArray(
            times.dt.dayofyear,
            coords={TIME: times},
            dims=[TIME]
        )
    )

    lat = np.deg2rad(lats)

    # incidence angle formula
    # returns: 'sine of the solar elevation angle' at solar culmination
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

def get_lon_std_meridian(ds: xr.Dataset) -> xr.DataArray:
    """
    Given a dataset with longitude coordinates, compute the standard meridian
    for each longitude point.

    Standard meridian is defined as the nearest multiple of 15 degrees to the longitude.
    """

    lons = ds[LONGITUDE]

    std_meridian = xr.apply_ufunc(
        lambda lon: 15 * np.round(lon / 15),
        lons
    )

    std_meridian = std_meridian.rename("lon_std_meridian")

    std_meridian.attrs["units"] = "degrees"

    return std_meridian

def get_sun_position(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute cos(zenith angle) of the sun for each (time, lat, lon)
    using UTC timestamps and correct solar-time conversion.
    """

    times = ds[TIME]
    lats = ds[LATITUDE]
    lons = ds[LONGITUDE]

    # declination angle from day of year
    doy = times.dt.dayofyear
    decl = get_declination_angle(
        xr.DataArray(doy, coords={TIME: times}, dims=[TIME])
    )

    lat_rad = np.deg2rad(lats)
    lon_deg = lons

    day = doy.values
    eq = np.zeros_like(day, dtype=float)

    mask1 = day <= 106
    mask2 = (day > 106) & (day <= 166)
    mask3 = (day > 166) & (day <= 247)
    mask4 = day > 247

    eq[mask1] = -14.2 * np.sin(np.pi * (day[mask1] + 7) / 111)
    eq[mask2] =   4.0 * np.sin(np.pi * (day[mask2] - 106) / 59)
    eq[mask3] =  -6.5 * np.sin(np.pi * (day[mask3] - 166) / 81)
    eq[mask4] =   6.8 * np.sin(np.pi * (day[mask4] - 247) / 118)

    eq_da = xr.DataArray(eq, coords={TIME: times}, dims=[TIME])

    utc_hour = (
        times.dt.hour
        + times.dt.minute / 60
        + times.dt.second / 3600
    )

    utc_hour_da = xr.DataArray(utc_hour, coords={TIME: times}, dims=[TIME])

    solar_time = utc_hour_da + lon_deg / 15.0 + eq_da / 60.0

    hour_angle = np.deg2rad(15 * (solar_time - 12))

    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )

    cos_zenith = xr.broadcast(cos_zenith, lats, lons)[0]
    cos_zenith = cos_zenith.rename(SUN_ANGLE)
    cos_zenith = cos_zenith.transpose(TIME, LATITUDE, LONGITUDE)
    cos_zenith.attrs["units"] = "cos(zenith)"

    return cos_zenith
