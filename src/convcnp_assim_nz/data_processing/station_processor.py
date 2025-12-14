
from convcnp_assim_nz.data_processing.utils_processor import DataProcess
from convcnp_assim_nz.config.env_loader import get_env_var
from convcnp_assim_nz.data_processing.file_loaders.station_fileloader import StationFileLoader
from convcnp_assim_nz.utils.variables.var_names import *
from convcnp_assim_nz.utils.variables.coord_names import *
import os
import pandas as pd
import xarray as xr

# Important functions to copy over:
# get_metadata_df()
# load_station_df()

class ProcessStations(DataProcess):
    stations_metadata: pd.DataFrame = None
    file_loader: StationFileLoader = None
    #stations_by_variable: dict[str, list[int]] = None

    # loading from the file system is abstracted to the file loader
    # different file structures can be handled there without changing this class (changes would be made only in the file loader)

    def __init__(self, mode='netcdf') -> None:
        super().__init__()
        
        self.mode = mode # set the mode

        self.file_loader = StationFileLoader(mode=mode) # initialize the file loader
        
        if mode == 'netcdf':
            self.stations_metadata = self.file_loader.load_station_metadata()
        else: 
            self.stations_metadata = None  # metadata not used in 'csv' mode
        #self.stations_by_variable = self.paths_of_stations_with_variable()

    '''
    def load_ds(self, vars, year_start=None, year_end=None, standarise_var_names: bool=True, standardise_coord_names: bool=True) -> xr.Dataset:
        """ 
        Load only stations which contain the specified variables (vars) in the specified time range (year_start to year_end).
        Form a new xarray Dataset by the station coordinates (continuous) and timestamp, with the variable values as data.
        """

        station_files, ids = self.file_loader.get_station_filenames(returnId=True)

        ds_list = []
        for file, id in zip(station_files, ids):
            ds_station = self.file_loader.load_station_file(file)

            if standarise_var_names:
                ds_station = self.rename_variables(ds_station)

            # check if the station contains all the requested variables
            if all(var in ds_station.data_vars for var in vars):
                # filter by time range if specified
                if year_start is not None and year_end is not None:
                    ds_station = ds_station.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

                # select only the requested variables
                ds_station = ds_station[vars]

                # add station coordinates as new dimensions
                try:
                    station_info = self.get_station_info_by_id(int(id))
                except:
                    # if station info not found, skip this station
                    print(f"Skipping {id}")
                    continue
                
                ds_station = ds_station.expand_dims({LATITUDE: [station_info['latitude']],
                                                    LONGITUDE: [station_info['longitude']]})
                
                ds_station = ds_station.assign(station_id=int(id))

                ds_list.append(ds_station)

        # combine all station datasets into one
        if ds_list:
            ds_combined = xr.concat(ds_list, dim='station')
        else:
            ds_combined = xr.Dataset()  # return empty dataset if no stations found

        if standardise_coord_names:
            ds_combined = self.rename_coords(ds_combined) # standardise coordinate names

        return ds_combined
    '''

    def load_ds(
            self,
            vars,
            csv_file=None,
            year_start=None,
            year_end=None,
            standarise_var_names=True,
            standardise_coord_names=True,
            ) -> xr.Dataset:
        """
        Efficiently load station datasets that contain `vars` using dask-backed lazy loading.
        """
        if self.mode == 'netcdf':
            station_files, ids = self.file_loader.get_station_filenames(returnId=True)

            def iter_stations():
                for fname, sid in zip(station_files, ids):

                    # Use dask for lazy loading (massive memory win)
                    ds = xr.open_dataset(
                        os.path.join(get_env_var("DATA_HOME"),
                                    get_env_var("STATION_SUFFIX"),
                                    fname),
                        chunks={}
                    )

                    if standarise_var_names:
                        ds = self.rename_variables(ds)

                    # skip stations with missing variables
                    if not all(v in ds.data_vars for v in vars):
                        continue

                    # time filtering early to limit memory
                    if year_start is not None and year_end is not None:
                        ds = ds.sel(
                            time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
                        )

                    ds = ds[vars]

                    var_to_check = vars[0]

                    # dask reduction – cheap and safe
                    all_nan = ds[var_to_check].isnull().all().compute()

                    if bool(all_nan):
                        # skip this station entirely
                        continue

                    # get station metadata
                    try:
                        info = self.get_station_info_by_id(int(sid))
                    except:
                        print(f"Skipping {sid}")
                        continue

                    # Add a station dimension (size 1, cheap)
                    ds = ds.expand_dims({"station": [int(sid)]})

                    # Assign station-level metadata (broadcast as coords)
                    ds = ds.assign_coords(
                        lat=("station", [info["latitude"]]),
                        lon=("station", [info["longitude"]]),
                    )

                    yield ds

            # Build final dataset by concatenating lazily
            station_iter = iter_stations()

            try:
                first = next(station_iter)
            except StopIteration:
                return xr.Dataset()

            ds_comb = xr.concat([first, *station_iter], dim="station")

            if standardise_coord_names:
                ds_comb = self.rename_coords(ds_comb)

            return ds_comb
        
        elif self.mode == 'csv':
            df = self.file_loader.load_station_file(csv_file=csv_file)

            if standarise_var_names:
                df = self.rename_variables(df)
            if standardise_coord_names:
                df = self.rename_coords(df)

            if standardise_coord_names:
                if TIME in df.columns:
                    if year_start is not None and year_end is not None:
                        print(f"Loaded CSV with length {len(df)}")
                        df = df[(df[TIME] >= f"{year_start}-01-01") & (df[TIME] <= f"{year_end}-12-31")]
                        print(f"Loaded CSV with length {len(df)}")
                        #file = file.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

            else:
                print("Year filtering is not available without standard coordinate names.")

            # filter down to the set of variables
            df = df[vars + [LATITUDE, LONGITUDE, TIME, 'AGENT_NO', 'NAME', 'HEIGHT']]
            #df = df[df[vars].notnull().all(axis=1)]  # drop rows with NaNs in any of the vars

            return df
        
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes: 'netcdf', 'csv'.")

            
    def load_df(self, vars, year_start=None, year_end=None, csv_file=None) -> pd.DataFrame:
        """ 
        Load only stations which contain the specified variables (vars) in the specified time range (year_start to year_end).
        Form a new pandas DataFrame by the station coordinates (continuous) and timestamp, with the variable values as data.
        """

        ds = self.load_ds(vars, csv_file=csv_file, year_start=year_start, year_end=year_end)

        if self.mode == 'netcdf':
            df = ds.to_dataframe()
            return df
        elif self.mode == 'csv':
            return ds
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes: 'netcdf', 'csv'.")
        
    
    # create a dictionary mapping stations to station ids for each measured variable
    # variables could be things like 'temperature', 'humidity', etc.
    def paths_of_stations_with_variable(self, var: str) -> dict[str, list[id]]:

        if self.mode != 'netcdf':
            raise ValueError("paths_of_stations_with_variable is only supported in 'netcdf' mode.")

        stations_by_var = {}

        for file, id in self.file_loader.get_station_filenames(returnId=True):
            ds = self.file_loader.load_station_file(file)

            for var in ds.data_vars:
                if var not in stations_by_var:
                    stations_by_var[var] = []
                stations_by_var[var].append(id)

        return stations_by_var
    
    def rename_variables(self, ds) -> xr.Dataset:

        if self.mode == 'netcdf':
            rename_dict = {}
            for var in ds.data_vars:
                if var == 'dry_bulb':
                    rename_dict[var] = TEMPERATURE
        
            ds = ds.rename(rename_dict)
            return ds
        
        elif self.mode == 'csv':
            ds[TEMPERATURE] = (ds['MAX_TEMP'] + ds['MIN_TEMP']) / 2.0
            ds = ds.drop(columns=['MAX_TEMP', 'MIN_TEMP'])
            return ds

        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes: 'netcdf', 'csv'.")

    def rename_coords(self, ds) -> xr.Dataset:

        if self.mode == 'netcdf':
            rename_dict = {}
            for coord in ds.coords:
                if coord == 'time':
                    rename_dict[coord] = TIME

            ds = ds.rename(rename_dict)
            return ds
        
        elif self.mode == 'csv':
            ds = ds.rename(columns={'LAT': LATITUDE, 'LONGT': LONGITUDE, 'OBS_DATE_UTC': TIME })
            ds[TIME] = pd.to_datetime(ds[TIME], format="%Y%m%d:%H%M")
            return ds

        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes: 'netcdf', 'csv'.")

    # ============= Station Info Retrieval =============

    def get_station_info_by_id(self, station_no: int):

        if self.mode != 'netcdf':
            raise ValueError("get_station_info_by_id is only supported in 'netcdf' mode.")

        station_info = self.stations_metadata[self.stations_metadata['station_no'] == station_no]
        if station_info.empty:
            raise ValueError(f"Station number {station_no} not found in metadata.")
        return station_info.iloc[0].to_dict()
    
    #def load_stations_time(self, )