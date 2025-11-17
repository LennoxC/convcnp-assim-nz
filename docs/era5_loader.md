# ERA5 Loader
### `src.utils.era5.main`

This is a command line tool to download climate data from the [Climate Data Store](https://cds.climate.copernicus.eu/).

To run this module:
`python -m src.utils.era5.main {args}`

## Arguments
- `-c` `--config` **(required)**. Path to the configuration file which specifies the data format you're requesting from the climate data store. Example files are in the `/era5_samples` folder.
- `-s` `--start_date` **(required)**. Start date for the data download in YYYYMMDD format.
- `-e` `--end_date` **(required)**. End date for the data download in YYYYMMDD format.
- `-o` `--output`. The directory where files will be saved as an absolute path. By default, this is `DATA_HOME` + `/era5`, where `DATA_HOME` is loaded from the `.env` file. This can be overriden with a custom output.
- `-p` `--parallel`. Use parallel downloading. Default `True`.

## Environmental Variables
These must be defined in the `.env` file at the root of this repository.
- `CDS_API_URL`: The base URL for the CDS API. If not specified, this will default to https://cds.climate.copernicus.eu/api, and a warning is thrown. This url is correct as of 18/11/2025.
- `CDS_API_KEY`: API key for the CDS API. Create a CDS API key by signing up at https://cds.climate.copernicus.eu/api-how-to. After creating an account, you can copy your API key. You will also need to accept the license agreement. See https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences. You will be prompted to do this the first time you attempt to use the API, if you haven't accepted the agreement already.