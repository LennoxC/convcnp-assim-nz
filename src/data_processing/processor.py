from src.data_processing.era5_processor import ProcessERA5

# the main data pre-processing module.
# combine ERA5, Sensor Readings, Topographic layers into one netCDF file

def main():
    surface = ProcessERA5().load_ds(mode="surface")
    pressure = ProcessERA5().load_ds(mode="pressure")
    


if __name__ == "__main__":
    main()