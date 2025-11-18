from src.data_processing.era5_processor import ProcessERA5

def main():
    surface = ProcessERA5().load_ds(mode="surface")
    pressure = ProcessERA5().load_ds(mode="pressure")
    
    


if __name__ == "__main__":
    main()