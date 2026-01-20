"""
Ensures Cartopy has all required Natural Earth files for offline use,
including higher-resolution coastlines needed for custom extents.

Run this on a node with internet access.
"""

import cartopy
import cartopy.io.shapereader as shapereader
import cartopy.feature as cfeature
import os

cartopy_file_path = os.getenv("CARTOPY_FILES_DIR")

if not cartopy_file_path:
    raise RuntimeError("CARTOPY_FILES_DIR environment variable is not set.")

os.makedirs(cartopy_file_path, exist_ok=True)

cartopy.config['data_dir'] = cartopy_file_path

print(f"Installing Natural Earth data into: {cartopy_file_path}")

features_to_download = [
    cfeature.COASTLINE,
    cfeature.BORDERS,
    cfeature.LAND,
    cfeature.OCEAN,
    cfeature.LAKES,
    cfeature.RIVERS,
]

print("\nDownloading standard 110m Natural Earth data...")
for feature in features_to_download:
    print(f"  {feature.name}")
    list(feature.geometries())

extra_items = [
    ("physical", "coastline", ["50m", "10m"]),
    ("cultural", "admin_0_boundary_lines_land", ["50m", "10m"]),
]

print("\nDownloading higher-resolution (50m, 10m) coastlines and borders...")
for category, name, resolutions in extra_items:
    for res in resolutions:
        print(f"  {name} ({res})")
        try:
            shp = shapereader.natural_earth(
                category=category,
                name=name,
                resolution=res
            )
            list(shapereader.Reader(shp).geometries())
        except Exception as e:
            print(f"    Skipped (not available): {e}")

print("\nAll Cartopy Natural Earth data downloaded successfully!")
