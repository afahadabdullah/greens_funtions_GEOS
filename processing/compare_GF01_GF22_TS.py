import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

# Define paths
gf01_dir = 'GF01_ref/holding/geosgcm_surf'
gf22_dir = 'GF22_CQF/geosgcm_surf'

# Define period (first year starting 2005-05)
months = pd.date_range(start='2005-05-01', end='2006-04-01', freq='MS')

gf01_files = []
gf22_files = []

print("Selecting files for the first year (2005-05 to 2006-04)...")

for date in months:
    # GF01 file format: GF01_ref.geosgcm_surf.YYYYMMDD_0000z.nc4
    gf01_fname = f"GF01_ref.geosgcm_surf.{date.strftime('%Y%m%d')}_0000z.nc4"
    gf01_path = os.path.join(gf01_dir, gf01_fname)
    
    # GF22 file format: GF22_CQF.geosgcm_surf.monthly.YYYYMM.nc4
    gf22_fname = f"GF22_CQF.geosgcm_surf.monthly.{date.strftime('%Y%m')}.nc4"
    gf22_path = os.path.join(gf22_dir, gf22_fname)
    
    if os.path.exists(gf01_path) and os.path.exists(gf22_path):
        gf01_files.append(gf01_path)
        gf22_files.append(gf22_path)
    else:
        print(f"Missing file for {date.strftime('%Y-%m')}")
        if not os.path.exists(gf01_path):
            print(f"  Missing GF01: {gf01_path}")
        if not os.path.exists(gf22_path):
            print(f"  Missing GF22: {gf22_path}")

print(f"Found {len(gf01_files)} matching pairs of files.")

if not gf01_files:
    print("No files found. Exiting.")
    exit(1)

# Open datasets
# Using open_mfdataset to concatenate along time dimension if needed, 
# but here we might just want to average over time or compare month by month.
# Let's just load them. 'TS' is the variable.

try:
    ds01 = xr.open_mfdataset(gf01_files, concat_dim='time', combine='nested')
    ds22 = xr.open_mfdataset(gf22_files, concat_dim='time', combine='nested')
    
    print("Datasets opened successfully.")
    print("DS01 variables:", list(ds01.data_vars))
    
    # Check if TS is in the datasets
    if 'TS' not in ds01.data_vars or 'TS' not in ds22.data_vars:
        print("Variable 'TS' not found in one of the datasets.")
        # Try to find case-insensitive match or similar names
        for v in ds01.data_vars:
            if 'ts' in v.lower():
                print(f"Potential match in GF01: {v}")
        exit(1)

    ts01 = ds01['TS']
    ts22 = ds22['TS']

    # Compute mean over time (first year average)
    ts01_mean = ts01.mean(dim='time')
    ts22_mean = ts22.mean(dim='time')

    # Compute difference
    diff = ts22_mean - ts01_mean

    print(f"TS GF01 Mean (Global): {ts01_mean.mean().values:.2f}")
    print(f"TS GF22 Mean (Global): {ts22_mean.mean().values:.2f}")
    print(f"Difference Min: {diff.min().values:.2f}")
    print(f"Difference Max: {diff.max().values:.2f}")
    print(f"Difference Mean: {diff.mean().values:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    diff.plot(cmap='RdBu_r')
    plt.title('Surface Temperature Difference (GF22 - GF01)\nFirst Year Mean (2005-05 to 2006-04)')
    plt.savefig('scripts/TS_diff_GF22_GF01.png')
    print("Plot saved to scripts/TS_diff_GF22_GF01.png")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

