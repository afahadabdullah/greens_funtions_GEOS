#!/usr/bin/env python
"""
Process local MERRA2 data for Green's Function Tuning.
Input: Local MERRA2 monthly mean files in scripts/GFO/M2data/
Output: merra2_target_2006_jul_dec.nc containing 'TS' and 'T200'

Variables:
- TS: From instM_2d_asm_Nx (Surface Temperature) - Mapped to 'TS'
- T200: From instM_3d_ana_Np (Temperature at 200 hPa) - Mapped to 'T200'
"""

import xarray as xr
import numpy as np
import os
import glob

# Configuration
BASE_DIR = '/home6/afahad/nobackup27/GFEXPv11.7/scripts/GFO'
INPUT_DIR = os.path.join(BASE_DIR, 'M2data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'merra2_target_2006_jul_dec.nc')

def process_merra2():
    print("Processing local MERRA2 data...")
    
    # 1. Load July 2006
    print("Loading July 2006...")
    f_2d_jul = os.path.join(INPUT_DIR, 'MERRA2_300.instM_2d_asm_Nx.200607.SUB.nc')
    f_3d_jul = os.path.join(INPUT_DIR, 'MERRA2_300.instM_3d_ana_Np.200607.SUB.nc')
    
    ds_2d_jul = xr.open_dataset(f_2d_jul)
    ds_3d_jul = xr.open_dataset(f_3d_jul)
    
    # Extract TS
    # MERRA2 'TS' is surface skin temperature
    ts_jul = ds_2d_jul['TS'].mean(dim='time') # Should be size 1 time dim anyway
    
    # Extract T200
    t200_jul = ds_3d_jul['T'].sel(lev=200, method='nearest').mean(dim='time').drop_vars('lev')

    # Extract H500 (Geopotential Height at 500 hPa)
    h500_jul = ds_3d_jul['H'].sel(lev=500, method='nearest').mean(dim='time').drop_vars('lev')
    
    # 2. Load Dec 2006
    print("Loading December 2006...")
    f_2d_dec = os.path.join(INPUT_DIR, 'MERRA2_300.instM_2d_asm_Nx.200612.SUB.nc')
    f_3d_dec = os.path.join(INPUT_DIR, 'MERRA2_300.instM_3d_ana_Np.200612.SUB.nc')
    
    ds_2d_dec = xr.open_dataset(f_2d_dec)
    ds_3d_dec = xr.open_dataset(f_3d_dec)
    
    ts_dec = ds_2d_dec['TS'].mean(dim='time')
    t200_dec = ds_3d_dec['T'].sel(lev=200, method='nearest').mean(dim='time').drop_vars('lev')
    h500_dec = ds_3d_dec['H'].sel(lev=500, method='nearest').mean(dim='time').drop_vars('lev')
    
    # 3. Combine into Target Dataset
    # We need to assign valid times for the tuner: '2006-07-01' and '2006-12-01'
    
    # Concatenate Time
    ts_concat = xr.concat([ts_jul, ts_dec], dim='time')
    ts_concat['time'] = [np.datetime64('2006-07-01'), np.datetime64('2006-12-01')]
    
    t200_concat = xr.concat([t200_jul, t200_dec], dim='time')
    t200_concat['time'] = [np.datetime64('2006-07-01'), np.datetime64('2006-12-01')]
    
    h500_concat = xr.concat([h500_jul, h500_dec], dim='time')
    h500_concat['time'] = [np.datetime64('2006-07-01'), np.datetime64('2006-12-01')]
    
    ds_out = xr.Dataset({
        'TS': ts_concat,
        'T200': t200_concat,
        'H500': h500_concat
    })
    
    # Ensure coordinates are named lat/lon for consistency if possible, 
    # but the tuning script handles renaming 'latitude'/'longitude'.
    # MERRA2 usually has 'lat', 'lon'.
    print("Coords:", list(ds_out.coords))
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    ds_out.to_netcdf(OUTPUT_FILE)
    print("Done.")
    print(ds_out)

if __name__ == "__main__":
    process_merra2()
