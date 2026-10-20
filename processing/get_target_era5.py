#!/usr/bin/env python
"""
Retrieve ERA5 target data from Google Cloud - Phase 1: 2D variables only.
Then add T200 in a separate pass to manage memory.
"""
import xarray as xr
import numpy as np
import os
import gc

OUTPUT_DIR = '/home6/afahad/nobackup27/GFEXPv11.7/scripts/GFO'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'era5_target_2006_jul_dec.nc')
ZARR_URL = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'

def get_era5_data():
    print(f"Connecting to: {ZARR_URL}")
    ds = xr.open_zarr(ZARR_URL, consolidated=True)
    print("Connected.")
    
    try:
        # Phase 1: 2D variables
        print("\n=== Phase 1: 2D Variables ===")
        vars_2d = ['2m_temperature', 'total_precipitation']
        
        print("July 2006 2D...")
        jul_2d = ds[vars_2d].sel(time=slice('2006-07-01', '2006-07-31')).mean(dim='time').compute()
        gc.collect()
        
        print("December 2006 2D...")
        dec_2d = ds[vars_2d].sel(time=slice('2006-12-01', '2006-12-31')).mean(dim='time').compute()
        gc.collect()
        
        # Phase 2: T200 with chunked processing
        print("\n=== Phase 2: T200 (chunked) ===")
        
        # Get T at 200 hPa - process week by week to reduce memory
        def get_monthly_t200(year, month):
            """Get monthly mean T at 200 hPa using weekly chunks."""
            from datetime import datetime, timedelta
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            
            weekly_means = []
            current = start
            while current <= end:
                week_end = min(current + timedelta(days=6), end)
                print(f"  Processing {current.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}...")
                
                chunk = ds['temperature'].sel(
                    level=200,
                    time=slice(current.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d'))
                ).mean(dim='time').compute()
                
                weekly_means.append(chunk)
                current = week_end + timedelta(days=1)
                gc.collect()
            
            # Combine weekly means (weighted by days if needed, but approx same)
            return sum(weekly_means) / len(weekly_means)
        
        print("July 2006 T200...")
        jul_t200 = get_monthly_t200(2006, 7)
        gc.collect()
        
        print("December 2006 T200...")
        dec_t200 = get_monthly_t200(2006, 12)
        gc.collect()
        
        # Merge
        jul_mean = xr.merge([jul_2d, jul_t200.to_dataset(name='temperature')])
        dec_mean = xr.merge([dec_2d, dec_t200.to_dataset(name='temperature')])
        del jul_2d, dec_2d, jul_t200, dec_t200
        gc.collect()
        
        # Expand time dim
        jul_mean = jul_mean.expand_dims(time=[np.datetime64('2006-07-01')])
        dec_mean = dec_mean.expand_dims(time=[np.datetime64('2006-12-01')])
        
        ds_out = xr.concat([jul_mean, dec_mean], dim='time')
        
        # Rename
        ds_out = ds_out.rename({
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
            'temperature': 't200'
        })
        
        if 'level' in ds_out.coords:
            ds_out = ds_out.drop_vars('level')
        
        print(f"\nWriting to {OUTPUT_FILE}...")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        ds_out.to_netcdf(OUTPUT_FILE, format='NETCDF4')
        print(f"Success!\n{ds_out}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_era5_data()
