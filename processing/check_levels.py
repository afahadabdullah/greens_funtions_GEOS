#!/usr/bin/env python
"""Check ERA5 pressure levels."""
import xarray as xr
ZARR_URL = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'
ds = xr.open_zarr(ZARR_URL, consolidated=True)
print("Pressure levels (hPa):")
print(ds.level.values)
