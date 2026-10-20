#!/usr/bin/env python
"""Check ARCO ERA5 for pressure level temperature."""
import xarray as xr

ZARR_URL = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'

ds = xr.open_zarr(ZARR_URL, consolidated=True)
print("All variables:")
for v in sorted(ds.data_vars):
    print(f"  {v}: {ds[v].dims}")

print("\nCoordinates:")
for c in ds.coords:
    print(f"  {c}: {ds[c].shape}")
