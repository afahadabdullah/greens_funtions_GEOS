#!/usr/bin/env python
import xarray as xr
import sys

fpath = sys.argv[1] if len(sys.argv) > 1 else '/home6/afahad/nobackup27/GFEXPv11.7/GF01_ref/holding/geosgcm_prog/GF01_ref.geosgcm_prog.20060701_0000z.nc4'

ds = xr.open_dataset(fpath)
print('Variables:', list(ds.data_vars)[:20])
print()
print('Coords:', list(ds.coords))
print()

# Check vertical coord
for lev_name in ['lev', 'level', 'plev', 'z']:
    if lev_name in ds.coords:
        levs = ds[lev_name].values
        print(f'{lev_name} values ({len(levs)}): {levs}')
        break

# Check T variable
for v in ['T', 'TEMP', 'ta', 'TT']:
    if v in ds.data_vars:
        print(f'\n{v} dims: {ds[v].dims}')
        print(f'{v} shape: {ds[v].shape}')
        break
