
import xarray as xr
import os
import glob
import sys

base_dir = '/nobackupp27/afahad/GFEXPv11.7/GF01_ref/holding'

def list_vars(collection_name):
    pattern = os.path.join(base_dir, collection_name, '*.nc4')
    files = sorted(glob.glob(pattern))
    if not files:
        return

    f = files[0]
    print(f"--- {collection_name} ---")
    try:
        with xr.open_dataset(f, decode_times=False) as ds:
            print(sorted(list(ds.data_vars)))
    except Exception as e:
        print(f"Error: {e}")

targets = ['geosgcm_surf', 'geosgcm_rad', 'geosgcm_moist', 'geosgcm_prec', 'geosgcm_flux']
for c in targets:
    if os.path.exists(os.path.join(base_dir, c)):
        list_vars(c)
