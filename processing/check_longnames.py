
import xarray as xr
import os
import glob

base_dir = '/nobackupp27/afahad/GFEXPv11.7/GF01_ref/holding'

def check_attr(collection, var_list):
    pattern = os.path.join(base_dir, collection, '*.nc4')
    files = sorted(glob.glob(pattern))
    if not files: return
    f = files[0]
    try:
        with xr.open_dataset(f, decode_times=False) as ds:
            for v in var_list:
                if v in ds:
                    print(f"{v} ({collection}): {ds[v].attrs.get('long_name', 'No long_name')}")
    except: pass

check_attr('geosgcm_surf', ['LWS', 'FLNS', 'PRECTOT', 'LW'])
check_attr('geosgcm_prog', ['SLP'])
