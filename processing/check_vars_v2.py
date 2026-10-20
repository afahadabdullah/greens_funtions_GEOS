
import xarray as xr
import os
import glob

base_dir = '/nobackupp27/afahad/GFEXPv11.7/GF01_ref/holding'
surf_pattern = os.path.join(base_dir, 'geosgcm_surf', '*geosgcm_surf*.nc4')
prog_pattern = os.path.join(base_dir, 'geosgcm_prog', '*geosgcm_prog*.nc4')

def get_first_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[0]

def check_vars(path, context):
    if not path:
        print(f"No files found for {context}")
        return

    print(f"Checking {context} file: {path}")
    try:
        # Open with xarray, no decoding to be fast
        with xr.open_dataset(path, decode_times=False) as ds:
            print(f"--- {context} Variables ---")
            all_vars = list(ds.data_vars)
            
            # Check for LW related vars
            if context == 'SURF':
                lw_vars = [v for v in all_vars if 'LW' in v]
                print(f"LW candidates: {lw_vars}")
                required = ['TS', 'T2M', 'TPRECTOT', 'SLP', 'FRSEAICE', 'SWGNET']
                found = [v for v in required if v in all_vars]
                missing = [v for v in required if v not in all_vars]
                print(f"Found required: {found}")
                print(f"Missing required: {missing}")

            if context == 'PROG':
                required = ['T200', 'V200', 'H500', 'U200']
                # These might be 3D variables subsetted? Or 2D?
                # Usually prog has T, U, V, H, PL etc.
                # If T200 is not there, we might need to extract T at 200hPa.
                # Let's check if T, U, V, H exist.
                print(f"Variables in PROG: {all_vars}")

    except Exception as e:
        print(f"Error reading {path}: {e}")

surf_file = get_first_file(surf_pattern)
check_vars(surf_file, 'SURF')

prog_file = get_first_file(prog_pattern)
check_vars(prog_file, 'PROG')
