
import xarray as xr
import os
import glob
import sys
import shutil
import warnings
import time
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/nobackupp27/afahad/GFEXPv11.7'

# Mapping Configuration
SURF_CONFIG = {
    'TS': 'TS',
    'T2M': 'T2M',
    'TPRECTOT': 'PRECTOT', 
    'FRSEAICE': 'FRSEAICE',
    'SWGNET': 'SWGNET',
    'LW': 'FLNS'
}

PROG_CONFIG = {
    'SLP': 'SLP', 
    'T200': {'var': 'T', 'lev': 200},
    'V200': {'var': 'V', 'lev': 200},
    'H500': {'var': 'H', 'lev': 500},
    'U200': {'var': 'U', 'lev': 200}
}

YEAR_START = 2008
YEAR_END = 2017
SEASONS = {
    'DJF': [12, 1, 2],
    'JJA': [6, 7, 8]
}

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def get_file_date(fname):
    # Try different formats
    parts = fname.split('.')
    for p in parts:
        # Format 1: YYYYMMDD
        if len(p) == 8 and p.isdigit():
            return int(p[:4]), int(p[4:6])
        # Format 2: YYYYMM (monthly files)
        if len(p) == 6 and p.isdigit():
            # Check range to avoid random 6 digit strings
            if 1900 <= int(p[:4]) <= 2100 and 1 <= int(p[4:6]) <= 12:
                return int(p[:4]), int(p[4:6])
                
    # Format 3: YYYYMMDD_HHmmz (e.g. 20050501_0000z) - strictly looking for YYYYMMDD start
    for p in parts:
        if '_' in p:
             sub = p.split('_')[0]
             if len(sub) == 8 and sub.isdigit():
                 return int(sub[:4]), int(sub[4:6])
                 
    return None, None

def find_files(exp_dir, collection):
    # Check holding dir first, then exp_dir
    # And check for direct .nc4 files in exp_dir/collection (recursive not needed usually)
    
    paths_to_check = [
        os.path.join(exp_dir, 'holding', collection),
        os.path.join(exp_dir, collection)
    ]
    
    all_files = []
    for p in paths_to_check:
        if os.path.exists(p):
            # log(f"  Scanning {p}")
            found = sorted(glob.glob(os.path.join(p, '*.nc4')))
            all_files.extend(found)
    
    return sorted(list(set(all_files))) # Unique files

def extract_vars(fpath, config):
    try:
        with xr.open_dataset(fpath, decode_times=False) as ds:
            out_ds = xr.Dataset()
            # out_ds.attrs = ds.attrs 
            
            for out_var, conf in config.items():
                val = None
                # Direct mapping or 3D extraction
                if isinstance(conf, str):
                    if conf in ds:
                        val = ds[conf]
                elif isinstance(conf, dict):
                    src = conf['var']
                    lev = conf['lev']
                    if src in ds:
                        # Find lev dim
                        found_lev = None
                        for d in ds[src].dims:
                            if 'lev' in d or 'plev' in d:
                                found_lev = d
                                break
                        if found_lev:
                            try:
                                val = ds[src].sel({found_lev: lev}, method='nearest')
                            except: pass
                
                if val is not None:
                     val.name = out_var
                     out_ds[out_var] = val
            
            return out_ds
    except Exception as e:
        log(f"    Error extracting from {os.path.basename(fpath)}: {e}")
        return None

def process_season(season_name, months, file_list, config, output_path):
    if os.path.exists(output_path):
        # log(f"  Skipping {season_name} (exists)")
        return

    relevant_files = []
    for f in file_list:
        y, m = get_file_date(os.path.basename(f))
        if y and m:
            if YEAR_START <= y <= YEAR_END:
                if m in months:
                    relevant_files.append(f)
        # else:
        #     log(f"    Skipping {os.path.basename(f)} (bad date)")
    
    if not relevant_files:
        log(f"  No files found for {season_name} ({YEAR_START}-{YEAR_END})")
        # log(f"    Checked {len(file_list)} candidates")
        return

    log(f"  Computng {season_name} Climatology from {len(relevant_files)} files...")
    
    # Process in chunks or list?
    datasets = []
    for f in relevant_files:
        ds = extract_vars(f, config)
        if ds and len(ds.data_vars) > 0:
            datasets.append(ds)
            
    if not datasets:
        log("  No valid data extracted.")
        return

    try:
        # Concatenate on time
        combined = xr.concat(datasets, dim='time')
        # Mean over time
        mean_ds = combined.mean(dim='time', keep_attrs=True)
        mean_ds.attrs['comment'] = f'Climatology {season_name} {YEAR_START}-{YEAR_END}'
        
        mean_ds.to_netcdf(output_path)
        log(f"  Saved {os.path.basename(output_path)}")
    except Exception as e:
        log(f"  Error aggregation/saving {season_name}: {e}")


def process_experiment(exp_name):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    if not os.path.isdir(exp_dir): return

    log(f"Processing Experiment: {exp_name}")
    clim_dir = os.path.join(exp_dir, 'clim')
    
    # Clean old runs
    if os.path.exists(clim_dir):
        log(f"  Cleaning old clim dir: {clim_dir}")
        try:
            shutil.rmtree(clim_dir)
        except Exception as e:
            log(f"  Failed to remove {clim_dir}: {e}")
            return

    os.makedirs(clim_dir, exist_ok=True)

    # Process SURF
    surf_files = find_files(exp_dir, 'geosgcm_surf')
    log(f"  Found {len(surf_files)} SURF files total")
    if surf_files:
        for seas, months in SEASONS.items():
            out_name = f"{exp_name}.clim.surf.{seas}.{YEAR_START}-{YEAR_END}.nc"
            process_season(seas, months, surf_files, SURF_CONFIG, os.path.join(clim_dir, out_name))

    # Process PROG
    prog_files = find_files(exp_dir, 'geosgcm_prog')
    log(f"  Found {len(prog_files)} PROG files total")
    if prog_files:
        for seas, months in SEASONS.items():
            out_name = f"{exp_name}.clim.prog.{seas}.{YEAR_START}-{YEAR_END}.nc"
            process_season(seas, months, prog_files, PROG_CONFIG, os.path.join(clim_dir, out_name))

def main():
    log("Starting Seasonal Preprocessing...")
    all_items = sorted(os.listdir(BASE_DIR))
    experiments = [d for d in all_items if d.startswith('GF') and os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for exp in experiments:
        process_experiment(exp)
    
    log("Processing Complete.")

if __name__ == "__main__":
    main()
