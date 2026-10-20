
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import os
import glob
import numpy as np
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = '/nobackupp27/afahad/GFEXPv11.7'
REF_EXP = 'GF01_ref'
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SEASONS = ['DJF', 'JJA']
VARIABLES = {
    'TS': {'label': 'Surface Temperature (K)', 'scale': 1.0, 'cmap': 'RdBu_r'},
    'TPRECTOT': {'label': 'Total Precipitation (mm/day)', 'scale': 86400.0, 'cmap': 'BrBG'},
    'SWGNET': {'label': 'Surface Net Shortwave Flux (W/m^2)', 'scale': 1.0, 'cmap': 'RdBu_r'}
}

def get_clim_file(exp_name, collection_type, season):
    fname = f"{exp_name}.clim.{collection_type}.{season}.2008-2017.nc"
    fpath = os.path.join(BASE_DIR, exp_name, 'clim', fname)
    if os.path.exists(fpath): return fpath
    return None

def plot_difference(exp_name, season, var_name, ref_ds, exp_ds):
    try:
        if var_name not in ref_ds or var_name not in exp_ds:
            print(f"  Variable {var_name} missing in data.")
            return

        ref_data = ref_ds[var_name]
        exp_data = exp_ds[var_name]

        # Calculate Diff
        diff = exp_data - ref_data
        scale = VARIABLES[var_name]['scale']
        if scale != 1.0: diff = diff * scale

        # Add Cyclic Point to avoid white line at 0 deg
        data = diff.values
        # Assuming last dim is lon
        if len(data.shape) == 2:
             # handle lat/lon vars
             # Check dims order
             lons = diff.lon.values
             lats = diff.lat.values
             
             data_cyc, lons_cyc = add_cyclic_point(data, coord=lons)
        else:
            # Fallback for non 2d
            data_cyc = data
            lons_cyc = diff.lon.values
            lats = diff.lat.values

        # Plotting
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()

        # Robust color range
        data_flat = data.flatten()
        data_flat = data_flat[~np.isnan(data_flat)]
        if len(data_flat) == 0:
            print(f"  No valid data for {var_name}")
            plt.close(fig)
            return

        vmax = np.percentile(np.abs(data_flat), 98)
        vmin = -vmax

        im = ax.pcolormesh(lons_cyc, lats, data_cyc, 
                           transform=ccrs.PlateCarree(),
                           cmap=VARIABLES[var_name]['cmap'],
                           vmin=vmin, vmax=vmax,
                           shading='auto')
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label(f"Diff ({exp_name} - REF) {VARIABLES[var_name]['label']}")
        
        plt.title(f"{var_name} Difference: {exp_name} - {REF_EXP} ({season})")
        
        out_name = f"{exp_name}_diff_{var_name}_{season}.png"
        out_path = os.path.join(PLOT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_name}")

    except Exception as e:
        print(f"  Error plotting {var_name}: {e}")

def process_experiment(exp_name):
    print(f"Processing plots for {exp_name}...")
    
    for season in SEASONS:
        ref_surf_path = get_clim_file(REF_EXP, 'surf', season)
        exp_surf_path = get_clim_file(exp_name, 'surf', season)

        if not (ref_surf_path and exp_surf_path):
            # print(f"  Missing SURF files for {season}")
            continue

        try:
            with xr.open_dataset(ref_surf_path) as ref_surf, \
                 xr.open_dataset(exp_surf_path) as exp_surf:
                
                for v in ['TS', 'TPRECTOT', 'SWGNET']:
                    if v in VARIABLES:
                        plot_difference(exp_name, season, v, ref_surf, exp_surf)
                        
        except Exception as e:
            print(f"  Error loading datasets: {e}")

def main():
    print("Starting Plotting Script (Cyclic)...")
    all_items = sorted(os.listdir(BASE_DIR))
    experiments = [d for d in all_items if d.startswith('GF') and d != REF_EXP and os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for exp in experiments:
        process_experiment(exp)
        
    print("Plotting Complete.")

if __name__ == "__main__":
    main()
