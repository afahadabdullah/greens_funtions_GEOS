
import xarray as xr
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
ROOT_DIR = "/nobackupp27/afahad/GFEXPv11.7"
MERRA2_DIR = os.path.join(ROOT_DIR, "target/m2")
EXP_REF = 'GF01_ref'
EXP_OPT = 'GF_OPT'
EXP_NEW = 'GF_OPT72'
EXP_UPD = 'GF_OPTUpd'

def get_file_path(base_dirs, year, month, collection_name='geosgcm_surf'):
    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]
    for base_dir in base_dirs:
        if not os.path.exists(base_dir): continue
        pat1 = os.path.join(base_dir, f"*.{collection_name}.*{year}{month:02d}*.nc4")
        f1 = glob.glob(pat1)
        if f1: return f1[0]
        pat2 = os.path.join(base_dir, f"{year}{month:02d}", f"*.{collection_name}.*{year}{month:02d}*.nc4")
        f2 = glob.glob(pat2)
        if f2: return f2[0]
    return None

def get_merra2_path(base_dir, year, month):
    pat = os.path.join(base_dir, f"MERRA2_*.instM_2d_asm_Nx.{year}{month:02d}.SUB.nc")
    f = glob.glob(pat)
    return f[0] if f else None

def calc_global_mean(ds, var_name):
    data = ds[var_name]
    lat_name = 'lat' if 'lat' in data.coords else 'YDim'
    weights = np.cos(np.deg2rad(ds[lat_name]))
    data_weighted = data.weighted(weights)
    return data_weighted.mean(dim=[lat_name, "lon"]).values.item()

def calc_global_ocean_mean(ds, var_name, frocean_ref=None):
    data = ds[var_name]
    lat_name = 'lat' if 'lat' in data.coords else 'YDim'
    if 'FROCEAN' in ds:
        frocean = ds['FROCEAN']
    elif frocean_ref is not None:
        try:
             frocean = frocean_ref.interp_like(data, method='nearest', kwargs={"fill_value": "extrapolate"}).fillna(0)
        except:
             frocean = xr.ones_like(data)
    else:
        frocean = xr.ones_like(data)
    weights = np.cos(np.deg2rad(ds[lat_name]))
    total_weights = weights * frocean
    if float(total_weights.sum()) == 0: return np.nan
    return data.weighted(total_weights).mean(dim=[lat_name, "lon"]).values.item()

def get_reference_mask():
    fpath = get_file_path([os.path.join(ROOT_DIR, EXP_REF, 'holding/geosgcm_surf')], 2005, 8)
    if fpath:
        with xr.open_dataset(fpath) as ds:
            if 'FROCEAN' in ds: return ds['FROCEAN'].load()
    return None

def calc_nino34_index(ds, var_name):
    data = ds[var_name]
    lat_name = 'lat' if 'lat' in data.coords else 'YDim'
    lon_name = 'lon' if 'lon' in data.coords else 'XDim'
    try:
        data_nino = data.sel({lat_name: slice(-5, 5), lon_name: slice(-170, -120)})
        if data_nino.size == 0:
             data_nino = data.sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})
        weights = np.cos(np.deg2rad(data_nino[lat_name]))
        data_weighted = data_nino.weighted(weights)
        return data_weighted.mean(dim=[lat_name, lon_name]).values.item()
    except: return np.nan

def generate_progressive_timeseries():
    print("\nGenerating Progressive Time Series Plots...")
    ref_mask = get_reference_mask()
    
    experiments = {
        EXP_REF: {'dir': os.path.join(ROOT_DIR, EXP_REF), 'color': 'blue', 'lw': 2, 'style': '-', 'label': EXP_REF},
        'MERRA2': {'dir': MERRA2_DIR, 'color': 'black', 'lw': 2, 'style': '--', 'label': 'MERRA2'},
        EXP_OPT: {'dir': os.path.join(ROOT_DIR, EXP_OPT), 'color': 'green', 'lw': 3, 'style': '-', 'label': EXP_OPT},
        EXP_NEW: {'dir': os.path.join(ROOT_DIR, EXP_NEW), 'color': 'red', 'lw': 3, 'style': '-', 'label': EXP_NEW},
        EXP_UPD: {'dir': os.path.join(ROOT_DIR, EXP_UPD), 'color': 'purple', 'lw': 3, 'style': '-', 'label': EXP_UPD}
    }
    
    dates = {k: [] for k in experiments}
    vals_global = {k: [] for k in experiments}
    vals_ocean = {k: [] for k in experiments}
    vals_nino34 = {k: [] for k in experiments}
    
    # Pre-load data for 2005-2008
    for year in range(2005, 2009):
        for month in range(1, 13):
            for name, cfg in experiments.items():
                if name == 'MERRA2':
                    fpath = get_merra2_path(cfg['dir'], year, month)
                else:
                    dirs = [os.path.join(cfg['dir'], 'holding', 'geosgcm_surf'), os.path.join(cfg['dir'], 'geosgcm_surf')]
                    fpath = get_file_path(dirs, year, month)
                if fpath:
                    try:
                        with xr.open_dataset(fpath) as ds:
                            if 'TS' in ds:
                                d = np.datetime64(f"{year}-{month:02d}-01")
                                vals_global[name].append(calc_global_mean(ds, 'TS'))
                                vals_ocean[name].append(calc_global_ocean_mean(ds, 'TS', frocean_ref=ref_mask))
                                vals_nino34[name].append(calc_nino34_index(ds, 'TS'))
                                dates[name].append(d)
                    except: pass

    # Plotting steps
    steps = [
        [EXP_REF, 'MERRA2'],
        [EXP_REF, 'MERRA2', EXP_OPT],
        [EXP_REF, 'MERRA2', EXP_OPT, EXP_NEW],
        [EXP_REF, 'MERRA2', EXP_OPT, EXP_NEW, EXP_UPD]
    ]

    for i, active_exps in enumerate(steps, 1):
        print(f"Saving Step {i} plots: {active_exps}")
        
        # Plot Global TS
        plt.figure(figsize=(12, 6))
        for name in active_exps:
            cfg = experiments[name]
            if vals_global[name]:
                plt.plot(dates[name], vals_global[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
        plt.title(f'Global Mean Surface Temperature (TS) - Step {i}')
        plt.ylabel('Temperature (K)'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.savefig(f"Global_TS_Step{i}.png", dpi=300); plt.close()

        # Plot Ocean SST
        plt.figure(figsize=(12, 6))
        for name in active_exps:
            cfg = experiments[name]
            if vals_ocean[name]:
                plt.plot(dates[name], vals_ocean[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
        plt.title(f'Global OCEAN Mean Surface Temperature (SST) - Step {i}')
        plt.ylabel('Temperature (K)'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.savefig(f"Global_SST_Step{i}.png", dpi=300); plt.close()

        # Plot Niño 3.4
        plt.figure(figsize=(12, 6))
        for name in active_exps:
            cfg = experiments[name]
            if vals_nino34[name]:
                plt.plot(dates[name], vals_nino34[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
        plt.title(f'Nino 3.4 Region SST (5N-5S, 170W-120W) - Step {i}')
        plt.ylabel('Temperature (K)'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.savefig(f"Nino34_Step{i}.png", dpi=300); plt.close()

if __name__ == "__main__":
    generate_progressive_timeseries()
