
import xarray as xr
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

# Comparison Variables for Maps
MAP_VARS = ['FRSEAICE', 'TS', 'PRECTOT', 'SWGNET', 'LHFX', 'CLDLO', 'CLDMD', 'CLDHI']

def get_file_path(base_dirs, year, month, collection_name='geosgcm_surf'):
    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]
    for base_dir in base_dirs:
        if not os.path.exists(base_dir): continue
        # Permissive pattern to handle optional '.monthly.' or other infixes
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
    """
    Calculates Area-Weighted Mean SST for Nino 3.4 Region.
    Region: 5S-5N, 170W-120W
    Note: Assuming coords are -180 to 180. 170W = -170, 120W = -120.
    """
    data = ds[var_name]
    lat_name = 'lat' if 'lat' in data.coords else 'YDim'
    lon_name = 'lon' if 'lon' in data.coords else 'XDim'
    
    # Select Region
    try:
        data_nino = data.sel({lat_name: slice(-5, 5), lon_name: slice(-170, -120)})
        if data_nino.size == 0:
             # Try other convention if needed (e.g. 0-360: 190 to 240)
             data_nino = data.sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})
             
        weights = np.cos(np.deg2rad(data_nino[lat_name]))
        data_weighted = data_nino.weighted(weights)
        return data_weighted.mean(dim=[lat_name, lon_name]).values.item()
    except Exception as e:
        print(f"Error calc Nino3.4: {e}")
        return np.nan

# --- Time Series Logic ---
def generate_timeseries():
    print("\nGenerating Time Series Plots...")
    ref_mask = get_reference_mask()
    
    experiments = {
        EXP_REF: {'dir': os.path.join(ROOT_DIR, EXP_REF), 'color': 'blue', 'lw': 2, 'style': '-', 'label': EXP_REF},
        EXP_OPT: {'dir': os.path.join(ROOT_DIR, EXP_OPT), 'color': 'green', 'lw': 3, 'style': '-', 'label': EXP_OPT},
        EXP_NEW: {'dir': os.path.join(ROOT_DIR, EXP_NEW), 'color': 'red', 'lw': 3, 'style': '-', 'label': EXP_NEW},
        EXP_UPD: {'dir': os.path.join(ROOT_DIR, EXP_UPD), 'color': 'purple', 'lw': 3, 'style': '-', 'label': EXP_UPD},
        'MERRA2': {'dir': MERRA2_DIR, 'color': 'black', 'lw': 2, 'style': '--', 'label': 'MERRA2'}
    }
    
    dates = {k: [] for k in experiments}
    vals_global = {k: [] for k in experiments}
    vals_ocean = {k: [] for k in experiments}
    vals_nino34 = {k: [] for k in experiments}
    
    for year in range(2005, 2009): # End of 2008
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
                                # Global
                                vg = calc_global_mean(ds, 'TS')
                                vals_global[name].append(vg)
                                # Ocean
                                vo = calc_global_ocean_mean(ds, 'TS', frocean_ref=ref_mask)
                                vals_ocean[name].append(vo)
                                # Nino 3.4
                                vn = calc_nino34_index(ds, 'TS')
                                vals_nino34[name].append(vn)
                                
                                dates[name].append(d)
                    except: pass
    
    # Plot Global
    plt.figure(figsize=(12, 6))
    for name, cfg in experiments.items():
        if vals_global[name]:
             plt.plot(dates[name], vals_global[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
    plt.title('Global Mean Surface Temperature (TS)')
    plt.ylabel('Temperature (K)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.savefig("Global_TS_Comparison.png", dpi=300)
    print("Saved Global_TS_Comparison.png")

    # Plot Ocean
    plt.figure(figsize=(12, 6))
    for name, cfg in experiments.items():
        if vals_ocean[name]:
             plt.plot(dates[name], vals_ocean[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
    plt.title('Global OCEAN Mean Surface Temperature (SST)')
    plt.ylabel('Temperature (K)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.savefig("Global_SST_Comparison.png", dpi=300)
    print("Saved Global_SST_Comparison.png")
    
    # Plot Nino 3.4
    plt.figure(figsize=(12, 6))
    for name, cfg in experiments.items():
        if vals_nino34[name]:
             plt.plot(dates[name], vals_nino34[name], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['style'], label=cfg['label'], marker='o', markersize=4)
    plt.title('Nino 3.4 Region SST (5N-5S, 170W-120W)')
    plt.ylabel('Temperature (K)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.savefig("Nino34_Comparison.png", dpi=300)
    print("Saved Nino34_Comparison.png")

# --- Map Logic ---
def find_last_month(exp_name):
    base_dirs = [os.path.join(ROOT_DIR, exp_name, 'holding/geosgcm_surf'), os.path.join(ROOT_DIR, exp_name, 'geosgcm_surf')]
    last_ym = None
    
    # Simple search: look for files and parse date
    # Walk and sort
    all_files = []
    for d in base_dirs:
        if os.path.exists(d):
            # Try flat
            # Use more permissive pattern here too
            files = glob.glob(os.path.join(d, "*.geosgcm_surf.*.nc4"))
            all_files.extend(files)
            # Try subdirs
            files_sub = glob.glob(os.path.join(d, "*/*.geosgcm_surf.*.nc4"))
            all_files.extend(files_sub)
    
    print(f"DEBUG: Found {len(all_files)} files for {exp_name}")
    if not all_files: return None
    
    sorted_files = sorted(all_files)
    last_file = sorted_files[-1]
    print(f"DEBUG: Last file found: {last_file}")
    
    # Parse date from filename: ...20050801...
    import re
    # Update regex to be robust for .monthly.infix
    m = re.search(r'\.(\d{6})\d*.*\.nc4', os.path.basename(last_file))
    if m:
        ym = m.group(1) # 200508
        print(f"DEBUG: Parsed date: {ym}")
        return int(ym[:4]), int(ym[4:])
    print("DEBUG: Regex failed on last file.")
    return None

def get_last_mean(exp_name, end_year, end_month, months=24, is_merra=False, var_list=None):
    """
    Calculates the mean of the last N months ending at (end_year, end_month).
    Returns a Dataset containing the mean of available overlapping variables.
    """
    
    # Generate list of (year, month) tuples
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1)
    start_date = end_date - pd.DateOffset(months=months-1)
    
    date_range = pd.period_range(start_date, end_date, freq='M')
    print(f"DEBUG: Computing {months}-month mean for {exp_name} from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    datasets = []
    valid_count = 0
    
    for d in date_range:
        y, m = d.year, d.month
        path = None
        if is_merra:
             path = get_merra2_path(MERRA2_DIR, y, m)
        else:
             dirs = [os.path.join(ROOT_DIR, exp_name, 'holding/geosgcm_surf'), os.path.join(ROOT_DIR, exp_name, 'geosgcm_surf')]
             path = get_file_path(dirs, y, m)
             
        if path and os.path.exists(path):
            try:
                ds = xr.open_dataset(path)
                # Keep only relevant vars to save memory/time if list provided
                if var_list:
                    ds = ds[list(set(var_list) & set(ds.data_vars))]
                datasets.append(ds)
                valid_count += 1
            except Exception as e:
                print(f"Warning: Failed to open {path}: {e}")
        else:
            print(f"Warning: Missing file for {exp_name} at {y}-{m:02d}")

    if not datasets:
        return None

    print(f"DEBUG: aggregation found {valid_count}/{months} files for {exp_name}")
    
    # Concatenate and mean
    # We assign a dummy time dimension to concat easily
    combined = xr.concat(datasets, dim='time')
    mean_ds = combined.mean(dim='time', keep_attrs=True)
    return mean_ds

def generate_maps():
    print("\nGenerating Global Maps (12-Month Mean: Jan 2005 - Dec 2005)...")
    
    # Force 2005-2008 Analysis
    end_year, end_month = 2008, 12
    months_duration = 48
    
    # Get Means
    print(f"Calculating means ending {end_year}-{end_month:02d}...")
    
    ds_ref = get_last_mean(EXP_REF, end_year, end_month, months=months_duration, var_list=MAP_VARS)
    ds_opt = get_last_mean(EXP_OPT, end_year, end_month, months=months_duration, var_list=MAP_VARS)
    ds_new = get_last_mean(EXP_NEW, end_year, end_month, months=months_duration, var_list=MAP_VARS)
    ds_upd = get_last_mean(EXP_UPD, end_year, end_month, months=months_duration, var_list=MAP_VARS)
    
    if not ds_ref:
        print("Missing REF dataset.")
        return
        
    start_dt = pd.Timestamp(year=end_year, month=end_month, day=1) - pd.DateOffset(months=months_duration-1)
    end_dt = pd.Timestamp(year=end_year, month=end_month, day=1)
    title_suffix = f"Mean: {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}"
    
    for var in MAP_VARS:
        if var in ds_ref:
            print(f"Plotting {var}...")
            
            # Special logic for Cloud Fraction Variables: 2x4 Layout
            if var in ['CLDLO', 'CLDMD', 'CLDHI']:
                fig, axes = plt.subplots(2, 4, figsize=(24, 10), subplot_kw={'projection': ccrs.Robinson()})
                
                # Use common limits for Means
                vmin = ds_ref[var].min().item()
                vmax = ds_ref[var].max().item()
                
                # --- Row 1: Means ---
                # 1. GF01
                ax1 = axes[0, 0]; ax1.set_global(); ax1.coastlines(linewidth=0.5)
                ds_ref[var].squeeze().plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, cbar_kwargs={'shrink':0.6, 'label': 'Fraction'})
                ax1.set_title(f"{EXP_REF}\n{title_suffix}")
                
                # 2. GF_OPT
                ax2 = axes[0, 1]; ax2.set_global(); ax2.coastlines(linewidth=0.5)
                if ds_opt and var in ds_opt:
                    ds_opt[var].squeeze().plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, cbar_kwargs={'shrink':0.6, 'label': 'Fraction'})
                    ax2.set_title(f"{EXP_OPT}")
                else: ax2.set_title(f"{EXP_OPT} Missing")

                # 3. GF_OPT72
                ax3 = axes[0, 2]; ax3.set_global(); ax3.coastlines(linewidth=0.5)
                if ds_new and var in ds_new:
                    ds_new[var].squeeze().plot(ax=ax3, transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, cbar_kwargs={'shrink':0.6, 'label': 'Fraction'})
                    ax3.set_title(f"{EXP_NEW}")
                else: ax3.set_title(f"{EXP_NEW} Missing")
                
                # 4. GF_OPTUpd
                ax4 = axes[0, 3]; ax4.set_global(); ax4.coastlines(linewidth=0.5)
                if ds_upd and var in ds_upd:
                    ds_upd[var].squeeze().plot(ax=ax4, transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, cbar_kwargs={'shrink':0.6, 'label': 'Fraction'})
                    ax4.set_title(f"{EXP_UPD}")
                else: ax4.set_title(f"{EXP_UPD} Missing")
                
                # --- Row 2: Differences ---
                max_diff = 0
                for d in [ds_opt, ds_new, ds_upd]:
                    if d and var in d:
                        diff = d[var] - ds_ref[var]
                        if diff.count() > 0:
                             max_diff = max(max_diff, abs(diff.min()), abs(diff.max()))
                if max_diff == 0: max_diff = 0.1
                
                # 5. Blank
                ax5 = axes[1, 0]; ax5.set_visible(False)
                
                # 6. Opt - Ref
                ax6 = axes[1, 1]; ax6.set_global(); ax6.coastlines(linewidth=0.5)
                if ds_opt and var in ds_opt:
                    (ds_opt[var] - ds_ref[var]).plot(ax=ax6, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6, 'label': 'Diff'})
                    ax6.set_title(f"{EXP_OPT} - {EXP_REF}")
                
                # 7. New - Ref
                ax7 = axes[1, 2]; ax7.set_global(); ax7.coastlines(linewidth=0.5)
                if ds_new and var in ds_new:
                    (ds_new[var] - ds_ref[var]).plot(ax=ax7, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6, 'label': 'Diff'})
                    ax7.set_title(f"{EXP_NEW} - {EXP_REF}")
                    
                # 8. Upd - Ref
                ax8 = axes[1, 3]; ax8.set_global(); ax8.coastlines(linewidth=0.5)
                if ds_upd and var in ds_upd:
                    (ds_upd[var] - ds_ref[var]).plot(ax=ax8, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6, 'label': 'Diff'})
                    ax8.set_title(f"{EXP_UPD} - {EXP_REF}")
                
                plt.suptitle(f"{var} 12-Month Mean Comparison", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"Map_Comparison_{var}.png", dpi=200)
                print(f"Saved Map_Comparison_{var}.png")
                continue 

            # Special handling for TS (2x4 with MERRA2)
            if var == 'TS':
                ds_m2_mean = get_last_mean('MERRA2', end_year, end_month, months=months_duration, is_merra=True, var_list=['TS'])
                ds_m2 = None
                if ds_m2_mean:
                    try:
                        ds_m2 = ds_m2_mean.interp_like(ds_ref, method='nearest', kwargs={"fill_value": "extrapolate"})
                    except: pass

                fig, axes = plt.subplots(2, 4, figsize=(24, 10), subplot_kw={'projection': ccrs.Robinson()})
                
                # Row 1: Ref, Opt-Ref, New-Ref, Upd-Ref
                # 1. Ref Actual
                ax1 = axes[0, 0]; ax1.set_global(); ax1.coastlines(linewidth=0.5)
                ds_ref[var].squeeze().plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='viridis', vmin=220, vmax=310, cbar_kwargs={'shrink':0.6,})
                ax1.set_title(f"{EXP_REF} (Actual)")
                
                # 2. Opt-Ref
                ax2 = axes[0, 1]; ax2.set_global(); ax2.coastlines(linewidth=0.5)
                if ds_opt: (ds_opt[var] - ds_ref[var]).plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-5, vmax=5, cbar_kwargs={'shrink':0.6,})
                ax2.set_title(f"{EXP_OPT} - {EXP_REF}")
                
                # 3. New-Ref
                ax3 = axes[0, 2]; ax3.set_global(); ax3.coastlines(linewidth=0.5)
                if ds_new: (ds_new[var] - ds_ref[var]).plot(ax=ax3, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-5, vmax=5, cbar_kwargs={'shrink':0.6,})
                ax3.set_title(f"{EXP_NEW} - {EXP_REF}")
                
                # 4. Upd-Ref
                ax4 = axes[0, 3]; ax4.set_global(); ax4.coastlines(linewidth=0.5)
                if ds_upd: (ds_upd[var] - ds_ref[var]).plot(ax=ax4, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-5, vmax=5, cbar_kwargs={'shrink':0.6,})
                ax4.set_title(f"{EXP_UPD} - {EXP_REF}")
                
                # Row 2: Differences from MERRA2
                levs = np.arange(-7.5, 7.6, 1)
                
                # 5. Ref-M2
                ax5 = axes[1, 0]; ax5.set_global(); ax5.coastlines(linewidth=0.5)
                if ds_m2: (ds_ref[var] - ds_m2[var]).plot(ax=ax5, transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levs, cbar_kwargs={'shrink':0.6,})
                ax5.set_title(f"{EXP_REF} - MERRA2")
                
                # 6. Opt-M2
                ax6 = axes[1, 1]; ax6.set_global(); ax6.coastlines(linewidth=0.5)
                if ds_m2 and ds_opt: (ds_opt[var] - ds_m2[var]).plot(ax=ax6, transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levs, cbar_kwargs={'shrink':0.6,})
                ax6.set_title(f"{EXP_OPT} - MERRA2")
                
                # 7. New-M2
                ax7 = axes[1, 2]; ax7.set_global(); ax7.coastlines(linewidth=0.5)
                if ds_m2 and ds_new: (ds_new[var] - ds_m2[var]).plot(ax=ax7, transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levs, cbar_kwargs={'shrink':0.6,})
                ax7.set_title(f"{EXP_NEW} - MERRA2")
                
                # 8. Upd-M2
                ax8 = axes[1, 3]; ax8.set_global(); ax8.coastlines(linewidth=0.5)
                if ds_m2 and ds_upd: (ds_upd[var] - ds_m2[var]).plot(ax=ax8, transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levs, cbar_kwargs={'shrink':0.6,})
                ax8.set_title(f"{EXP_UPD} - MERRA2")

                plt.suptitle(f"{var} 12-Month Mean Comparison\n{title_suffix}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"Map_Comparison_{var}.png", dpi=200)
                print(f"Saved Map_Comparison_{var}.png")
                
            else:
                # Standard Logic: 1x4 (Ref, Opt-Ref, New-Ref, Upd-Ref)
                fig, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': ccrs.Robinson()})
                
                # 1. Ref Actual
                ax1 = axes[0]; ax1.set_global(); ax1.coastlines(linewidth=0.5)
                ds_ref[var].squeeze().plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'shrink':0.6})
                ax1.set_title(f"{EXP_REF} (Actual)")
                
                # Calc Max Diff
                max_diff = 0
                for d in [ds_opt, ds_new, ds_upd]:
                    if d:
                        diff = d[var] - ds_ref[var]
                        max_diff = max(max_diff, abs(diff.min()), abs(diff.max()))
                
                if var == 'FRSEAICE': max_diff = 0.5
                
                # 2. Opt-Ref
                ax2 = axes[1]; ax2.set_global(); ax2.coastlines(linewidth=0.5)
                if ds_opt: (ds_opt[var] - ds_ref[var]).plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6})
                ax2.set_title(f"{EXP_OPT} - {EXP_REF}")
                
                # 3. New-Ref
                ax3 = axes[2]; ax3.set_global(); ax3.coastlines(linewidth=0.5)
                if ds_new: (ds_new[var] - ds_ref[var]).plot(ax=ax3, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6})
                ax3.set_title(f"{EXP_NEW} - {EXP_REF}")
                
                # 4. Upd-Ref
                ax4 = axes[3]; ax4.set_global(); ax4.coastlines(linewidth=0.5)
                if ds_upd: (ds_upd[var] - ds_ref[var]).plot(ax=ax4, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, cbar_kwargs={'shrink':0.6})
                ax4.set_title(f"{EXP_UPD} - {EXP_REF}")
                
                plt.suptitle(f"{var} 12-Month Mean Comparison\n{title_suffix}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"Map_Comparison_{var}.png", dpi=200)
                print(f"Saved Map_Comparison_{var}.png")

def main():
    generate_timeseries()
    generate_maps()

if __name__ == "__main__":
    main()
