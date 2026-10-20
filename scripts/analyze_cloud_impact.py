
import xarray as xr
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = "/nobackupp27/afahad/GFEXPv11.7"
EXP_REF = 'GF01_ref'
EXP_OPT = 'GF_OPT'
EXP_NEW = 'GF_OPT72'

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

def get_reference_mask():
    fpath = get_file_path([os.path.join(ROOT_DIR, EXP_REF, 'holding/geosgcm_surf')], 2005, 8)
    if fpath:
        with xr.open_dataset(fpath) as ds:
            if 'FROCEAN' in ds: return ds['FROCEAN'].load()
    return None

def weighted_correlation(da1, da2, weights):
    """Calculates weighted spatial correlation coefficient."""
    # Flatten
    v1 = da1.values.flatten()
    v2 = da2.values.flatten()
    w = weights.values.flatten()
    
    # Mask NaNs
    mask = ~np.isnan(v1) & ~np.isnan(v2) & ~np.isnan(w)
    v1 = v1[mask]
    v2 = v2[mask]
    w = w[mask]
    
    # Weighted Mean
    m1 = np.average(v1, weights=w)
    m2 = np.average(v2, weights=w)
    
    # Weighted Covariance
    num = np.sum(w * (v1 - m1) * (v2 - m2))
    
    # Weighted Std Dev
    den1 = np.sqrt(np.sum(w * (v1 - m1)**2))
    den2 = np.sqrt(np.sum(w * (v2 - m2)**2))
    
    if den1 == 0 or den2 == 0: return np.nan
    
    return num / (den1 * den2)

def get_last_mean(exp_name, end_year, end_month, months=24, var_list=None):
    import pandas as pd
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1)
    start_date = end_date - pd.DateOffset(months=months-1)
    date_range = pd.period_range(start_date, end_date, freq='M')
    datasets = []
    for d in date_range:
        y, m = d.year, d.month
        dirs = [os.path.join(ROOT_DIR, exp_name, 'holding/geosgcm_surf'), os.path.join(ROOT_DIR, exp_name, 'geosgcm_surf')]
        path = get_file_path(dirs, y, m)
        if path and os.path.exists(path):
            try:
                ds = xr.open_dataset(path)
                if var_list: ds = ds[list(set(var_list) & set(ds.data_vars))]
                datasets.append(ds)
            except: pass
    if not datasets: return None
    combined = xr.concat(datasets, dim='time')
    return combined.mean(dim='time', keep_attrs=True)

def analyze():
    print("Calculating correlations (Mean of last 24 months: Aug 2005 - July 2007)")
    end_year, end_month = 2007, 7
    vars_to_load = ['TS', 'CLDLO', 'CLDMD', 'SWGNET', 'LHFX', 'FROCEAN']
    
    ds_ref = get_last_mean(EXP_REF, end_year, end_month, months=24, var_list=vars_to_load)
    ds_opt = get_last_mean(EXP_OPT, end_year, end_month, months=24, var_list=vars_to_load)
    ds_new = get_last_mean(EXP_NEW, end_year, end_month, months=24, var_list=vars_to_load)
    
    if not ds_ref or not ds_opt or not ds_new:
        print("Data missing.")
        return

    # Differences
    diff_opt = ds_opt - ds_ref
    diff_new = ds_new - ds_ref
    
    # Masks
    frocean = ds_ref['FROCEAN']
    # Create 2D weights
    weights_1d = np.cos(np.deg2rad(ds_ref['lat']))
    weights, _ = xr.broadcast(weights_1d, frocean)
    
    w_ocean = weights * frocean
    w_land = weights * (1 - frocean)
    
    print("\n--- Correlation Analysis ---")
    print("Format: (Variable vs TS Diff) -> Correlation Coefficient [Global, Ocean, Land]")
    
    for exp_name, diff_ds in [("GF_OPT - GF01", diff_opt), ("GF_OPT72 - GF01", diff_new)]:
        print(f"\nExperiment: {exp_name}")
        
        for var in ['CLDLO', 'CLDMD', 'SWGNET', 'LHFX']:
            if var not in diff_ds: continue
            
            ts_diff = diff_ds['TS']
            var_diff = diff_ds[var]
            
            # Global
            corr_g = weighted_correlation(var_diff, ts_diff, weights)
            # Ocean
            corr_o = weighted_correlation(var_diff, ts_diff, w_ocean)
            # Land
            corr_l = weighted_correlation(var_diff, ts_diff, w_land)
            
            print(f"  Delta {var} vs Delta TS: Global={corr_g:.2f}, Ocean={corr_o:.2f}, Land={corr_l:.2f}")

    print("\n--- Radiative Check ---")
    print("Correlation between Delta Clouds and Delta SWGNET (Check for albedo effect)")
    for exp_name, diff_ds in [("GF_OPT - GF01", diff_opt), ("GF_OPT72 - GF01", diff_new)]:
        print(f"\nExperiment: {exp_name}")
        for var in ['CLDLO', 'CLDMD']:
             sw_diff = diff_ds['SWGNET']
             var_diff = diff_ds[var]
             corr_g = weighted_correlation(var_diff, sw_diff, weights)
             print(f"  Delta {var} vs Delta SWGNET: Global={corr_g:.2f}")

if __name__ == "__main__":
    analyze()
