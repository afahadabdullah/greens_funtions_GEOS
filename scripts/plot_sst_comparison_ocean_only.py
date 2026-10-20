
import xarray as xr
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_file_path(base_dirs, year, month, collection_name):
    """
    Constructs the file path searching in multiple potential directories.
    """
    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]
        
    for base_dir in base_dirs:
        if not os.path.exists(base_dir): continue
        
        # Pattern 1: Files directly in base_dir
        file_pattern = f"*.{collection_name}.{year}{month:02d}*.nc4"
        full_pattern_direct = os.path.join(base_dir, file_pattern)
        files = glob.glob(full_pattern_direct)
        
        if not files:
            file_pattern_alt = f"*{collection_name}.{year}{month:02d}*.nc4"
            full_pattern_alt_direct = os.path.join(base_dir, file_pattern_alt)
            files = glob.glob(full_pattern_alt_direct)
            
        if files: return files[0]
        
        # Pattern 2: Files in YYYYMM subdirectory
        ym_subdir = f"{year}{month:02d}"
        subdir_path = os.path.join(base_dir, ym_subdir)
        if os.path.exists(subdir_path):
             full_pattern_subdir = os.path.join(subdir_path, file_pattern)
             files = glob.glob(full_pattern_subdir)
             
             if not files:
                 file_pattern_alt = f"*{collection_name}.{year}{month:02d}*.nc4"
                 full_pattern_alt_subdir = os.path.join(subdir_path, file_pattern_alt)
                 files = glob.glob(full_pattern_alt_subdir)
                 
             if files: return files[0]

    return None

def get_merra2_path(base_dir, year, month):
    """Find MERRA2 file for a given year and month."""
    # Pattern: MERRA2_*.instM_2d_asm_Nx.YYYYMM.SUB.nc
    pattern = f"MERRA2_*.instM_2d_asm_Nx.{year}{month:02d}.SUB.nc"
    full_pattern = os.path.join(base_dir, pattern)
    files = glob.glob(full_pattern)
    if files:
        return files[0]
    return None

def calc_global_ocean_mean(ds, var_name, frocean_ref=None):
    """
    Calculate area-weighted global mean over OCEAN only.
    Uses 'FROCEAN' if available, otherwise interpolates frocean_ref.
    """
    data = ds[var_name]
    
    # Check for correct latitude name
    if 'lat' in data.coords:
        lat = data.coords['lat']
    elif 'YDim' in data.coords: 
        lat = data.coords['YDim']
    else:
        lat = ds.coords['lat']
        
    # Get FROCEAN
    if 'FROCEAN' in ds:
        frocean = ds['FROCEAN']
    elif frocean_ref is not None:
        try:
            # Extrapolate to cover edges
            frocean = frocean_ref.interp_like(data, method='nearest', kwargs={"fill_value": "extrapolate"})
            frocean = frocean.fillna(0) # Ensure no NaNs
        except Exception as e:
            print(f"    Interpolation error: {e}")
            frocean = xr.ones_like(data)
    else:
        frocean = xr.ones_like(data)

    # Weights = Area * OceanFraction
    weights = np.cos(np.deg2rad(lat))
    weights.name = "weights"
    
    # Total Weight = weights * frocean
    total_weights = weights * frocean
    
    # Debug for MERRA2 (or generally if suspicious)
    if 'FROCEAN' not in ds and np.random.rand() < 0.1: # Sample debug for MERRA2
         print(f"    [DEBUG] Mask Mean: {float(frocean.mean()):.4f}, Data Mean: {float(data.mean()):.2f}")
         print(f"    [DEBUG] Total Weights Sum: {float(total_weights.sum()):.4f}")

    # Check if weights are valid
    if float(total_weights.sum()) == 0:
        return np.nan
        
    # Calculate Weighted Mean
    data_weighted = data.weighted(total_weights)
    val = data_weighted.mean(dim=["lat", "lon"]).values.item()
    return val

def get_reference_mask():
    """Get a FROCEAN mask from a reference file (GF01)."""
    # Pick a sample file from GF01
    root_dir = "/nobackupp27/afahad/GFEXPv11.7"
    exp_dir = os.path.join(root_dir, 'GF01_ref')
    surf_dirs = [os.path.join(exp_dir, 'holding', 'geosgcm_surf')]
    
    # Try to find one file
    fpath = get_file_path(surf_dirs, 2005, 8, 'geosgcm_surf')
    if fpath:
        with xr.open_dataset(fpath) as ds:
            if 'FROCEAN' in ds:
                return ds['FROCEAN'].load() # Load into memory
    return None

def process_and_plot():
    root_dir = "/nobackupp27/afahad/GFEXPv11.7"
    merra2_dir = os.path.join(root_dir, "target/m2")
    
    # Experiments configuration
    experiments = {
        'GF01_ref': {
            'dir': os.path.join(root_dir, 'GF01_ref'), 
            'color': 'blue', 'linewidth': 2, 'style': '-', 'label': 'GF01_ref'
        },
        'GF_OPT': {
            'dir': os.path.join(root_dir, 'GF_OPT'), 
            'color': 'green', 'linewidth': 3, 'style': '-', 'label': 'GF_OPT'
        },
        'GF_OPT72': {
            'dir': os.path.join(root_dir, 'GF_OPT72'), 
            'color': 'red', 'linewidth': 3, 'style': '-', 'label': 'GF_OPT72'
        },
        'MERRA2': {
            'dir': merra2_dir,
            'color': 'black', 'linewidth': 2, 'style': '--', 'label': 'MERRA2'
        }
    }
    
    # Get Reference Mask for MERRA2 interpolation
    print("Loading Reference Ocean Mask...")
    ref_mask = get_reference_mask()
    if ref_mask is None:
        print("Error: Could not load reference ocean mask.")
        return

    # Store time series
    results = {}
    
    start_year = 2005
    end_year = 2010 
    
    for name, config in experiments.items():
        print(f"Processing {name} (Ocean Only)...")
        base_dir = config['dir']
        
        if not os.path.exists(base_dir):
            print(f"  Warning: Directory {base_dir} does not exist.")
            continue
            
        dates = []
        values = []
        
        # Iterate over all months
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                
                fpath = None
                if name == 'MERRA2':
                    fpath = get_merra2_path(base_dir, year, month)
                else:
                    surf_dirs = [
                        os.path.join(base_dir, 'holding', 'geosgcm_surf'),
                        os.path.join(base_dir, 'geosgcm_surf')
                    ]
                    fpath = get_file_path(surf_dirs, year, month, 'geosgcm_surf')
                
                if fpath:
                    try:
                        with xr.open_dataset(fpath) as ds:
                            if 'TS' in ds:
                                # USE OCEAN MEAN
                                val = calc_global_ocean_mean(ds, 'TS', frocean_ref=ref_mask)
                                values.append(val)
                                dates.append(np.datetime64(f"{year}-{month:02d}-01"))
                    except Exception as e:
                        print(f"Error processing {fpath}: {e}")
                        pass
        
        if values:
            results[name] = {'dates': dates, 'values': values}
            print(f"  Extracted {len(values)} points.")
            v_array = np.array(values)
            print(f"  Stats for {name}: Min={v_array.min():.2f}, Max={v_array.max():.2f}, Mean={v_array.mean():.2f}")
        else:
            print(f"  No data found for {name}.")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    for name, config in experiments.items():
        if name in results:
            data = results[name]
            plt.plot(data['dates'], data['values'], 
                     color=config['color'], linewidth=config['linewidth'], 
                     linestyle=config['style'], label=config['label'],
                     marker='o', markersize=4)
    
    plt.title('Global OCEAN Mean Surface Temperature (SST)\nGF01_ref vs GF_OPT vs MERRA2')
    plt.ylabel('Temperature (K)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_file = "SST_OceanOnly_Comparison.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    process_and_plot()
