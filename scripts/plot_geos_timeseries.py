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

def calc_global_mean(ds, var_name):
    """
    Calculate area-weighted global mean for a variable.
    Assumes lat/lon dimensions.
    """
    data = ds[var_name]
    
    # Check for correct latitude name
    if 'lat' in data.coords:
        lat = data.coords['lat']
    elif 'YDim' in data.coords: # Standard GEOS/GMAO names sometimes
        lat = data.coords['YDim']
    else:
        # Fallback or try to find by standard names if needed
        # Assuming 'lat' for now which fits previous script assumptions
        lat = ds.coords['lat']
        
    weights = np.cos(np.deg2rad(lat))
    weights.name = "weights"
    
    # Broadcast weights if necessary, but weighted mean handles this usually
    data_weighted = data.weighted(weights)
    
    # Mean over lat and lon
    return data_weighted.mean(dim=["lat", "lon"]).values.item()

def process_and_plot():
    root_dir = "/nobackupp27/afahad/GFEXPv11.7"
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} not found. Test mode?")
        return

    all_dirs = glob.glob(os.path.join(root_dir, "GF*"))
    experiments = [os.path.basename(d) for d in all_dirs if os.path.isdir(d)]
    experiments.sort()
    if 'GFO' in experiments:
        experiments.remove('GFO')
    
    print(f"Found experiments: {experiments}")
    
    # Store time series: {exp_name: {'dates': [], 'values': []}}
    results = {}
    
    start_year = 2005
    end_year = 2017 # As per previous script ranges roughly

    # Iterate experiments
    for exp in experiments:
        print(f"Processing {exp}...")
        exp_dir = os.path.join(root_dir, exp)
        
        # Potential dirs for 'geosgcm_surf'
        surf_dirs = [
             os.path.join(exp_dir, 'holding', 'geosgcm_surf'),
             os.path.join(exp_dir, 'geosgcm_surf') # Check root too just in case
        ]
        
        dates = []
        values = []
        
        # Iterate over all months in range
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Bounds check from previous request
                if year == 2008 and month < 5: continue
                if year == 2017 and month > 2: continue
                
                fpath = get_file_path(surf_dirs, year, month, 'geosgcm_surf')
                
                if fpath:
                    try:
                        with xr.open_dataset(fpath) as ds:
                            if 'TS' in ds:
                                val = calc_global_mean(ds, 'TS')
                                values.append(val)
                                dates.append(np.datetime64(f"{year}-{month:02d}-01"))
                            else:
                                pass # Missing variable
                    except Exception as e:
                        # print(f"  Error reading {fpath}: {e}")
                        pass
        
        if values:
            results[exp] = {'dates': dates, 'values': values}
            print(f"  Extracted {len(values)} points.")
        else:
            print(f"  No data found.")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot non-reference first (Grey)
    for exp, data in results.items():
        if exp == 'GF01_ref': continue
        if exp == 'GF_OPT': continue
        plt.plot(data['dates'], data['values'], color='grey', alpha=0.5, linewidth=1, label='_nolegend_')
        
    # Plot reference (Blue)
    if 'GF01_ref' in results:
        data = results['GF01_ref']
        plt.plot(data['dates'], data['values'], color='blue', linewidth=2, label='GF01_ref', zorder=10)

    # Plot optimized (Green)
    if 'GF_OPT' in results:
        data = results['GF_OPT']
        plt.plot(data['dates'], data['values'], color='green', linewidth=3, label='GF_OPT', zorder=10)
    
    # Formatting
    plt.title('Global Mean Surface Temperature (TS)')
    plt.ylabel('Temperature (K)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legend - Hack to show just Grey and Blue
    # Add a dummy grey line for legend
    plt.plot([], [], color='grey', label='Other Experiments')
    plt.legend()
    
    out_file = "GEOS_TS_GlobalMean_TimeSeries.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    process_and_plot()
