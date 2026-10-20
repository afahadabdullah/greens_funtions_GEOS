#!/usr/bin/env python
"""
Analyze warming pattern (TS difference) between GF_OPT and GF01_ref.
Focuses on the overlapping period: May 2005 - March 2008.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os
import pandas as pd

# Configuration
BASE_DIR = '/nobackupp27/afahad/GFEXPv11.7'
EXP_REF = 'GF01_ref'
EXP_OPT = 'GF_OPT'
START_DATE = '2005-05-01'
END_DATE = '2008-03-31'

def get_file_list(exp_name, collection='geosgcm_surf'):
    """Get list of files for the experiment within the date range."""
    # Possible locations
    paths = [
        os.path.join(BASE_DIR, exp_name, 'holding', collection),
        os.path.join(BASE_DIR, exp_name, collection)
    ]
    
    valid_path = None
    for p in paths:
        if os.path.exists(p):
            valid_path = p
            break
            
    if not valid_path:
        print(f"Error: Could not find data directory for {exp_name}")
        return []

    # Find files using recursive glob or smart walking? 
    # The structure seems to be flat or YYYYMM subdirs.
    # We will search recursively to be safe.
    files = []
    
    # Generate expected monthly dates
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    for date in dates:
        year = date.year
        month = date.month
        # Search patterns
        patterns = [
            f"*.{collection}.{year}{month:02d}*.nc4", # Standard
            f"*{collection}.{year}{month:02d}*.nc4",   # Loose
            f"{year}{month:02d}/*.{collection}.{year}{month:02d}*.nc4" # Subdir
        ]
        
        found = False
        for pat in patterns:
            full_pat = os.path.join(valid_path, pat)
            matches = glob.glob(full_pat)
            if matches:
                files.append(matches[0]) # Take first match
                found = True
                break
        
        if not found:
            print(f"Warning: Missing file for {exp_name} - {year}-{month:02d}")

    return sorted(files)

def load_data(files, var_name='TS'):
    """Load variable from files and compute time mean."""
    if not files:
        return None
        
    print(f"Loading {len(files)} files...")
    # Use nested combine to handle potentially mixed file structures if needed, 
    # but 'by_coords' is usually safest if time coord is clean.
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=False) 
    
    # Extract variable
    data = ds[var_name]
    
    # Calculate time mean
    mean_data = data.mean(dim='time').load() # Load into memory
    return mean_data

def global_weighted_mean(da):
    """Calculate area-weighted global mean."""
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    weighted = da.weighted(weights)
    return weighted.mean(dim=["lat", "lon"]).values.item()

def analyze_warming(ref_mean, opt_mean):
    """Calculate difference and stats."""
    # Difference: OPT - REF (Warming)
    diff = opt_mean - ref_mean
    
    # Stats
    glob_mean_ref = global_weighted_mean(ref_mean)
    glob_mean_opt = global_weighted_mean(opt_mean)
    glob_mean_diff = global_weighted_mean(diff)
    
    local_min = diff.min().values.item()
    local_max = diff.max().values.item()
    std_dev = diff.std().values.item()
    
    # Root Mean Square Difference
    rmsd = np.sqrt((diff**2).mean()).values.item()
    
    print("\n" + "="*40)
    print("WARMING STATISTICS (opt - ref)")
    print("="*40)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Global Mean REF (TS): {glob_mean_ref:.4f} K")
    print(f"Global Mean OPT (TS): {glob_mean_opt:.4f} K")
    print("-" * 20)
    print(f"Global Mean Warming:   {glob_mean_diff:.4f} K")
    print(f"Max Local Warming:     {local_max:.4f} K")
    print(f"Min Local Warming:     {local_min:.4f} K")
    print(f"Spatial Std Dev:       {std_dev:.4f} K")
    print(f"RMS Difference:        {rmsd:.4f} K")
    print("="*40 + "\n")
    
    return diff

def plot_map(diff_data):
    """Plot the warming pattern map."""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    
    # Determine color limits centered on 0
    abs_max = max(abs(diff_data.min()), abs(diff_data.max()))
    # Cap at reasonable value if outliers exist, e.g., 5K, or use robust=True
    vlim = min(abs_max, 5.0) 
    
    im = diff_data.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='RdBu_r', 
        vmin=-vlim, vmax=vlim,
        cbar_kwargs={'label': 'Temperature Difference (K)', 'shrink': 0.8}
    )
    
    plt.title(f'Warming Pattern: {EXP_OPT} - {EXP_REF}\nPeriod: {START_DATE} to {END_DATE}')
    
    out_file = "Warming_Pattern_Map.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Map saved to {out_file}")

def main():
    print(f"Analyzing {EXP_REF} vs {EXP_OPT}...")
    
    # Get files
    files_ref = get_file_list(EXP_REF)
    files_opt = get_file_list(EXP_OPT)
    
    if not files_ref or not files_opt:
        print("Aborting: Missing files.")
        return

    # Check file counts
    print(f"Found {len(files_ref)} files for Reference.")
    print(f"Found {len(files_opt)} files for Optimized.")
    
    # Load and Mean
    print("Processing Reference data...")
    ref_mean = load_data(files_ref)
    
    print("Processing Optimized data...")
    opt_mean = load_data(files_opt)

    # Analyze
    print("Calculating statistics...")
    diff = analyze_warming(ref_mean, opt_mean)
    
    # Plot
    print("Generating plot...")
    plot_map(diff)

if __name__ == "__main__":
    main()
