#!/usr/bin/env python
"""
Compare TS (Surface Temperature) between GF01_ref and GF22_CQF experiments.
Uses first year of data (May 2005 - April 2006) from geosgcm_surf collection.

Run with:
    tcsh
    conda activate earth
    python compare_ts_gf01_gf22.py
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
import os

# Base paths
BASE_DIR = '/home6/afahad/nobackup27/GFEXPv11.7'
GF01_SURF_DIR = os.path.join(BASE_DIR, 'GF01_ref/holding/geosgcm_surf')
GF22_SURF_DIR = os.path.join(BASE_DIR, 'GF22_CQF/geosgcm_surf')
OUTPUT_DIR = os.path.join(BASE_DIR, 'scripts')

# First year: May 2005 - April 2006 (12 months)
# GF01 file pattern: GF01_ref.geosgcm_surf.YYYYMMDD_0000z.nc4
# GF22 file pattern: GF22_CQF.geosgcm_surf.monthly.YYYYMM.nc4

def get_gf01_files():
    """Get first year files for GF01_ref experiment"""
    files = []
    # First year: May 2005 to April 2006
    year_months = [
        ('2005', '05'), ('2005', '06'), ('2005', '07'), ('2005', '08'),
        ('2005', '09'), ('2005', '10'), ('2005', '11'), ('2005', '12'),
        ('2006', '01'), ('2006', '02'), ('2006', '03'), ('2006', '04')
    ]
    for year, month in year_months:
        pattern = f"GF01_ref.geosgcm_surf.{year}{month}01_0000z.nc4"
        fpath = os.path.join(GF01_SURF_DIR, pattern)
        if os.path.exists(fpath):
            files.append(fpath)
    return sorted(files)

def get_gf22_files():
    """Get first year files for GF22_CQF experiment"""
    files = []
    # First year: May 2005 to April 2006
    year_months = [
        ('2005', '05'), ('2005', '06'), ('2005', '07'), ('2005', '08'),
        ('2005', '09'), ('2005', '10'), ('2005', '11'), ('2005', '12'),
        ('2006', '01'), ('2006', '02'), ('2006', '03'), ('2006', '04')
    ]
    for year, month in year_months:
        pattern = f"GF22_CQF.geosgcm_surf.monthly.{year}{month}.nc4"
        fpath = os.path.join(GF22_SURF_DIR, pattern)
        if os.path.exists(fpath):
            files.append(fpath)
    return sorted(files)

def load_ts_data(files, exp_name):
    """Load TS variable from files and compute annual mean"""
    print(f"\nLoading {exp_name} files:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    
    # Open all files and concatenate
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', 
                           decode_times=True, use_cftime=True)
    
    # Get TS variable
    if 'TS' in ds:
        ts = ds['TS']
    else:
        print(f"Available variables: {list(ds.data_vars)}")
        raise KeyError("TS variable not found in dataset")
    
    print(f"  Shape: {ts.shape}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    return ts

def plot_comparison(ts_gf01, ts_gf22):
    """Create comparison plots"""
    
    # Compute annual means
    ts_gf01_mean = ts_gf01.mean(dim='time').squeeze()
    ts_gf22_mean = ts_gf22.mean(dim='time').squeeze()
    
    # Compute difference
    ts_diff = ts_gf22_mean - ts_gf01_mean
    
    # Get coordinates (try different possible names)
    if 'lon' in ts_gf01_mean.coords:
        lon = ts_gf01_mean['lon'].values
        lat = ts_gf01_mean['lat'].values
    elif 'longitude' in ts_gf01_mean.coords:
        lon = ts_gf01_mean['longitude'].values
        lat = ts_gf01_mean['latitude'].values
    else:
        # Use coordinate dimensions directly
        lon = ts_gf01_mean.coords[ts_gf01_mean.dims[-1]].values
        lat = ts_gf01_mean.coords[ts_gf01_mean.dims[-2]].values
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), 
                              subplot_kw={'projection': ccrs.Robinson()})
    
    # Common settings
    cmap_abs = 'RdYlBu_r'
    cmap_diff = 'RdBu_r'
    
    # Get robust vmin/vmax for absolute values
    vmin = min(ts_gf01_mean.min().values, ts_gf22_mean.min().values)
    vmax = max(ts_gf01_mean.max().values, ts_gf22_mean.max().values)
    
    # Plot 1: GF01_ref TS
    ax1 = axes[0]
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    im1 = ax1.pcolormesh(lon, lat, ts_gf01_mean.values, 
                         transform=ccrs.PlateCarree(),
                         cmap=cmap_abs, vmin=vmin, vmax=vmax)
    ax1.set_title('GF01_ref - Surface Temperature (TS)\nFirst Year Mean (May 2005 - Apr 2006)', 
                  fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label('Temperature (K)')
    
    # Plot 2: GF22_CQF TS
    ax2 = axes[1]
    ax2.set_global()
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    im2 = ax2.pcolormesh(lon, lat, ts_gf22_mean.values, 
                         transform=ccrs.PlateCarree(),
                         cmap=cmap_abs, vmin=vmin, vmax=vmax)
    ax2.set_title('GF22_CQF - Surface Temperature (TS)\nFirst Year Mean (May 2005 - Apr 2006)', 
                  fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.set_label('Temperature (K)')
    
    # Plot 3: Difference (GF22 - GF01)
    ax3 = axes[2]
    ax3.set_global()
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
    
    # Symmetric colorbar for difference
    diff_max = max(abs(ts_diff.min().values), abs(ts_diff.max().values))
    diff_max = min(diff_max, 5)  # Cap at 5K for visibility
    
    im3 = ax3.pcolormesh(lon, lat, ts_diff.values, 
                         transform=ccrs.PlateCarree(),
                         cmap=cmap_diff, vmin=-diff_max, vmax=diff_max)
    ax3.set_title('Difference (GF22_CQF - GF01_ref)\nFirst Year Mean', 
                  fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar3.set_label('Temperature Difference (K)')
    
    plt.tight_layout()
    
    # Save figure
    outfile = os.path.join(OUTPUT_DIR, 'ts_comparison_gf01_gf22.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to: {outfile}")
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"\nGF01_ref TS:")
    print(f"  Global Mean: {float(ts_gf01_mean.mean()):.2f} K")
    print(f"  Min: {float(ts_gf01_mean.min()):.2f} K, Max: {float(ts_gf01_mean.max()):.2f} K")
    
    print(f"\nGF22_CQF TS:")
    print(f"  Global Mean: {float(ts_gf22_mean.mean()):.2f} K")
    print(f"  Min: {float(ts_gf22_mean.min()):.2f} K, Max: {float(ts_gf22_mean.max()):.2f} K")
    
    print(f"\nDifference (GF22 - GF01):")
    print(f"  Global Mean Diff: {float(ts_diff.mean()):.4f} K")
    print(f"  Min Diff: {float(ts_diff.min()):.4f} K")
    print(f"  Max Diff: {float(ts_diff.max()):.4f} K")
    print(f"  RMS Diff: {float(np.sqrt((ts_diff**2).mean())):.4f} K")
    
    return ts_diff

def main():
    print("="*60)
    print("TS Comparison: GF01_ref vs GF22_CQF")
    print("="*60)
    
    # Get file lists
    gf01_files = get_gf01_files()
    gf22_files = get_gf22_files()
    
    print(f"\nFound {len(gf01_files)} GF01 files")
    print(f"Found {len(gf22_files)} GF22 files")
    
    if len(gf01_files) == 0 or len(gf22_files) == 0:
        print("ERROR: Missing files!")
        return
    
    # Load TS data
    ts_gf01 = load_ts_data(gf01_files, 'GF01_ref')
    ts_gf22 = load_ts_data(gf22_files, 'GF22_CQF')
    
    # Create comparison plots
    ts_diff = plot_comparison(ts_gf01, ts_gf22)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
