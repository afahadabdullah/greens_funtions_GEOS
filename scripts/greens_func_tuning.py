import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import time
# ... (Imports)

# --- Configuration & Constants ---

# Variable Definitions based on User Request
GEOS_VARS = ['FRSEAICE', 'T2M', 'T_300', 'U_200', 'H_500', 'LWGNET', 'SWGNET'] 
MIT_VARS = ['SALT_0m', 'SALT_300m', 'THETA_0m', 'THETA_300m']

VAR_MAP_GEOS = {
    'FRSEAICE': 'FRSEAICE',
    'T2M': 'T2M',

    'T300': 'T300',
    'U200': 'U200',
    'H500': 'H500',
    'LWGNET': 'LWGNET',
    'SWGNET': 'SWGNET'
}

VAR_MAP_MIT = {
    'SALT0': 'SALT_0m',
    'SALT300': 'SALT_300m',
    'THETA0': 'THETA_0m',
    'THETA300': 'THETA_300m'
}

TARGET_DIR_DEFAULT = "/nobackupp27/afahad/GFEXPv11.7/target"
OBS_FILE_MERRA2_DJF = "MERRA2.seasonal.mean.DJF.nc" 
OBS_FILE_MERRA2_JJA = "MERRA2.seasonal.mean.JJA.nc"
OBS_FILE_ECCO = "ECCO_L4_Climatology_2008-2017.nc"

SEASONS = ['DJF', 'JJA']

# --- Parameter Configuration (Hardcoded) ---

def get_experiment_parameters():
    """
    Returns the hardcoded experiment parameters as a DataFrame.
    Derived from walkthrough_bundled_equal_weights.html
    """
    data = [
        {'Exp': 'GF02_TURN', 'Parameter': 'TURNRHCRIT', 'Original': 726.0, 'Perturbed': 850.0},
        {'Exp': 'GF03_ENTR', 'Parameter': 'ENTR_DP', 'Original': 1.0e-4, 'Perturbed': 4.0e-4},
        {'Exp': 'GF04_WBF', 'Parameter': 'WBFFACTOR', 'Original': 0.091, 'Perturbed': 0.3},
        {'Exp': 'GF05_DCS', 'Parameter': 'DCS', 'Original': 2.9e-4, 'Perturbed': 2.5e-4},
        {'Exp': 'GF06_DUST', 'Parameter': 'FDROP_DUST', 'Original': 0.5, 'Perturbed': 0.1},
        {'Exp': 'GF07_DCNV', 'Parameter': 'DROPSZCNV', 'Original': 2.5e-5, 'Perturbed': 5.0e-5},
        {'Exp': 'GF08_FSOOT', 'Parameter': 'FDROP_SOOT', 'Original': 0.05, 'Perturbed': 0.09},
        {'Exp': 'GF09_MINCN', 'Parameter': 'MINCDNC', 'Original': 52.0, 'Perturbed': 25.0},
        {'Exp': 'GF10_TBRAD', 'Parameter': 'TURB_BETA_RAD', 'Original': 0.2, 'Perturbed': 0.3},
        {'Exp': 'GF11_TRBKH', 'Parameter': 'TURB_KHRADFAC', 'Original': 0.85, 'Perturbed': 0.5},
        {'Exp': 'GF12_MINRH', 'Parameter': 'MINRHCRIT', 'Original': 0.89, 'Perturbed': 0.8},
        {'Exp': 'GF13_QRC', 'Parameter': 'QRC_CRIT', 'Original': 2.0e-4, 'Perturbed': 4.0e-4},
        {'Exp': 'GF14_RKM', 'Parameter': 'RKM', 'Original': 8.0, 'Perturbed': 12.0},
        {'Exp': 'GF15_AUT', 'Parameter': 'AUT_SCALE', 'Original': 1.7, 'Perturbed': 0.5},
        {'Exp': 'GF16_UIS', 'Parameter': 'UISCALE', 'Original': 1.0, 'Perturbed': 0.75},
        {'Exp': 'GF17_CAHMAX', 'Parameter': 'CICE_AH_MAX', 'Original': 0.3, 'Perturbed': 0.2},
        {'Exp': 'GF18_ALBICEV', 'Parameter': 'ALBICEV', 'Original': 0.78, 'Perturbed': 0.73},
        {'Exp': 'GF19_BMDL', 'Parameter': 'VSLfactor', 'Original': 1.0, 'Perturbed': 0.8},
        {'Exp': 'GF20_CH1', 'Parameter': 'Charnok1', 'Original': .2030325e-5, 'Perturbed': .2842455e-5},
        {'Exp': 'GF21_CH2', 'Parameter': 'Charnok2', 'Original': 0.215e-2, 'Perturbed': 0.315e-02},
        {'Exp': 'GF22_CQF', 'Parameter': 'CQfactor', 'Original': 1.0, 'Perturbed': 0.5},
    ]
    
    df = pd.DataFrame(data)
    df['Delta'] = df['Perturbed'] - df['Original']
    return df

# --- Data Loading ---

def load_geos_data(exp_dir, exp_name, season):
    """
    Loads GEOS climatology for a specific experiment and season.
    Path: {exp_dir}/{exp_name}/clim/geos_clim/{exp_name}.mean.{season}.nc
    """
    fpath = os.path.join(exp_dir, exp_name, 'clim', 'geos_clim', f"{exp_name}.mean.{season}.nc")
    if not os.path.exists(fpath):
        print(f"  [WARN] GEOS file missing: {fpath}")
        return None
    
    try:
        ds = xr.open_dataset(fpath)
        return ds
    except Exception as e:
        print(f"  [ERROR] Failed to load GEOS file {fpath}: {e}")
        return None

def load_mitgcm_data(exp_dir, exp_name, season):
    """
    Loads MITgcm climatology.
    Path: {exp_dir}/{exp_name}/clim/mitgcm.{VAR}.{DEPTH}.{season}.2008-2017.nc
    Returns a merged dataset of all MIT variables for the season.
    """
    base_clim_dir = os.path.join(exp_dir, exp_name, 'clim')
    if not os.path.exists(base_clim_dir):
        return None
    
    # We need to construct the dataset from multiple files: mitgcm.SALT.0m.DJF..., mitgcm.THETA.300m.DJF...
    merged_vars = {}
    
    # Map our internal keys to filenames patterns
    # Internal: SALT_0m -> File: mitgcm.SALT.0m.SEASON*.nc
    
    # Variables to load matching VAR_MAP_MIT
    # keys: SALT0, SALT300, THETA0, THETA300
    
    file_map = {
        'SALT0': ('SALT', '0m'),
        'SALT300': ('SALT', '300m'),
        'THETA0': ('THETA', '0m'),
        'THETA300': ('THETA', '300m')
    }
    
    datasets = []
    
    for var_key, (vname, depth) in file_map.items():
        pattern = os.path.join(base_clim_dir, f"mitgcm.{vname}.{depth}.{season}*.nc")
        files = glob.glob(pattern)
        if not files:
            print(f"  [WARN] MITgcm file missing for {vname} {depth} {season} in {exp_name}")
            continue
            
        try:
            ds = xr.open_dataset(files[0])
            
            # Rename variable to standard internal name
            # e.g. 'THETA' -> 'THETA_0m'
            target_name = VAR_MAP_MIT[var_key] # e.g. 'THETA_0m'
            
            if vname in ds:
                ds = ds.rename({vname: target_name})
            
            datasets.append(ds)
        except Exception as e:
            print(f"  [ERROR] Failed to load MIT file {files[0]}: {e}")
            
    if not datasets:
        return None
        
    # Merge
    try:
        ds_merged = xr.merge(datasets)
        return ds_merged
    except Exception as e:
        print(f"  [ERROR] Failed to merge MIT datasets for {exp_name}: {e}")
        return None

def load_targets(target_dir):
    """
    Loads target observations.
    Returns a dictionary structure: {'GEOS': {Season: ds}, 'MIT': {Season: ds}}
    """
    obs_data = {'GEOS': {}, 'MIT': {}}
    
    # Load GEOS Targets (MERRA2)
    for season in SEASONS:
        fname = f"MERRA2.seasonal.mean.{season}.nc"
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath):
            ds_ref = xr.open_dataset(fpath)
            
            # Align Variable Names (MERRA2 vs Internal)
            # MERRA2: LWGNT, SWGNT -> Internal: LWGNET, SWGNET
            rename_map = {'LWGNT': 'LWGNET', 'SWGNT': 'SWGNET'}
            for old, new in rename_map.items():
                if old in ds_ref:
                    ds_ref = ds_ref.rename({old: new})
            
            # User Correction: Flip sign of LWGNET in Target
            if 'LWGNET' in ds_ref:
                ds_ref['LWGNET'] = -1.0 * ds_ref['LWGNET']
                
            obs_data['GEOS'][season] = ds_ref
        else:
            print(f"  [WARN] Missing MERRA2 target: {fpath}")

    # Load MIT Targets (ECCO)
    # User said: ECCO_L4_Climatology_2008-2017.nc
    # Does this file contain all seasons? Or separate vars?
    # process_ecco script saves separate files similarly to MITgcm outputs: ECCO_L4.SALT.0m.DJF...
    # BUT user said "target observed data are in ... ECCO_L4_Climatology_2008-2017.nc"
    # This implies a unified file was created possibly.
    # We will try to load the unified file first.
    
    fpath_ecco = os.path.join(target_dir, OBS_FILE_ECCO)
    
    if os.path.exists(fpath_ecco):
        ds_ecco = xr.open_dataset(fpath_ecco)
        
        # Process ECCO variables to match internal keys (Slices)
        # We need THETA_0m, THETA_300m, SALT_0m, SALT_300m
        # Assuming ECCO has 'THETA' and 'SALT' with 'Z' or 'depth' dimension
        
        processed_vars = {}
        
        # Define depth map for ECCO
        # We need to pull from VAR_MAP_MIT reversely or explicitly
        # VAR_MAP_MIT = {'THETA0': 'THETA_0m', ...}
        # Let's simple hardcode the extraction based on known needs
        
        ecco_map = [
            ('THETA', 0, 'THETA_0m'),
            ('THETA', 300, 'THETA_300m'),
            ('SALT', 0, 'SALT_0m'),
            ('SALT', 300, 'SALT_300m')
        ]
        
        for vname, depth_val, target_name in ecco_map:
            if vname in ds_ecco:
                # Find depth dim
                z_dim = None
                for d in ['Z', 'depth', 'k']:
                     if d in ds_ecco[vname].dims:
                         z_dim = d
                         break
                
                if z_dim:
                    try:
                        # Select nearest
                        da = ds_ecco[vname].sel({z_dim: depth_val}, method='nearest')
                        da.name = target_name
                        # Drop depth dim to make it 2D
                        if z_dim in da.coords:
                            da = da.drop_vars(z_dim)
                        processed_vars[target_name] = da
                    except:
                        pass
                else:
                    # If 2D already?
                    pass
        
        if processed_vars:
            ds_ecco_proc = xr.merge(processed_vars.values())
            
            # Check for Time/Season dimension
            # Likely 'time' from concat
            t_dim = None
            for d in ['time', 'season', 't']:
                if d in ds_ecco_proc.dims:
                    t_dim = d
                    break
            
            if t_dim and ds_ecco_proc.sizes[t_dim] == 2:
                print(f"  [INFO] Found 2 time steps in ECCO data. Assuming idx 0=DJF, 1=JJA.")
                # Sorted by time: DJF (Jan) < JJA (July)
                ds_djf = ds_ecco_proc.isel({t_dim: 0})
                ds_jja = ds_ecco_proc.isel({t_dim: 1})
                
                # Drop time dim
                if t_dim in ds_djf.coords: ds_djf = ds_djf.drop_vars(t_dim)
                if t_dim in ds_jja.coords: ds_jja = ds_jja.drop_vars(t_dim)
                
                obs_data['MIT']['DJF'] = ds_djf
                obs_data['MIT']['JJA'] = ds_jja
                
            elif t_dim and ds_ecco_proc.sizes[t_dim] == 1:
                # Squeeze
                ds_ecco_proc = ds_ecco_proc.squeeze(dim=t_dim)
                if t_dim in ds_ecco_proc.coords: ds_ecco_proc = ds_ecco_proc.drop_vars(t_dim)
                obs_data['MIT']['DJF'] = ds_ecco_proc
                obs_data['MIT']['JJA'] = ds_ecco_proc
            else:
                 # No time dim or unknown size.
                 # Warn if size > 1
                 if t_dim and ds_ecco_proc.sizes[t_dim] > 1:
                     print(f"  [WARN] ECCO data has {ds_ecco_proc.sizes[t_dim]} time steps. Using Isel 0 for DJF and 1 for JJA if avail, else broadcasting.")
                     # Fallback attempts
                     obs_data['MIT']['DJF'] = ds_ecco_proc.isel({t_dim: 0})
                     if ds_ecco_proc.sizes[t_dim] > 1:
                         obs_data['MIT']['JJA'] = ds_ecco_proc.isel({t_dim: 1})
                     else:
                         obs_data['MIT']['JJA'] = ds_ecco_proc.isel({t_dim: 0})
                 else:
                     # Static
                     obs_data['MIT']['DJF'] = ds_ecco_proc
                     obs_data['MIT']['JJA'] = ds_ecco_proc

        else:
             # Fallback if processing failed or vars missing
             obs_data['MIT']['DJF'] = ds_ecco
             obs_data['MIT']['JJA'] = ds_ecco

    else:
         # Fallback search for pattern ECCO_L4.*.{Season}.*.nc
         for season in SEASONS:
             pass # ... (Existing logic can stay or be removed if unused)
             # Removing fallback logic for brevity if implied not needed, 
             # but let's keep it simply by just closing the block if we essentially replaced the main path.
             print(f"  [WARN] Missing ECCO target: {fpath_ecco}")
             
    return obs_data

# --- Mock Data Generation ---

def generate_mock_data(base_test_dir):
    """
    Generates a full directory structure with synthetic NetCDF files 
    to test the pipeline locally.
    """
    print("Generating MOCK DATA...")
    if not os.path.exists(base_test_dir):
        os.makedirs(base_test_dir)
        
    # 1. Mock HTML (No longer needed)
    # html_content = ... (removed)
    # with open(...) ... (removed)
        
    # 2. Mock Targets
    target_dir = os.path.join(base_test_dir, "target")
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    
    def make_ds(var_names):
        lat = np.linspace(-90, 90, 46)
        lon = np.linspace(-180, 180, 72)
        ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
        for v in var_names:
            ds[v] = (('lat', 'lon'), np.random.rand(46, 72))
        return ds
        
    # MERRA2
    for s in SEASONS:
        # Create using Internal Names first, then rename for Mock File
        ds_merra = make_ds(VAR_MAP_GEOS.values())
        rename_map_rev = {'LWGNET': 'LWGNT', 'SWGNET': 'SWGNT'}
        for internal, external in rename_map_rev.items():
            if internal in ds_merra:
                ds_merra = ds_merra.rename({internal: external})
        
        ds_merra.to_netcdf(os.path.join(target_dir, f"MERRA2.seasonal.mean.{s}.nc"))
        
    # ECCO (3D Variables with Time)
    lat = np.linspace(-90, 90, 46)
    lon = np.linspace(-180, 180, 72)
    ds_ecco = xr.Dataset(coords={'lat': lat, 'lon': lon, 'depth': [0, 300], 'time': [0, 1]})
    ds_ecco['THETA'] = (('time', 'depth', 'lat', 'lon'), np.random.rand(2, 2, 46, 72))
    ds_ecco['SALT'] = (('time', 'depth', 'lat', 'lon'), np.random.rand(2, 2, 46, 72))
    ds_ecco.to_netcdf(os.path.join(target_dir, OBS_FILE_ECCO))
    
    # 3. Mock Experiments (GF01_ref, GF03_ENTR, GF14_RKM)
    exps = ['GF01_ref', 'GF03_ENTR', 'GF14_RKM']
    
    # file_map from load_mit_data
    mit_file_map = {
        'SALT0': ('SALT', '0m'),
        'SALT300': ('SALT', '300m'),
        'THETA0': ('THETA', '0m'),
        'THETA300': ('THETA', '300m')
    }
    
    for exp in exps:
        # GEOS Structure (Original Grid 46x72)
        geos_dir = os.path.join(base_test_dir, exp, 'clim', 'geos_clim')
        os.makedirs(geos_dir, exist_ok=True)
        for s in SEASONS:
             # Add signal to GF03_ENTR to test finding it
             ds = make_ds(VAR_MAP_GEOS.values())
             if exp == 'GF03_ENTR':
                 ds = ds + 0.1 # Perturbation
             ds.to_netcdf(os.path.join(geos_dir, f"{exp}.mean.{s}.nc"))
             
        # MIT Structure
        mit_dir = os.path.join(base_test_dir, exp, 'clim')
        os.makedirs(mit_dir, exist_ok=True)
        
        for s in SEASONS:
            for key, (vname, depth) in mit_file_map.items():
                fname = f"mitgcm.{vname}.{depth}.{s}.nc"
                # Content variable name must be vname (e.g. THETA) to match loader expectation
                ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
                ds[vname] = (('lat', 'lon'), np.random.rand(46, 72))
                ds.to_netcdf(os.path.join(mit_dir, fname))
    return target_dir

# --- Optimization Logic ---

def compute_cost(model_ds, obs_ds, vars_map):
    """
    Computes cost component for a single dataset (season/model).
    Cost = sum( ((Model - Obs)/Sigma)^2 )
    
    For simplicty in this version:
    1. Regrid Obs to Model (or vice versa).
    2. Compute Sigma (Std Dev of Obs).
    3. Compute MSE normalized by Variance.
    """
    total_sq_err = 0.0
    total_points = 0
    
    # Ensure grids match
    # Assuming already regridded or close enough for nearest neighbor in xarray
    
    for k, v_name in vars_map.items():
        if v_name not in model_ds or v_name not in obs_ds:
            continue
            
        mod = model_ds[v_name]
        obs = obs_ds[v_name]
        
        # Simple align/regrid
        try:
            obs_aligned = obs.interp_like(mod, method='nearest')
        except:
            continue
            
        # Variance of observations (Spatial)
        # Use a minimum threshold to avoid div by zero
        sigma2 = obs_aligned.var().values
        if sigma2 < 1e-12: sigma2 = 1.0 # Fallback
        
        diff = mod - obs_aligned
        sq_err = (diff ** 2).mean().values # Spatial mean of squared error
        
        # Normalize
        norm_cost = sq_err / sigma2
        
        total_sq_err += norm_cost
        total_points += 1
        
    if total_points == 0:
        return 0.0
        
    return total_sq_err # Sum of normalized errors for all variables

# --- Diagnostics ---
    
def generate_diagnostics(ref_data, pert_data_list, obs_data, weights_map, params_df):
    """
    Generates diagnostic plots:
    1. Mean Fields: Ref, Obs, Projected.
    2. Cost Maps: Ref Cost, Projected Cost, Each Exp Cost.
    For: TS (DJF), THETA 0m (DJF).
    """
    print("\nGenerating Diagnostics...")
    plot_dir = "plots"
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
    
    # --- GLOBAL COST BREAKDOWN PLOTS ---
    print("Generating Global Cost Breakdown Plots...")
    
    # We need to re-scan to aggregate costs globally
    # 1. By Experiment (Ref vs Exp1 vs Exp2...)
    # 2. By Variable (Ref vs Opt per var)
    
    exp_global_costs = {exp['ID']: 0.0 for exp in pert_data_list if exp['ID'] in weights_map}
    var_costs = {} # {var_name: {'Ref': 0, 'Opt': 0}}
    
    total_ref_cost = 0.0
    total_opt_cost = 0.0
    
    # Unique set of vars/seasons
    for season in SEASONS:
        all_vars = list(VAR_MAP_GEOS.values()) + list(VAR_MAP_MIT.values())
        
        for var_name in all_vars:
            coll_type = 'GEOS' if var_name in VAR_MAP_GEOS.values() else 'MIT'
            
            # Get Ref/Obs
            if coll_type == 'GEOS':
                if season not in ref_data['GEOS']: continue
                if var_name not in ref_data['GEOS'][season]: continue
                d_ref = ref_data['GEOS'][season][var_name]
                d_obs = obs_data['GEOS'][season][var_name]
            else:
                if season not in ref_data['MIT']: continue
                if var_name not in ref_data['MIT'][season]: continue
                d_ref = ref_data['MIT'][season][var_name]
                d_obs = obs_data['MIT'][season][var_name]
                
            # Align
            d_obs_aligned = d_obs.interp_like(d_ref, method='nearest')
            
            # Sigma
            sigma = d_obs_aligned.std().values
            if sigma < 1e-12: sigma = 1.0
            
            # Ref Cost
            c_ref = ((d_ref - d_obs_aligned)**2).sum().item() / (sigma**2)
            total_ref_cost += c_ref
            
            # Initialize Var Costs
            if var_name not in var_costs: var_costs[var_name] = {'Ref': 0.0, 'Opt': 0.0}
            var_costs[var_name]['Ref'] += c_ref
            
            # 1. Experiment Costs
            for p_data in pert_data_list:
                exp_id = p_data['ID']
                if exp_id not in weights_map: continue
                
                d_pert = None
                if coll_type == 'GEOS' and season in p_data['GEOS']:
                     d_pert = p_data['GEOS'][season].get(var_name)
                elif coll_type == 'MIT' and season in p_data['MIT']:
                     d_pert = p_data['MIT'][season].get(var_name)
                     
                if d_pert is not None:
                     if d_pert.shape != d_ref.shape:
                         d_pert = d_pert.interp_like(d_ref, method='nearest')
                         
                     c_exp = ((d_pert - d_obs_aligned)**2).sum().item() / (sigma**2)
                     exp_global_costs[exp_id] += c_exp
                else:
                    # If missing, assume same as Ref (no impact)
                    exp_global_costs[exp_id] += c_ref
            
            # 2. Optimized Cost (Projected)
            # Re-calculate projected field locally or assuming linear combo?
            # Proj = Ref + Sum(w * (Pert - Ref))
            # Cost = Sum((Proj - Obs)^2) / sigma^2
            
            # Construct Delta Field
            d_delta_sum = xr.zeros_like(d_ref)
            for p_data in pert_data_list:
                exp_id = p_data['ID']
                if exp_id not in weights_map: continue
                w = weights_map[exp_id]
                
                d_pert = None
                if coll_type == 'GEOS' and season in p_data['GEOS']:
                     d_pert = p_data['GEOS'][season].get(var_name)
                elif coll_type == 'MIT' and season in p_data['MIT']:
                     d_pert = p_data['MIT'][season].get(var_name)
                     
                if d_pert is not None:
                     if d_pert.shape != d_ref.shape:
                         d_pert = d_pert.interp_like(d_ref, method='nearest')
                     d_delta_sum += w * (d_pert - d_ref)
            
            d_proj = d_ref + d_delta_sum
            c_opt = ((d_proj - d_obs_aligned)**2).sum().item() / (sigma**2)
            
            total_opt_cost += c_opt
            var_costs[var_name]['Opt'] += c_opt

    # --- Plot 1: Global Cost Breakdown by Experiment ---
    # Bars: Ref, Opt, Exp1, Exp2... 
    # Normalized to Ref = 1.0
    
    if total_ref_cost > 0:
        labels = ['Ref', 'Opt'] + sorted(exp_global_costs.keys())
        values = [1.0, total_opt_cost/total_ref_cost]
        colors = ['gray', 'green']
        
        for k in sorted(exp_global_costs.keys()):
            val = exp_global_costs[k] / total_ref_cost
            values.append(val)
            colors.append('blue')
            
        plt.figure(figsize=(14, 6))
        x_pos = np.arange(len(labels))
        bars = plt.bar(x_pos, values, color=colors)
        
        plt.title("Cost Comparison: Reference, Optimized, and Individual Experiments", fontsize=14)
        plt.ylabel("Relative Cost (Ref=1)")
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", 
                     ha='center', va='bottom', fontsize=8, rotation=90)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Global_Cost_Breakdown_Experiments.png"))
        plt.close()
        
    # --- Plot 2: Global Cost Breakdown by Variable ---
    # Grouped Bar: Ref (Gray), Opt (Green) for each var
    # Norm to Ref=1 for each var
    
    plot_vars = sorted(var_costs.keys())
    ref_vals = [] # Always 1.0
    opt_vals = []
    opt_labels = []
    
    valid_plot_vars = []
    
    for v in plot_vars:
        r = var_costs[v]['Ref']
        o = var_costs[v]['Opt']
        if r > 0:
            valid_plot_vars.append(v)
            ref_vals.append(1.0)
            ratio = o / r
            opt_vals.append(ratio)
            change = (ratio - 1.0) * 100
            opt_labels.append(f"{ratio:.2f}\n({change:+.1f}%)")
            
    if valid_plot_vars:
        x = np.arange(len(valid_plot_vars))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        rects1 = plt.bar(x - width/2, ref_vals, width, label='Reference (1.0)', color='gray')
        rects2 = plt.bar(x + width/2, opt_vals, width, label='Optimized', color='green')
        
        plt.ylabel('Relative Cost (Ref=1)')
        plt.title('Global Cost Breakdown by Variable')
        plt.xticks(x, valid_plot_vars)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Labels
        for rect, label in zip(rects2, opt_labels):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2, height, label,
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Global_Cost_Breakdown_Variables.png"))
        plt.close()

    # Define targets to plot
    # (Collection Type, Var Name, Season, Label)
    targets = []
    
    for season in SEASONS:
        # GEOS Vars
        for v in VAR_MAP_GEOS.values():
            targets.append(('GEOS', v, season, f"{v}"))
            
        # MIT Vars
        for v in VAR_MAP_MIT.values():
            targets.append(('MIT', v, season, f"{v}"))
    
    # weights_map is {ExpID: Weight}
    
    for coll_type, var_name, season, label in targets:
        print(f"  Plotting {label} ({season})...")
        
        # 1. Get Data Arrays
        if coll_type == 'GEOS':
            if season not in ref_data['GEOS']: continue
            d_ref = ref_data['GEOS'][season][var_name] # xarray DataArray
            d_obs = obs_data['GEOS'][season][var_name]
        else:
            if season not in ref_data['MIT']: continue
            d_ref = ref_data['MIT'][season][var_name]
            d_obs = obs_data['MIT'][season][var_name]
            
        # Align Obs to Ref (Nearest) for Map Plotting
        d_obs_aligned = d_obs.interp_like(d_ref, method='nearest')
        
        # 2. Compute Projected Field (Linear Approximation)
        # F_proj = F_ref + Sum( w_i * (F_pert_i - F_ref) )
        d_proj = d_ref.copy(deep=True)
        
        # We need the perturbation fields for this specific var/season
        pert_deltas = xr.zeros_like(d_ref)
        
        for i, p_data in enumerate(pert_data_list):
            exp_id = p_data['ID']
            if exp_id not in weights_map:
                continue
                
            weight = weights_map[exp_id]
            
            d_pert = None
            if coll_type == 'GEOS':
                if season in p_data['GEOS']:
                     d_pert = p_data['GEOS'][season][var_name]
            else:
                if season in p_data['MIT']:
                     d_pert = p_data['MIT'][season][var_name]
            
            if d_pert is None:
                continue
                
            # Align if needed (should be same grid usually)
            if d_pert.shape != d_ref.shape:
                d_pert = d_pert.interp_like(d_ref, method='nearest')
                
            # Delta * Weight
            pert_deltas = pert_deltas + weight * (d_pert - d_ref)
            
        d_proj = d_ref + pert_deltas
        
        # 3. Compute Costs (Regional)
        # J = (Model - Obs)^2 / Sigma_Obs^2
        sigma = d_obs_aligned.std()
        if sigma < 1e-12: sigma = 1.0
        
        cost_ref = ((d_ref - d_obs_aligned)**2) / (sigma**2)
        cost_proj = ((d_proj - d_obs_aligned)**2) / (sigma**2)
        
        # --- PLOT 1: Mean Field Comparison ---
        # Layout: 2x3
        # Row 1: Reference, Observation, Projected
        # Row 2: (Ref - Obs), (Proj - Obs), [Empty]
        
        diff_ref = d_ref - d_obs_aligned
        diff_proj = d_proj - d_obs_aligned
        
        # Determine Color Limits for Fields
        vmin_f = min(d_ref.min().item(), d_obs_aligned.min().item(), d_proj.min().item())
        vmax_f = max(d_ref.max().item(), d_obs_aligned.max().item(), d_proj.max().item())
        
        # Determine Color Limits for Differences (Centered)
        max_diff = max(abs(diff_ref.min().item()), abs(diff_ref.max().item()),
                       abs(diff_proj.min().item()), abs(diff_proj.max().item()))
        vmin_d, vmax_d = -max_diff, max_diff
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()
        
        # 1. Reference
        im1 = axs[0].imshow(d_ref, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[0].set_title(f"Reference Mean\n(Cost: {((d_ref-d_obs_aligned)**2).mean().item():.3e})")
        plt.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 2. Observation
        im2 = axs[1].imshow(d_obs_aligned, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[1].set_title("Target Observation")
        plt.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 3. Projected
        im3 = axs[2].imshow(d_proj, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[2].set_title(f"Projected Mean\n(Cost: {((d_proj-d_obs_aligned)**2).mean().item():.3e})")
        plt.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 4. Difference (Ref - Obs)
        im4 = axs[3].imshow(diff_ref, origin='lower', cmap='RdBu_r', vmin=vmin_d, vmax=vmax_d)
        axs[3].set_title("Difference: Reference - Target")
        plt.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 5. Difference (Proj - Obs)
        im5 = axs[4].imshow(diff_proj, origin='lower', cmap='RdBu_r', vmin=vmin_d, vmax=vmax_d)
        axs[4].set_title("Difference: Projected - Target")
        plt.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 6. Empty
        axs[5].axis('off')
        
        fig.suptitle(f"{label} ({season}) - Mean Fields & Differences", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{var_name}_{season}_MeanFields.png"))
        plt.close()
        
        # --- PLOT 2: Cost Comparison ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # For Cost, use common log scale or just 0 to some max?
        # Difference might be subtle.
        c_max = max(cost_ref.max(), cost_proj.max())
        
        im0 = axs[0].imshow(cost_ref, origin='lower', cmap='Reds', vmin=0, vmax=c_max)
        axs[0].set_title(f"Ref Cost (Mean={cost_ref.mean().item():.2f})")
        plt.colorbar(im0, ax=axs[0])
        
        im1 = axs[1].imshow(cost_proj, origin='lower', cmap='Reds', vmin=0, vmax=c_max)
        axs[1].set_title(f"Proj Cost (Mean={cost_proj.mean().item():.2f})")
        plt.colorbar(im1, ax=axs[1])
        
        # Difference (Improvement)
        # Red = Worsened, Blue = Improved
        imp = cost_ref - cost_proj
        bound = max(abs(imp.min()), abs(imp.max()))
        im2 = axs[2].imshow(imp, origin='lower', cmap='RdBu_r', vmin=-bound, vmax=bound)
        axs[2].set_title(f"Improvement (Ref - Proj)\nPositive = Better")
        plt.colorbar(im2, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{var_name}_{season}_CostComparison.png"))
        plt.close()

        # --- PLOT 3: Regional Cost for Each Experiment (3x7 Grid) ---
        # Requested: 21 experiments, 3x7 layout, one colorbar
        
        # Filter valid experiments with data
        valid_exps_pdata = []
        for p_data in pert_data_list:
            # Check if this exp has data for this season/var
            val_exists = False
            if coll_type == 'GEOS' and season in p_data['GEOS']: val_exists = True
            if coll_type == 'MIT' and season in p_data['MIT']: val_exists = True
            
            if val_exists and p_data['ID'] in weights_map:
                valid_exps_pdata.append(p_data)
        
        # We assume there are 21 experiments. If fewer, we leave empty.
        # Fixed 3x7 grid
        fig, axs = plt.subplots(3, 7, figsize=(24, 10))
        axs = axs.flatten()
        
        # Determine common scale (using Ref or Global Max for context?)
        # User asked for "Regional Cost from Reference and each exp". 
        # Usually easier to compare if fixed scale.
        # Or maybe (Exp - Ref)? "Regional cost from reference and each exp"
        # implies seeing the cost Field itself.
        
        files_plotted = 0
        
        # Pre-calculate global max for colorbar
        # Or use max of Ref? Often Exp deviations are small. 
        # Let's use robust max from Ref and a few Exps to avoid outliers?
        # Safe bet: vmin=0, vmax=max(Ref Cost).
        vmax_g = cost_ref.max().item() # Default to Ref Max
        
        # Collect cost arrays to verify global max? 
        # It's expensive to load all first. Let's use Ref max or 1.5*Ref Max.
        
        for i, ax in enumerate(axs):
            if i >= len(valid_exps_pdata):
                ax.axis('off')
                continue
                
            p_data = valid_exps_pdata[i]
            exp_id = p_data['ID']
            
            d_pert = None
            if coll_type == 'GEOS':
                 d_pert = p_data['GEOS'][season][var_name]
            else:
                 d_pert = p_data['MIT'][season][var_name]
            
            # Align
            if d_pert.shape != d_ref.shape:
                d_pert = d_pert.interp_like(d_ref, method='nearest')
            
            cost_exp = ((d_pert - d_obs_aligned)**2) / (sigma**2)
            
            im = ax.imshow(cost_exp, origin='lower', cmap='Reds', vmin=0, vmax=vmax_g)
            ax.set_title(f"{exp_id}\nMean: {cost_exp.mean().item():.3f}", fontsize=9)
            ax.axis('off')
            files_plotted += 1
            
        # Common Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label='Normalized Cost')
        
        fig.suptitle(f"Individual Experiment Costs: {label} ({season}) - Ref Mean: {cost_ref.mean().item():.3f}", fontsize=16)
        # plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for cbar
        
        plt.savefig(os.path.join(plot_dir, f"{var_name}_{season}_AllExps_Cost.png"), bbox_inches='tight')
        plt.close()
            
    print("Diagnostics Generation Complete.\n")
    

# --- Optimization Engine ---

def optimize_greens_function(ref_data, pert_data_list, obs_data, param_info, weights=None):
    start_time = time.time()
    """
    Solves for optimal parameter increments.
    dp = argmin || (R_ref + S * dp) ||^2
    """
    print("\n--- Starting Green's Function Optimization ---")
    
    # 1. Build Residual Vector (Reference)
    # Actually, we usually formulate as: Cost J = ...
    # But for least squares, we want to minimize magnitude of vector R_pred.
    
    # Let's simplify:
    # We define a "Global Cost" for each experiment.
    # But GF requires the Sensitivity Matrix S of the *fields* or at least aggregated metrics.
    # Full Field GF: Huge matrix.
    # Scalar Cost Minimization (Quadratic Approx):
    # J(p) ~ J(p0) + g^T dp + 0.5 dp^T H dp
    # Where g and H are estimated from perturbations.
    
    # Alternative (classic):
    # Assume linear response of the Cost Function itself? No, cost is quadratic.
    # Assume linear response of the STATE (Model Fields).
    # X_pred(p) = X_ref + sum( (X_i - X_ref)/(dp_i) * dp_new_i )
    # This is better.
    
    # We need to construct the predicted state X_pred as a function of dp vectors.
    # Then iterate to minimize Cost(X_pred) vs Obs.
    # Since Cost is sum of squares, and X_pred is linear in dp, this is a linear least squares problem.
    
    # Formulate A * x = b
    # x = [dp_1, dp_2, ...] (normalized by their perturbation steps?)
    # It's better to solve for weights 'w' where p_new = p_ref + w * (p_pert - p_ref)
    # Then w=0 -> Ref, w=1 -> Perturbed.
    
    # A matrix columns: (X_i - X_ref) for each experiment i
    # b vector: (Obs - X_ref)
    
    # We need to flatten ALL variables, ALL lat/lon, ALL seasons into one giant vector?
    # That might be too big for memory (144*91 * 2 seasons * 10 vars ~ 260k points).
    # That fits in memory easily for 22 experiments.
    
    # Steps:
    # 1. Flatten Ref Data -> Vector X_ref
    # 2. Flatten Obs Data -> Vector X_obs
    # 3. Flatten Each Pert Data -> Vector X_i
    # 4. Construct Matrix A where col i = X_i - X_ref
    # 5. Construct b = X_obs - X_ref
    # 6. Weighting: We need to weight rows by 1/Sigma (Obs noise/variance).
    #    Equivalent to normalizing data by obs std.
    
    print("Gathering Data Vectors...")
    
    # Helper to flatten
    def flatten_state(geos_d, mit_d, obs_geos_d, obs_mit_d):
        """
        Extracts all relevant variables, weights them, and flattens to 1D array.
        weights are 1/std(obs).
        """
        vecs = []
        
        # GEOS
        for season in SEASONS:
            if season not in geos_d or season not in obs_geos_d: continue
            mod = geos_d[season]
            obs = obs_geos_d[season]
            
            for v_key, v_name in VAR_MAP_GEOS.items():
                if v_name in mod and v_name in obs:
                    m_data = mod[v_name]
                    o_data = obs[v_name].interp_like(m_data, method='nearest')
                    
                    std = o_data.std().values
                    if std < 1e-9: std = 1.0
                    
                    # Normalize
                    diff = (m_data - o_data) / std
                    vecs.append(diff.values.flatten())

        # MIT
        for season in SEASONS:
            if season not in mit_d or season not in obs_mit_d: continue
            mod = mit_d[season]
            obs = obs_mit_d[season]
            
            for v_key, v_name in VAR_MAP_MIT.items():
                if v_name in mod and v_name in obs:
                    m_data = mod[v_name]
                    o_data = obs[v_name].interp_like(m_data, method='nearest')
                    
                    std = o_data.std().values
                    if std < 1e-9: std = 1.0
                    
                    diff = (m_data - o_data) / std
                    vecs.append(diff.values.flatten())
                            
        if not vecs: return np.array([])
        return np.concatenate(vecs)
    
    # To construct A, we need vectors of (X_mod), NOT (X_mod - Obs).
    # Wait, simple least squares A x = b
    # We want to minimize || (X_ref + A x) - Obs ||^2
    # So minimize || A x - (Obs - X_ref) ||^2
    # So b = Obs - X_ref.
    # And columns of A are (X_pert_i - X_ref).
    # However, strictly speaking, S_i = (X_pert_i - X_ref) / delta_p_i
    # Then x is delta_p.
    # But usually we just say x is the "fraction of perturbation" if we define column as full diff.
    # Let's solve for 'w' (weights) then convert to parameters.
    
    # Identify common grid structure from Reference Experiment
    # We will interpolate EVERYTHING (Obs and Perturbations) to the Reference Model Grid.
    
    def align_and_extract(mod_geos, mod_mit, obs_geos, obs_mit, ref_geos=None, ref_mit=None, weights_dict=None):
        """
        Extracts flattened vector from model and obs.
        Crucially: Interps OBS to MODEL grid.
        Returns: (vec_mod, vec_obs, vec_sigma, vec_wgt)
        If ref_geos/ref_mit provided (for perturbations), interps mod to ref grid (though usually same).
        """
        vals_mod = []
        vals_obs = [] # Aligned to mod
        vals_sig = []
        vals_wgt = []
        
        # GEOS
        for season in SEASONS:
             if season in mod_geos and season in obs_geos:
                 m = mod_geos[season]
                 o = obs_geos[season]
                 r = ref_geos[season] if ref_geos else m # Template grid
                 
                 for v in VAR_MAP_GEOS.values():
                     if v in m and v in o:
                         # 1. Get Model Data
                         dat_m = m[v]
                         
                         # 2. Get Obs Data & Aligh to Model Grid (Template)
                         # Note: using 'r' (Reference) as template ensures all perts align to Ref.
                         dat_o = o[v].interp_like(r[v], method='nearest')
                         
                         # 3. If this is a perturbation, align it to Reference too?
                         # Usually Model grids are identical. Safe to assume.
                         # But consistency check:
                         if dat_m.shape != r[v].shape:
                             # This happens if perturbation changed grid? Unlikely.
                             # Or if we compare different resolutions.
                             dat_m = dat_m.interp_like(r[v], method='nearest')
                         
                         # Force Squeeze to remove degenerate dims (e.g. time=1, depth=1)
                         dat_m = dat_m.squeeze()
                         dat_o = dat_o.squeeze()
                             
                         # 4. Sigma (Spatial STD of Obs)
                         std = dat_o.std().values
                         if std < 1e-12: std = 1.0
                         
                         if dat_m.shape != dat_o.shape:
                             # Try one last align if squeeze didn't enforce it (e.g. slight grid mismatch)
                             dat_m = dat_m.interp_like(dat_o, method='nearest')

                         if dat_m.shape != dat_o.shape:
                             raise ValueError(f"Shape Mismatch for {v} {season}! Mod:{dat_m.shape} Obs:{dat_o.shape}. Check data dimensions.")
                         
                         vals_mod.append(dat_m.values.flatten())
                         vals_obs.append(dat_o.values.flatten())
                         vals_sig.append(np.full(dat_m.size, std))
                         
                         # Weight
                         w = 1.0
                         if weights_dict and v in weights_dict:
                             w = weights_dict[v]
                         vals_wgt.append(np.full(dat_m.size, w))

        # MIT
        for season in SEASONS:
             if season in mod_mit and season in obs_mit:
                 m = mod_mit[season]
                 o = obs_mit[season]
                 r = ref_mit[season] if ref_mit else m

                 for v in VAR_MAP_MIT.values():
                     if v in m and v in o:
                         dat_m = m[v]
                         dat_o = o[v].interp_like(r[v], method='nearest')
                         
                         if dat_m.shape != r[v].shape:
                             dat_m = dat_m.interp_like(r[v], method='nearest')
                             
                         # Force Squeeze
                         dat_m = dat_m.squeeze()
                         dat_o = dat_o.squeeze()
                             
                         std = dat_o.std().values
                         if std < 1e-12: std = 1.0
                         
                         if dat_m.shape != dat_o.shape:
                              dat_m = dat_m.interp_like(dat_o, method='nearest')
                              
                         if dat_m.shape != dat_o.shape:
                             raise ValueError(f"Shape Mismatch for {v} {season}! Mod:{dat_m.shape} Obs:{dat_o.shape}. Check data dimensions.")
                         
                         vals_mod.append(dat_m.values.flatten())
                         vals_obs.append(dat_o.values.flatten())
                         vals_sig.append(np.full(dat_m.size, std))
                         
                         # Weight
                         w = 1.0
                         if weights_dict and v in weights_dict:
                             w = weights_dict[v]
                         vals_wgt.append(np.full(dat_m.size, w))
                         
        if not vals_mod: return None, None, None, None
        
        return np.concatenate(vals_mod), np.concatenate(vals_obs), np.concatenate(vals_sig), np.concatenate(vals_wgt)

    print("  Processing Reference...")
    # Get Ref Vector and Aligned Obs Vector
    # For Ref, template is itself.
    v_ref, v_obs, v_sigma, v_weights = align_and_extract(ref_data['GEOS'], ref_data['MIT'], 
                                              obs_data['GEOS'], obs_data['MIT'], 
                                              weights_dict=weights)
                                              
    if v_ref is None:
        print("Error: No common variables found between Model and Obs.")
        return

    # Normalized Ref and Obs
    # Apply Weights: Minimize || sqrt(W) * (Model - Obs) / Sigma ||^2
    # So we scale normalized vector by sqrt(W)
    
    sqrt_w = np.sqrt(v_weights)
    
    X_ref = (v_ref / v_sigma) * sqrt_w
    X_obs = (v_obs / v_sigma) * sqrt_w # Obs mapped to Ref Grid
    
    # Perturbations
    A_cols = []
    exps_indices = []
    
    print(f"  Processing {len(pert_data_list)} perturbations...")
    
    for i, p_data in enumerate(pert_data_list):
        # We pass Ref Grid as template to ensure exact alignment
        # Weights passed here too (though just for consistency, we assume structure same)
        v_p, _, _, _ = align_and_extract(p_data['GEOS'], p_data['MIT'], 
                                      obs_data['GEOS'], obs_data['MIT'],
                                      ref_geos=ref_data['GEOS'], ref_mit=ref_data['MIT'],
                                      weights_dict=weights)
        
        if v_p is None or len(v_p) != len(X_ref):
            print(f"    [WARN] Size mismatch for {p_data['ID']}. Skipping.")
            continue
            
        # Normalize by SAME sigma as Ref AND apply same weights
        X_p = (v_p / v_sigma) * sqrt_w
        
        # Column = X_p - X_ref
        diff = X_p - X_ref
        A_cols.append(diff)
        exps_indices.append(p_data['ID'])
        
    if not A_cols:
        print("No valid perturbation data.")
        return
        
    A = np.stack(A_cols, axis=1) # (N_points, N_params)
    b = X_obs - X_ref            # (N_points,)
    
    # --- Sanitize Data (Remove NaNs/Infs) ---
    # Common in geophysical data (land/ocean masks)
    valid_mask = np.isfinite(b) & np.all(np.isfinite(A), axis=1)
    
    n_total = len(b)
    n_valid = np.sum(valid_mask)
    
    if n_valid < n_total:
        print(f"  [INFO] Masking invalid points: Keeping {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
        A = A[valid_mask]
        b = b[valid_mask]
        
    if n_valid == 0:
        print("  [ERROR] No valid data points left after masking NaNs. Aborting.")
        return

    print(f"  Solving Least Squares: A shape {A.shape}, b shape {b.shape}")
    
    # --- Regularization (Alpha) Sensitivity Analysis ---
    alphas = [0, 0.1, 0.3]
    w_final = None # Will store alpha=0 result
    
    print("\n--- Sensitivity Analysis for Regularization (Alpha) ---")
    
    for alpha in alphas:
        print(f"\nSolving with Alpha = {alpha}...")
        
        if alpha == 0:
            # Standard Least Squares
            w, res, rank, s = lstsq(A, b)
            w_final = w # Save for downstream use
            
        else:
            # Tikhonov Regularization (Ridge Regression)
            # Solves: min ||Ax - b||^2 + alpha * ||x||^2
            # Normal Formula: w = (A^T A + alpha I)^-1 A^T b
            # Note: A is (N_obs, N_params). 
            # Check dimension of alpha * I. It matches N_params x N_params.
            # But the formula relies on A^T A (Covariance matrix).
            
            # Using scaled alpha for consistency with number of observations?
            # Standard form is usually just alpha*I added to Hessian.
            # PROBLEM: ATA scales with N_obs (~100k+). Alpha scales with 1.
            # Alpha gets drowned out. We must scale alpha by N_obs or similar.
            
            n_obs = A.shape[0]
            n_params = A.shape[1]
            ATA = A.T @ A
            ATb = A.T @ b
            
            # Ridge Solver
            # (ATA + alpha * N_obs * I) w = ATb
            # We scale alpha by N_obs to make it invariant to grid size
            
            effective_alpha = alpha * n_obs
            REG_MATRIX = ATA + effective_alpha * np.eye(n_params)
            
            try:
                w = np.linalg.solve(REG_MATRIX, ATb)
            except np.linalg.LinAlgError:
                print(f"  [ERROR] Singular matrix for alpha={alpha}. Using SVD inverse.")
                w = np.linalg.pinv(REG_MATRIX) @ ATb

        # --- Report for this Alpha ---
        cost_init = np.sum(b**2)
        cost_final = np.sum((A @ w - b)**2) + alpha * np.sum(w**2) # Regularized Cost? Or just fit?
        # Usually report Actual Fit Cost (without penalty) to see trade-off
        fit_cost = np.sum((A @ w - b)**2)
        reduc = (cost_init - fit_cost) / cost_init * 100
        
        print(f"  Fit Cost: {fit_cost:.4e} (Reduction: {reduc:.2f}%)")
        
        if alpha > 0:
             print("  Detailed Parameter Changes for Information Only:")
             print(f"  {'Exp':<6} {'Original':<12} {'Optimized':<12} {'Change%':<10}")
             
             # Collect and Sort
             alpha_results = []
             for i, exp_id in enumerate(exps_indices):
                 weight = w[i]
                 p_row = param_info[param_info['Exp'] == exp_id]
                 if p_row.empty: continue
                 p_row = p_row.iloc[0]
                 orig = p_row['Original']
                 pert = p_row['Perturbed']
                 opt_val = orig + weight * (pert - orig)
                 pct_change = ((opt_val - orig) / orig) * 100 if orig != 0 else 0
                 
                 alpha_results.append({
                     'Exp': exp_id,
                     'Original': orig,
                     'Optimized': opt_val,
                     'ChangePct': pct_change
                 })
             
             # Sort by absolute change
             alpha_results.sort(key=lambda x: abs(x['ChangePct']), reverse=True)
             
             for res in alpha_results:
                 print(f"  {res['Exp']:<6} {res['Original']:<12.4g} {res['Optimized']:<12.4g} {res['ChangePct']:<+10.2f}%")

    print(f"\n--- Finalizing Results with Alpha = 0 (Standard Least Squares) ---")
    w = w_final
    
    # --- Report ---
    # Optimal Parameter = Orig + w_i * (Perturbed - Orig)
    
    results = []
    print_rows = []
    
    for i, exp_id in enumerate(exps_indices):
        weight = w[i]
        
        # Find param info
        p_row = param_info[param_info['Exp'] == exp_id]
        if p_row.empty:
            continue
        p_row = p_row.iloc[0]
        
        orig = p_row['Original']
        pert = p_row['Perturbed']
        # delta = p_row['Delta']
        
        # P_opt = P_ref + w * (P_pert - P_ref)
        opt_val = orig + weight * (pert - orig)
        
        # Change % relative to full range or original? 
        pct_change = ((opt_val - orig) / orig) * 100 if orig != 0 else 0
        
        # Collect for printing and saving
        row_data = {
            'Rank': 0, # Placeholder, will fill after sort
            'Exp': exp_id,
            'Parameter': p_row['Parameter'],
            'Original': orig,
            'Perturbed': pert,
            'Optimized': opt_val,
            'Weight': weight,
            'ChangePct': pct_change
        }
        print_rows.append(row_data)

    # Sort Results by Sensitivity (Abs Change %)
    print_rows.sort(key=lambda x: abs(x['ChangePct']), reverse=True)
    
    print("\nOPTIMIZATION RESULTS (Alpha=0) - Sorted by Sensitivity:")
    print(f"{'Exp':<6} {'Parameter':<20} {'Original':<12} {'Perturbed':<12} {'Optimized':<12} {'Change%':<10}")
    
    for idx, row in enumerate(print_rows):
        # Update Rank based on sensitivity
        row['Rank'] = idx + 1
        results.append(row)
        
        print(f"{row['Exp']:<6} {row['Parameter']:<20} {row['Original']:<12.4g} {row['Perturbed']:<12.4g} {row['Optimized']:<12.4g} {row['ChangePct']:<+10.2f}%")
        
    # Validation/Cost Reduction (Final Alpha=0)
    cost_init = np.sum(b**2)
    cost_final = np.sum((A @ w - b)**2)
    reduc = (cost_init - cost_final) / cost_init * 100
    
    print(f"\nInitial Cost: {cost_init:.4e}")
    print(f"Final Cost:   {cost_final:.4e}")
    print(f"Reduction:    {reduc:.2f}%")
    
    # Save CSV
    pd.DataFrame(results).to_csv("optimization_results.csv", index=False)
    print("\nSaved results to optimization_results.csv")
    
    # Generate Diagnostics
    # Create map of ExpID -> Weight
    weight_map = {id: w for id, w in zip(exps_indices, w)}
    generate_diagnostics(ref_data, pert_data_list, obs_data, weight_map, param_info)

# --- Main Driver ---

# --- Main Driver ---

def main():
    parser = argparse.ArgumentParser(description="Green's Function Tuning")
    parser.add_argument('--test', action='store_true', help="Run in mock test mode")
    parser.add_argument('--weighted', action='store_true', help="Apply variable-specific weights to cost function")
    parser.add_argument('--root_dir', type=str, default="/nobackupp27/afahad/GFEXPv11.7", help="Root experiment directory")
    args = parser.parse_args()
    
    ROOT_DIR = args.root_dir
    TARGET_DIR = TARGET_DIR_DEFAULT
    
    if args.test:
        print("!!! RUNNING IN TEST MODE !!!")
        ROOT_DIR = "MOCK_DATA/GFEXP"
        TARGET_DIR = generate_mock_data(ROOT_DIR)
        
    # 1. Get Parameters (Hardcoded)
    print("Loading parameters from script configuration...")
    param_df = get_experiment_parameters()
    print(f"Loaded {len(param_df)} parameters.")
    
    # 2. Identify Experiments
    ref_exp = "GF01_ref"
    perturbed_exps = param_df['Exp'].unique().tolist()
    
    # 3. Load Targets
    print("Loading Observations...")
    obs_data = load_targets(TARGET_DIR)
    
    # 4. Load Reference Data
    print(f"Loading Reference {ref_exp}...")
    ref_geos = {}
    ref_mit = {}
    for s in SEASONS:
        ref_geos[s] = load_geos_data(ROOT_DIR, ref_exp, s)
        ref_mit[s] = load_mitgcm_data(ROOT_DIR, ref_exp, s)
        
    ref_container = {'GEOS': ref_geos, 'MIT': ref_mit}
    
    # 5. Load Perturbed Data
    pert_container_list = []
    
    for exp_id in perturbed_exps:
        print(f"Loading {exp_id}...")
        p_geos = {}
        p_mit = {}
        valid = True
        for s in SEASONS:
            ds_g = load_geos_data(ROOT_DIR, exp_id, s)
            ds_m = load_mitgcm_data(ROOT_DIR, exp_id, s)
            
            if ds_g: p_geos[s] = ds_g
            if ds_m: p_mit[s] = ds_m
            
        pert_container_list.append({
            'ID': exp_id,
            'GEOS': p_geos,
            'MIT': p_mit
        })
        
    # 6. Optimization
    # Define Weights (Internal Names)
    # FRSEAICE 1, H500 0.2, LWGNET 1, SWGNET 1, SALT300 .8, THETA300 .8, THETA0 1.2, SALT0 .8, T2M 1.2, TS 1.2, T300 1.2, U200 1.2
    
    var_weights = None
    if args.weighted:
        print(">>> USING WEIGHTED COST FUNCTION <<<")
        var_weights = {
            'FRSEAICE': 1.0,
            'H500': 0,
            'LWGNET': 1.0,
            'SWGNET': 1.0,
            'SALT_300m': 0.8,
            'THETA_300m': 0.8,
            'THETA_0m': 4,
            'SALT_0m': 0.8,
            'T2M': 1.2,
            'TS': 0,
            'T300': 2,
            'U200': 4
        }
    
    optimize_greens_function(ref_container, pert_container_list, obs_data, param_df, weights=var_weights)


if __name__ == "__main__":
    main()
