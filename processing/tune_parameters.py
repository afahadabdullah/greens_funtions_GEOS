#!/usr/bin/env python
"""
Green's Function Parameter Tuning for GEOS Model.
Targeting: 2m Temperature (TS) and Precipitation (PRECTOT).
Timeframe: July 2006 and December 2006.

Algorithm:
1.  Load Reference (GF01) and Perturbed (GF02-GF22) runs.
2.  Load Target (ERA5) data.
3.  Compute Sensitivity Matrix (G).
4.  Solve delta_lambda = (G^T G + alpha*I)^-1 G^T (y_obs - y_ref).
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.linalg import pinv

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = '/home6/afahad/nobackup27/GFEXPv11.7'
GFO_DIR = os.path.join(BASE_DIR, 'scripts/GFO')
OBS_FILE = os.path.join(GFO_DIR, 'merra2_target_2006_jul_dec.nc')

# Parameter Definitions (from HTML)
# Exp: (ParamName, RefValue, PertValue)
PARAMS = {
    'GF02': ('TURNRHCRIT', 726.0, 850.0),
    'GF03': ('ENTR_DP', 1.0e-4, 4.0e-4),
    'GF04': ('WBFFACTOR', 0.091, 0.3),
    'GF05': ('DCS', 2.91e-4, 2.5e-4),
    'GF06': ('FDROP_DUST', 0.5, 0.1),
    'GF07': ('DROPSZCNV', 2.5e-5, 5.0e-5),
    'GF08': ('FDROP_SOOT', 0.05, 0.09),
    'GF09': ('MINCDNC', 52.0, 25.0),
    'GF10': ('TURB_BETA_RAD', 0.2, 0.3),
    'GF11': ('TURB_KHRADFAC', 0.85, 0.5),
    'GF12': ('MINRHCRIT', 0.89, 0.80),
    'GF13': ('QRC_CRIT', 2.0e-4, 4.0e-4),
    'GF14': ('RKM', 8.0, 12.0),
    'GF15': ('AUT_SCALE', 1.7, 0.5),
    'GF16': ('UISCALE', 1.0, 0.7),
    'GF17': ('CICE_AH_MAX', 0.3, 0.2),
    'GF18': ('ALBICEV', 0.78, 0.73),
    'GF19': ('VSLfactor', 1.0, 0.8),
    'GF20': ('Charnok1', 0.20303e-5, 0.28425e-5),
    'GF21': ('Charnok2', 0.215e-2, 0.315e-2),
    'GF22': ('CQfactor', 1.0, 0.5),
}

# Experiments to include in optimization
EXP_LIST = [f'GF{i:02d}' for i in range(2, 23)]

# Variables to use
# GEOS names: TS, T200, H500
# Obs Mapped names: TS, T200, H500
TARGET_VARS = ['TS', 'T200', 'H500'] 

# Scaling factors for weights (manual adjustment)
# Weight = (1/std) * Scale
VAR_SCALES = {
    'TS': 1.0,
    'T200': 1.0,
    'H500': 1.0
} 

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_data(exp_name, year, month):
    """
    Load monthly mean data for a specific experiment, year, month.
    Returns ds with 'TS' from geosgcm_surf and 'T200' from geosgcm_prog.
    """
    date_str = f"{year}{month:02d}"
    
    # 1. Determine Full Experiment Name and Directory
    if exp_name == 'GF01_ref':
        full_exp_name = 'GF01_ref'
        exp_dir = os.path.join(BASE_DIR, 'GF01_ref')
    else:
        exp_dirs = glob.glob(os.path.join(BASE_DIR, f"{exp_name}_*"))
        if not exp_dirs:
            print(f"Warning: Directory for {exp_name} not found.")
            return None
        exp_dir = exp_dirs[0]
        full_exp_name = os.path.basename(exp_dir)

    def find_file(collection):
        """Find file for a given collection (geosgcm_surf or geosgcm_prog)."""
        candidates = [
            os.path.join(exp_dir, collection, f"{full_exp_name}.{collection}.monthly.{date_str}.nc4"),
            os.path.join(exp_dir, f'holding/{collection}', f"{full_exp_name}.{collection}.monthly.{date_str}.nc4"),
            os.path.join(exp_dir, collection, f"{full_exp_name}.{collection}.{date_str}??_0000z.nc4"),
            os.path.join(exp_dir, f'holding/{collection}', f"{full_exp_name}.{collection}.{date_str}??_0000z.nc4")
        ]
        for cand in candidates:
            found = glob.glob(cand)
            if found:
                return found[0]
        return None

    # Load geosgcm_surf for TS
    surf_file = find_file('geosgcm_surf')
    if not surf_file:
        print(f"Warning: geosgcm_surf not found for {exp_name} {date_str}")
        return None

    # Load geosgcm_prog for T (3D temperature)
    prog_file = find_file('geosgcm_prog')
    if not prog_file:
        print(f"Warning: geosgcm_prog not found for {exp_name} {date_str}")
        return None

    try:
        ds_surf = xr.open_dataset(surf_file)
        ds_prog = xr.open_dataset(prog_file)
        
        # Extract T at 200 hPa
        T200 = ds_prog['T'].sel(lev=200, method='nearest').drop_vars('lev')
        
        # Extract H at 500 hPa
        H500 = ds_prog['H'].sel(lev=500, method='nearest').drop_vars('lev')
        
        # Create combined dataset
        ds_out = xr.Dataset()
        ds_out['TS'] = ds_surf['TS']
        ds_out['T200'] = T200
        ds_out['H500'] = H500
        ds_out.coords.update(ds_surf.coords)
        
        return ds_out
    except Exception as e:
        print(f"Error loading {exp_name} {date_str}: {e}")
        return None

def preprocess_obs_month(obs_path, model_grid_ds, target_date):
    """
    Load Observations (MERRA2/ERA5), select specific month, regrid to model grid.
    """
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Obs file not found: {obs_path}")
        
    ds_obs = xr.open_dataset(obs_path)
    
    # Rename coords
    model_coords = list(model_grid_ds.coords)
    obs_coords = list(ds_obs.coords)
    
    rename_dict = {}
    if 'latitude' in obs_coords:
        if 'lat' in model_coords: rename_dict['latitude'] = 'lat'
        elif 'lats' in model_coords: rename_dict['latitude'] = 'lats'
    if 'longitude' in obs_coords:
        if 'lon' in model_coords: rename_dict['longitude'] = 'lon'
        elif 'lons' in model_coords: rename_dict['longitude'] = 'lons'
            
    if rename_dict:
        ds_obs = ds_obs.rename(rename_dict)
    
    # Handle Longitude wrap (0..360 -> -180..180) if model is -180..180
    if 'lon' in model_grid_ds.coords:
        model_lon_min = model_grid_ds.lon.min()
        if model_lon_min < 0:
            # Check if obs needs shifting
            # MERRA2 is -180..180 usually, but let's check
            if ds_obs.lon.max() > 180:
                print("Shifting Obs longitude to -180..180 to match model.")
                ds_obs.coords['lon'] = (ds_obs.coords['lon'] + 180) % 360 - 180
                ds_obs = ds_obs.sortby('lon')
    
    # Select month
    ds_obs_mon = ds_obs.sel(time=target_date, method='nearest')
    
    # Regrid
    ds_obs_interp = ds_obs_mon.interp_like(model_grid_ds, method='linear')
    
    ds_out = xr.Dataset()
    
    # TS Mapping
    if 'TS' in ds_obs_interp:
        ds_out['TS'] = ds_obs_interp['TS']
    elif 't2m' in ds_obs_interp:
        ds_out['TS'] = ds_obs_interp['t2m']
        
    # T200 Mapping
    if 'T200' in ds_obs_interp:
        ds_out['T200'] = ds_obs_interp['T200']
    elif 't200' in ds_obs_interp:
        ds_out['T200'] = ds_obs_interp['t200']
        
    # H500 Mapping
    if 'H500' in ds_obs_interp:
        ds_out['H500'] = ds_obs_interp['H500']
    
    return ds_out

class GreenTuner:
    def __init__(self):
        self.ref_data = {}  # {'Jul': ds, 'Dec': ds}
        self.pert_data = {} # {'GF02': {'Jul': ds, 'Dec': ...}, ...}
        self.obs_data = {}  # {'Jul': ds, 'Dec': ds}
        self.weights = {}   # Normalization weights
        
    def load_datasets(self):
        print("Loading Reference (GF01)...")
        self.ref_data['Jul'] = get_model_data('GF01_ref', 2006, 7)
        self.ref_data['Dec'] = get_model_data('GF01_ref', 2006, 12)
        
        print("Loading Observations (MERRA2)...")
        try:
            self.obs_data['Jul'] = preprocess_obs_month(OBS_FILE, self.ref_data['Jul'], '2006-07-01')
            self.obs_data['Dec'] = preprocess_obs_month(OBS_FILE, self.ref_data['Dec'], '2006-12-01')
        except Exception as e:
            print(f"Error loading Obs: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("Loading Perturbed Experiments...")
        for exp in EXP_LIST:
            ds_jul = get_model_data(exp, 2006, 7)
            ds_dec = get_model_data(exp, 2006, 12)
            
            if ds_jul is not None and ds_dec is not None:
                self.pert_data[exp] = {}
                self.pert_data[exp]['Jul'] = ds_jul
                self.pert_data[exp]['Dec'] = ds_dec
            else:
                print(f"Skipping {exp} due to missing data.")
                
        return True

    def compute_cost(self, data_dict):
        """
        Compute cost function J = (y - y_obs)^T W (y - y_obs)
        """
        pass

    def flat_state_vector(self, data_dict):
        """
        Flatten TS_Jul, TS_Dec, Pr_Jul, Pr_Dec into a single 1D vector.
        Apply area weighting? For now, flat.
        """
        state = []
        # Order: TS_Jul, TS_Dec, Pr_Jul, Pr_Dec
        for var in TARGET_VARS:
            for mon in ['Jul', 'Dec']:
                field = data_dict[mon][var].values.flatten()
                # Normalize?
                # We need normalization factors to combine Temperature (K) and Precip (kg/m2/s)
                norm = self.weights.get((var, mon), 1.0)
                state.append(field / norm)
        
        return np.concatenate(state)

    def compute_weights(self):
        """
        Compute normalization weights based on Reference Variance or Standard Deviation.
        """
        for var in TARGET_VARS:
            for mon in ['Jul', 'Dec']:
                # Inverse variance weighting (or Inverse Std Dev)
                # W = 1 / sigma
                # Here we use 1 / std(Obs) as a baseline normalization
                
                obs_data = self.obs_data[mon][var]
                std_val = np.nanstd(obs_data.values.flatten())
                if std_val == 0 or np.isnan(std_val): std_val = 1.0
                
                base_weight = 1.0 / std_val
                
                # Apply manual scale
                scale = VAR_SCALES.get(var, 1.0)
                final_weight = base_weight * scale
                
                key = f"{var}_{mon}"
                self.weights[key] = final_weight
                print(f"Weight for {key}: {final_weight:.4f} (Base: {base_weight:.4f}, Scale: {scale})")

    def tune(self):
        # 1. Setup Reference Vector (y_ref)
        self.compute_weights()
        y_ref = self.flat_state_vector(self.ref_data)
        y_obs = self.flat_state_vector(self.obs_data)
        
        # 2. Build Sensitivity Matrix G
        # Matrix G has shape (N_obs, N_params)
        # N_obs = len(y_ref)
        # N_params = len(self.pert_data)
        
        valid_exps = sorted(list(self.pert_data.keys()))
        n_params = len(valid_exps)
        n_obs = len(y_ref)
        
        print(f"State Vector Size: {n_obs}")
        print(f"Number of Parameters: {n_params}")
        
        G = np.zeros((n_obs, n_params))
        
        # delta_lambda calculation
        # Normalized parameters? 
        # Usually we solve for relative change or absolute change.
        # Let's solve for relative change x = delta_param / (param_pert - param_ref)
        # No, let's solve for x where param_new = param_ref + x * (param_pert - param_ref)
        # So x=0 is Ref, x=1 is Pert.
        # Then Sensitivity = (y_pert - y_ref) / (1 - 0) = y_pert - y_ref.
        # The optimized parameter value will be Ref + x_opt * (Pert - Ref).
        
        for i, exp in enumerate(valid_exps):
            y_pert = self.flat_state_vector(self.pert_data[exp])
            # Sensitivity column
            G[:, i] = y_pert - y_ref
            
        # 3. Optimization
        d_obs = y_obs - y_ref
        
        # Check for NaNs
        print(f"NaNs in d_obs: {np.isnan(d_obs).sum()}")
        print(f"NaNs in G: {np.isnan(G).sum()}")
        
        # Mask out NaNs
        valid_mask = ~np.isnan(d_obs)
        # Also mask rows in G where any column is NaN
        valid_mask &= ~np.any(np.isnan(G), axis=1)
        
        print(f"Valid points for optimization: {valid_mask.sum()} / {len(d_obs)}")
        
        G_clean = G[valid_mask]
        d_obs_clean = d_obs[valid_mask]
        

        
        # Regularization (Ridge)
        alpha = 0.1 
        
        x_opt, residuals, rank, s = np.linalg.lstsq(G_clean, d_obs_clean, rcond=None)
        
        print("\nOptimization Results:")
        print(f"{'Exp':<6} {'Param':<15} {'Ref':<10} {'Pert':<10} {'Optimized':<10} {'Change(%)':<10}")
        print("-" * 75)
        
        for i, exp in enumerate(valid_exps):
            param_name, ref_val, pert_val = PARAMS[exp]
            
            # x_opt[i] is the scaling factor of the perturbation
            # New Val = Ref + x * (Pert - Ref)
            
            opt_val = ref_val + x_opt[i] * (pert_val - ref_val)
            pct_change = ((opt_val - ref_val) / ref_val) * 100.0 if ref_val != 0 else 0
            
            print(f"{exp:<6} {param_name:<15} {ref_val:<10.2g} {pert_val:<10.2g} {opt_val:<10.4g} {pct_change:<+10.1f}")
            
        # Error Improvement
        y_opt = y_ref + np.dot(G, x_opt)
        
        def calc_rms(vec):
            return np.sqrt(np.nanmean(vec**2))
            
        rms_ref = calc_rms(y_obs - y_ref)
        rms_opt = calc_rms(y_obs - y_opt)
        
        print("\nCost Analysis (Normalized RMS):")
        print(f"Reference Error: {rms_ref:.4f}")
        print(f"Optimized Error: {rms_opt:.4f}")
        print(f"Reduction:       {(rms_ref - rms_opt)/rms_ref * 100:.1f}%")

        # Plotting optimization weights
        plt.figure(figsize=(10, 6))
        plt.bar(valid_exps, x_opt)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title('Optimization Weights (x)\nNew = Ref + x * (Pert - Ref)')
        plt.ylabel('x (Perturbation Factor)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(GFO_DIR, 'optimization_results.png'))
        print(f"Saved plot to {os.path.join(GFO_DIR, 'optimization_results.png')}")
        
        # 4. Visualization
        opt_data = self.reconstruct_fields(x_opt, valid_exps)
        
        # Plot analysis for each variable
        for var in TARGET_VARS:
            self.plot_variable_analysis(var, opt_data)
            
        # Plot Cost Comparison
        self.plot_cost_comparison(valid_exps, x_opt)
        
        # Plot Cost Breakdown by Variable
        self.plot_cost_breakdown(valid_exps, x_opt)

    def reconstruct_fields(self, x_opt, valid_exps):
        """Reconstruct optimized fields."""
        opt_data = {} # {'Jul': ds, 'Dec': ds}
        
        for mon in ['Jul', 'Dec']:
            ds_opt = self.ref_data[mon].copy(deep=True)
            for i, exp in enumerate(valid_exps):
                weight = x_opt[i]
                if exp in self.pert_data and mon in self.pert_data[exp]:
                    ds_pert = self.pert_data[exp][mon]
                    ds_ref = self.ref_data[mon]
                    ds_delta = (ds_pert - ds_ref) * weight
                    ds_opt = ds_opt + ds_delta
            opt_data[mon] = ds_opt
        return opt_data

    def plot_variable_analysis(self, var, opt_data):
        import cartopy.crs as ccrs
        print(f"Plotting analysis for {var}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), 
                                 subplot_kw={'projection': ccrs.Robinson()})
        
        rows = ['Jul', 'Dec']
        cols = ['Ref Bias (Ref-Obs)', 'Projected Change (Opt-Ref)', 'New Bias (Opt-Obs)']
        
        # Set shared vmin/vmax based on Ref Bias to keep scale consistent
        # Or adaptive. Let's use fixed reasonable ranges.
        # TS: 5K, T200: 5K, H500: 50m?
        if var == 'TS': vmax=5; units='K'
        elif var == 'T200': vmax=5; units='K'
        elif var == 'H500': vmax=50; units='m'
        else: vmax=None; units=''
        
        for r, mon in enumerate(rows):
            obs = self.bias_prep(self.obs_data[mon][var])
            ref = self.bias_prep(self.ref_data[mon][var])
            opt = self.bias_prep(opt_data[mon][var])
            
            # 1. Ref Bias
            diff1 = ref - obs
            # 2. Change
            diff2 = opt - ref
            # 3. New Bias
            diff3 = opt - obs
            
            data_list = [diff1, diff2, diff3]
            
            for c, data in enumerate(data_list):
                ax = axes[r, c]
                ax.set_global()
                ax.coastlines(alpha=0.3)
                
                plot_data = data
                # Handle 1D time/lev
                if len(plot_data.shape) > 2: plot_data = plot_data.squeeze()
                
                im = ax.pcolormesh(plot_data.lon, plot_data.lat, plot_data,
                                   transform=ccrs.PlateCarree(),
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                
                if r == 0: ax.set_title(cols[c], fontsize=12, fontweight='bold')
                if c == 0: ax.text(-0.07, 0.55, mon, transform=ax.transAxes, 
                                   va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)
                                   
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label(f"{var} ({units})")
        
        plt.suptitle(f"{var} Analysis: Biases and Projected Changes", fontsize=16)
        outfile = os.path.join(GFO_DIR, f'analysis_{var}.png')
        plt.savefig(outfile)
        print(f"Saved {outfile}")
        plt.close()

    def plot_cost_breakdown(self, valid_exps, x_opt):
        print("Breakdown Cost by Variable...")
        
        # Reconstruct Opt
        opt_data = self.reconstruct_fields(x_opt, valid_exps)
        
        # Calculate RMS for each variable (Ref vs Opt)
        vars_list = TARGET_VARS
        ref_costs = []
        opt_costs = []
        
        for var in vars_list:
            # Gather all months for this var
            # Ref
            ref_vec = []
            obs_vec = []
            opt_vec = []
            
            for mon in ['Jul', 'Dec']:
                # Weights are applied in flat_state_vector but here we want raw RMS or weighted?
                # User asked for "cost breakdown". "Cost" implies J, which includes weights.
                # But usually "RMS" is easier to interpret physically.
                # If variables have different units (K vs m), raw RMS is not comparable on same plot.
                # Normalized RMS (as used in cost function) is better for comparison.
                # I'll use Weighted/Normalized RMS.
                
                # Fetch weight
                weight = self.weights.get(f"{var}_{mon}", 1.0)
                
                # Get data vectors
                o = self.obs_data[mon][var].values.flatten()
                r = self.ref_data[mon][var].values.flatten()
                p = opt_data[mon][var].values.flatten()
                
                # Mask NaNs
                mask = ~np.isnan(o)
                
                # Append weighted differences
                # J = sum(w * (sim - obs))^2
                # RMS_J = sqrt(mean( (w*(sim-obs))^2 ))
                
                obs_vec.extend(o[mask])
                ref_vec.extend(r[mask])
                opt_vec.extend(p[mask])
                
                # Actually, I should apply weight to difference
                # But wait, weights might differ per month.
                # Let's compute weighted squared error sum, then normalize.
                
            # Compute Weighted RMS for this variable
            # We need to loop again properly to apply specific weights
            
            sq_err_ref = 0
            sq_err_opt = 0
            count = 0
            
            for mon in ['Jul', 'Dec']:
                weight = self.weights.get(f"{var}_{mon}", 1.0)
                o = self.obs_data[mon][var].values.flatten()
                r = self.ref_data[mon][var].values.flatten()
                p = opt_data[mon][var].values.flatten()
                mask = ~np.isnan(o)
                
                diff_ref = (r[mask] - o[mask]) * weight
                diff_opt = (p[mask] - o[mask]) * weight
                
                sq_err_ref += np.sum(diff_ref**2)
                sq_err_opt += np.sum(diff_opt**2)
                count += len(diff_ref)
                
            rms_ref = np.sqrt(sq_err_ref / count) if count > 0 else np.nan
            rms_opt = np.sqrt(sq_err_opt / count) if count > 0 else np.nan
            
            ref_costs.append(rms_ref)
            opt_costs.append(rms_opt)
            
        # Normalize per variable (Ref = 1)
        # We need to normalize each pair (Ref_i, Opt_i) by Ref_i
        
        norm_ref_costs = []
        norm_opt_costs = []
        
        for i in range(len(vars_list)):
            r_val = ref_costs[i]
            if r_val == 0: r_val = 1.0
            
            norm_ref_costs.append(ref_costs[i] / r_val) # Always 1.0
            norm_opt_costs.append(opt_costs[i] / r_val)
        
        # Plot Grouped Bar Chart
        x = np.arange(len(vars_list))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        rects1 = plt.bar(x - width/2, norm_ref_costs, width, label='Reference (1.0)', color='gray')
        rects2 = plt.bar(x + width/2, norm_opt_costs, width, label='Optimized', color='green')
        
        plt.ylabel('Relative Cost (Ref=1)')
        plt.title('Global Cost Breakdown by Variable')
        plt.xticks(x, vars_list)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add improvement labels
        for i, val_opt_norm in enumerate(norm_opt_costs):
            val_ref_norm = norm_ref_costs[i] # 1.0
            
            # Pct improvement is same regardless of scaling
            # (1 - Opt/Ref) * 100
            val_ref_abs = ref_costs[i]
            val_opt_abs = opt_costs[i]
            if val_ref_abs != 0:
                pct = (val_ref_abs - val_opt_abs) / val_ref_abs * 100
            else:
                pct = 0.0
                
            plt.text(x[i] + width/2, val_opt_norm, f"{val_opt_norm:.2f}\n(-{pct:.1f}%)", 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        outfile = os.path.join(GFO_DIR, 'cost_breakdown.png')
        plt.savefig(outfile)
        print(f"Saved {outfile}")
        plt.close()

    def bias_prep(self, da):
        """Helper to squeeze/clean data for diffing."""
        return da.squeeze()

    def plot_cost_comparison(self, valid_exps, x_opt):
        print("Comparing Costs...")
        
        # Calculate Costs (RMS)
        def get_rms(d_dict):
            # Compute total RMS across all vars and months (weighted?)
            # Simplified: Just RMS of the vector y - y_obs
            # But here we want individual exp cost which implies re-running the flattening.
            # Faster: Construct the flat vector for the specific case.
            vec = self.flat_state_vector(d_dict)
            obs_vec = self.flat_state_vector(self.obs_data)
            return np.sqrt(np.nanmean((vec - obs_vec)**2))

        # 1. Ref Cost
        cost_ref = get_rms(self.ref_data)
        
        # 2. Opt Cost
        opt_data = self.reconstruct_fields(x_opt, valid_exps)
        cost_opt = get_rms(opt_data)
        
        # 3. Individual Exps
        exp_costs = []
        exp_names = []
        for exp in valid_exps:
            # Check if exp has data
            if exp in self.pert_data:
                cost = get_rms(self.pert_data[exp])
                exp_costs.append(cost)
                exp_names.append(exp)
        
        # Plot
        labels = ['Ref', 'Opt'] + exp_names
        values = [cost_ref, cost_opt] + exp_costs
        colors = ['gray', 'green'] + ['blue']*len(exp_names)
        
        # Normalize by Reference Cost
        scale = cost_ref if cost_ref > 0 else 1.0
        
        values = [v / scale for v in values]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(labels, values, color=colors)
        
        # Add labels
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", 
                     va='bottom', ha='center', fontsize=8, rotation=90)
            
        plt.ylabel('Relative Cost (Ref=1)')
        plt.title('Cost Comparison: Reference, Optimized, and Individual Experiments')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        outfile = os.path.join(GFO_DIR, 'cost_comparison.png')
        plt.savefig(outfile)
        print(f"Saved {outfile}")



if __name__ == "__main__":
    tuner = GreenTuner()
    if tuner.load_datasets():
        tuner.tune()
