import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import matplotlib.pyplot as plt
import time

# --- GFO Class (Adapted from gfo.py for Python 3) ---

class GFO(object):
    '''
    Green's functions optimization of model parameters based on 
    Menemenlis, D., I. Fukumori, and T. Lee, 2005: Using Green's 
    Functions to Calibrate an Ocean General Circulation Model., 
    Mon. Wea. Rev., 133, 1224-1240, https://doi.org/10.1175/MWR2912.1
    '''

    def __init__(self, y, x0, p0, x, p, w):
        # observations of type t_1, t_2, ..., t_N
        self.y = y     # y =   ((t_1:...), (t_2:...),..., (t_N:...))

        # model estimates from the unperturbed experiment e0
        self.x0 = x0   # x =   ((t_1:...), (t_2:...),..., (t_N:...))  # e0

        # model estimates from the perturbed experiments (e_1,..., e_k)
        self.x = x     

        # parameters of the unperturbed experiment
        self.p0 = p0   

        # list of perturbed parameters from the perturbed experiments
        self.p = p     

        # weights
        self.w = w

        if all(None is _ for _ in (y, x0, p0, x, p, w)):
            self.experiments = None
            self.variables   = None
        else:
            self.test()
            self.experiments = len(self.p)
            self.variables = len(self.y)

    def test(self):
        ''' Sanity check '''
        assert len(self.p0) == len(self.p)
        assert len(self.x)  == len(self.p)
        assert len(self.y)  == len(self.x0)
        assert len(self.y)  == len(self.w)
        for i in range(len(self.y)):
            assert len(self.y[i]) == len(self.x0[i])
            assert len(self.w[i]) == len(self.x0[i])

        for x_p in self.x:
            assert len(x_p) == len(self.x0)
            for i in range(len(self.x0)):
                assert len(x_p[i]) == len(self.x0[i])

    def estimate(self):
        n_var = self.variables              
        n_exp = self.experiments            

        # To store results
        ETA = {}  # {Q_val: eta_array}

        Yd = np.empty([0, n_exp])
        R  = np.empty(0)
        G  = np.empty([0, n_exp])
        W  = np.empty(0)

        cost = np.zeros([n_exp, n_var+1])   

        for j in range(n_var):
            S = len(self.y[j])              

            Yd0 = np.zeros([S, n_exp])      
            R0  = np.zeros([S])             
            G0  = np.zeros([S, n_exp])      

            for i in range(n_exp):
                y = np.array(self.y[j])     
                r = np.array(self.x0[j])    
                x = np.array(self.x[i][j]) 

                w = self.w[j]

                Yd0[:,i] = y - r            
                G0 [:,i] = x - r             

                # calculate covariance matrix (assume diagonal)
                R0[:] = np.nansum(w * (y - r)**2)
                
                # Check for zero variance to avoid division by zero
                if np.all(R0 == 0):
                    R0[:] = 1.0

                if np.abs(np.nansum(w)) > 0:
                    cost[i, j+1] = np.nansum(w * (Yd0[:,i] - G0[:,i])**2 / R0)
                else:
                    cost[i, j+1] = 0.0

            Yd = np.concatenate((Yd, Yd0))
            R  = np.concatenate((R, R0))
            G  = np.concatenate((G, G0))
            W  = np.concatenate((W, w))

        cost[:,0] = np.nansum(cost[:,1:], axis=1)

        ind = ((~np.isnan(Yd[:,0])) & (~np.isnan(R))) 
        if np.sum(ind) == 0:
            print("  [ERROR] No valid data points found in GFO estimate.")
            return None

        Ri = np.tile(W/R,(n_exp,1)).transpose()      
        GRG = np.matmul(np.transpose(G[ind,:]), (Ri[ind] * G[ind,:]))

        old_parameters       = np.array(self.p0)
        perturbed_parameters = np.array(self.p)
        diff_matrix = (perturbed_parameters - old_parameters) # (N_exp, N_param)
        
        # Initial cost
        re = np.nansum(W/R*(Yd[:,0])**2)
        
        results = {}  # {Q_val: {'eta': array, 'params': array, 'cost': float, 'improvement': float}}
        
        # --- Loop over Q values ---
        Q_values = [0, 0.5, 1.0]
        
        for Q_val in Q_values:
            try:
                if Q_val == 0:
                    # Unconstrained (Flat Prior)
                    P = np.linalg.inv(GRG)
                else:
                    Q = np.ones(n_exp) * Q_val
                    Qd = np.diag(1./(Q**2))
                    P = np.linalg.inv(Qd + GRG) 
                    
                eta = np.matmul(np.matmul(P, np.transpose(G[ind,:])), (Ri[ind,0] * Yd[ind,0]))
                update_vector = np.matmul(eta, diff_matrix)
                eta_new = old_parameters + update_vector
                
                # Projected cost
                pe = np.nansum(W/R*(Yd[:,0]-np.matmul(G, eta))**2)
                improvement = (1 - pe/re) * 100 if re > 0 else 0
                
                results[Q_val] = {
                    'eta': eta,
                    'params': eta_new,
                    'cost': pe,
                    'improvement': improvement
                }
                ETA[Q_val] = eta
                
            except np.linalg.LinAlgError:
                print(f"  [WARN] Matrix Singular for Q={Q_val}. Skipping.")
                results[Q_val] = None
        
        # Return all results
        return {
            'results': results,
            'initial_cost': re,
            'diff_matrix': diff_matrix,
            'old_params': old_parameters
        }

# --- Configuration & Constants ---

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
OBS_FILE_ECCO = "ECCO_L4_Climatology_2008-2017.nc" 
SEASONS = ['DJF', 'JJA']

# --- Parameter Configuration ---

def get_experiment_parameters():
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

# --- Data Loading (Simplified Copy from greens_func_tuning.py) ---

def load_geos_data(exp_dir, exp_name, season):
    fpath = os.path.join(exp_dir, exp_name, 'clim', 'geos_clim', f"{exp_name}.mean.{season}.nc")
    if not os.path.exists(fpath): return None
    try: return xr.open_dataset(fpath)
    except: return None

def load_mitgcm_data(exp_dir, exp_name, season):
    base_clim_dir = os.path.join(exp_dir, exp_name, 'clim')
    if not os.path.exists(base_clim_dir): return None
    file_map = {
        'SALT0': ('SALT', '0m'), 'SALT300': ('SALT', '300m'),
        'THETA0': ('THETA', '0m'), 'THETA300': ('THETA', '300m')
    }
    datasets = []
    for var_key, (vname, depth) in file_map.items():
        pattern = os.path.join(base_clim_dir, f"mitgcm.{vname}.{depth}.{season}*.nc")
        files = glob.glob(pattern)
        if not files: continue
        try:
            ds = xr.open_dataset(files[0])
            target_name = VAR_MAP_MIT[var_key]
            if vname in ds: ds = ds.rename({vname: target_name})
            datasets.append(ds)
        except: pass
    if not datasets: return None
    return xr.merge(datasets)

def load_targets(target_dir):
    obs_data = {'GEOS': {}, 'MIT': {}}
    for season in SEASONS:
        fname = f"MERRA2.seasonal.mean.{season}.nc"
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath):
            ds_ref = xr.open_dataset(fpath)
            rename_map = {'LWGNT': 'LWGNET', 'SWGNT': 'SWGNET'}
            for old, new in rename_map.items():
                if old in ds_ref: ds_ref = ds_ref.rename({old: new})
            if 'LWGNET' in ds_ref: ds_ref['LWGNET'] = -1.0 * ds_ref['LWGNET']
            obs_data['GEOS'][season] = ds_ref

    fpath_ecco = os.path.join(target_dir, OBS_FILE_ECCO)
    if os.path.exists(fpath_ecco):
        ds_ecco = xr.open_dataset(fpath_ecco)
        # Process ECCO variables to match internal keys (Slices)
        # We need THETA_0m, THETA_300m, SALT_0m, SALT_300m
        
        processed_vars = {}
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
        
        if processed_vars:
            ds_ecco_proc = xr.merge(processed_vars.values())
            
            # Check for Time/Season dimension
            t_dim = None
            for d in ['time', 'season', 't']:
                if d in ds_ecco_proc.dims:
                    t_dim = d
                    break
            
            if t_dim and ds_ecco_proc.sizes[t_dim] == 2:
                print(f"  [INFO] Found 2 time steps in ECCO data. Assuming idx 0=DJF, 1=JJA.")
                ds_djf = ds_ecco_proc.isel({t_dim: 0})
                ds_jja = ds_ecco_proc.isel({t_dim: 1})
                if t_dim in ds_djf.coords: ds_djf = ds_djf.drop_vars(t_dim)
                if t_dim in ds_jja.coords: ds_jja = ds_jja.drop_vars(t_dim)
                obs_data['MIT']['DJF'] = ds_djf
                obs_data['MIT']['JJA'] = ds_jja
                
            elif t_dim and ds_ecco_proc.sizes[t_dim] == 1:
                ds_ecco_proc = ds_ecco_proc.squeeze(dim=t_dim)
                if t_dim in ds_ecco_proc.coords: ds_ecco_proc = ds_ecco_proc.drop_vars(t_dim)
                obs_data['MIT']['DJF'] = ds_ecco_proc
                obs_data['MIT']['JJA'] = ds_ecco_proc
            else:
                 # No time dim or unknown size.
                 # Warn if size > 1
                 if t_dim and ds_ecco_proc.sizes[t_dim] > 1:
                     print(f"  [WARN] ECCO data has {ds_ecco_proc.sizes[t_dim]} time steps. Using Isel 0 for DJF and 1 for JJA.")
                     obs_data['MIT']['DJF'] = ds_ecco_proc.isel({t_dim: 0})
                     obs_data['MIT']['JJA'] = ds_ecco_proc.isel({t_dim: 1})
                 else:
                     obs_data['MIT']['DJF'] = ds_ecco_proc
                     obs_data['MIT']['JJA'] = ds_ecco_proc
        else:
             # Fallback
             obs_data['MIT']['DJF'] = ds_ecco
             obs_data['MIT']['JJA'] = ds_ecco
    return obs_data

# --- Mock Data ---

def generate_mock_data(base_test_dir):
    # Minimal mock for imports to work
    target_dir = os.path.join(base_test_dir, "target")
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    return target_dir

# --- Diagnostics (Adapted with GFO suffix) ---
    
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
                ds_r = ref_data['GEOS'][season].get(var_name)
                ds_o = obs_data['GEOS'][season].get(var_name)
                d_ref = ds_r
                d_obs = ds_o
            else:
                if season not in ref_data['MIT']: continue
                ds_r = ref_data['MIT'][season].get(var_name)
                ds_o = obs_data['MIT'][season].get(var_name)
                d_ref = ds_r
                d_obs = ds_o

            if d_ref is None or d_obs is None: continue

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
                if coll_type == 'GEOS':
                     ds_tmp = p_data['GEOS'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                elif coll_type == 'MIT':
                     ds_tmp = p_data['MIT'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                     
                if d_pert is not None:
                     if d_pert.shape != d_ref.shape:
                         d_pert = d_pert.interp_like(d_ref, method='nearest')
                         
                     c_exp = ((d_pert - d_obs_aligned)**2).sum().item() / (sigma**2)
                     exp_global_costs[exp_id] += c_exp
                else:
                    exp_global_costs[exp_id] += c_ref
            
            # 2. Optimized Cost (Projected)
            d_delta_sum = xr.zeros_like(d_ref)
            for p_data in pert_data_list:
                exp_id = p_data['ID']
                if exp_id not in weights_map: continue
                # In GFO result, we need the *applied weight* (eta) for projection
                # The weights_map passed here is eta
                w = weights_map[exp_id]
                
                d_pert = None
                if coll_type == 'GEOS':
                     ds_tmp = p_data['GEOS'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                elif coll_type == 'MIT':
                     ds_tmp = p_data['MIT'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                     
                if d_pert is None: d_pert = d_ref

                if d_pert.shape != d_ref.shape:
                    d_pert = d_pert.interp_like(d_ref, method='nearest')
                d_delta_sum += w * (d_pert - d_ref)
            
            d_proj = d_ref + d_delta_sum
            c_opt = ((d_proj - d_obs_aligned)**2).sum().item() / (sigma**2)
            
            total_opt_cost += c_opt
            var_costs[var_name]['Opt'] += c_opt

    # --- Plot 1: Global Cost Breakdown by Experiment ---
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
        
        plt.title("Cost Comparison (GFO): Reference, Optimized, and Individual Experiments", fontsize=14)
        plt.ylabel("Relative Cost (Ref=1)")
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", 
                     ha='center', va='bottom', fontsize=8, rotation=90)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Global_Cost_Breakdown_Experiments_GFO.png"))
        plt.close()
        
    # --- Plot 2: Global Cost Breakdown by Variable ---
    plot_vars = sorted(var_costs.keys())
    ref_vals = [] 
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
        plt.bar(x - width/2, ref_vals, width, label='Reference (1.0)', color='gray')
        rects2 = plt.bar(x + width/2, opt_vals, width, label='Optimized (GFO)', color='green')
        
        plt.ylabel('Relative Cost (Ref=1)')
        plt.title('Global Cost Breakdown by Variable (GFO)')
        plt.xticks(x, valid_plot_vars)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        for rect, label in zip(rects2, opt_labels):
            plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), label,
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Global_Cost_Breakdown_Variables_GFO.png"))
        plt.close()

    # Define targets to plot
    targets = []
    for season in SEASONS:
        for v in VAR_MAP_GEOS.values(): targets.append(('GEOS', v, season, f"{v}"))
        for v in VAR_MAP_MIT.values(): targets.append(('MIT', v, season, f"{v}"))
    
    for coll_type, var_name, season, label in targets:
        print(f"  Plotting {label} ({season})...")
        
        if coll_type == 'GEOS':
            ds_r = ref_data['GEOS'][season].get(var_name)
            ds_o = obs_data['GEOS'][season].get(var_name)
        else:
            ds_r = ref_data['MIT'][season].get(var_name)
            ds_o = obs_data['MIT'][season].get(var_name)
            
        if ds_r is None or ds_o is None: continue
        d_ref = ds_r
        d_obs = ds_o
            
        d_obs_aligned = d_obs.interp_like(d_ref, method='nearest')
        
        # Projected Field
        d_proj = d_ref.copy(deep=True)
        pert_deltas = xr.zeros_like(d_ref)
        
        for i, p_data in enumerate(pert_data_list):
            exp_id = p_data['ID']
            if exp_id not in weights_map: continue
            w = weights_map[exp_id]
            
            d_pert = None
            if coll_type == 'GEOS': 
                 ds_tmp = p_data['GEOS'].get(season)
                 if ds_tmp: d_pert = ds_tmp.get(var_name)
            else: 
                 ds_tmp = p_data['MIT'].get(season)
                 if ds_tmp: d_pert = ds_tmp.get(var_name)
                
            if d_pert is None: d_pert = d_ref
            if d_pert.shape != d_ref.shape: d_pert = d_pert.interp_like(d_ref, method='nearest')
            pert_deltas = pert_deltas + w * (d_pert - d_ref)
            
        d_proj = d_ref + pert_deltas
        
        sigma = d_obs_aligned.std()
        if sigma < 1e-12: sigma = 1.0
        
        cost_ref = ((d_ref - d_obs_aligned)**2) / (sigma**2)
        cost_proj = ((d_proj - d_obs_aligned)**2) / (sigma**2)
        
        diff_ref = d_ref - d_obs_aligned
        diff_proj = d_proj - d_obs_aligned
        
        vmin_f = min(d_ref.min().item(), d_obs_aligned.min().item(), d_proj.min().item())
        vmax_f = max(d_ref.max().item(), d_obs_aligned.max().item(), d_proj.max().item())
        max_diff = max(abs(diff_ref.min().item()), abs(diff_ref.max().item()),
                       abs(diff_proj.min().item()), abs(diff_proj.max().item()))
        vmin_d, vmax_d = -max_diff, max_diff
        
        # PLOT 1: Mean Field
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()
        im1 = axs[0].imshow(d_ref, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[0].set_title(f"Ref Cost: {((d_ref-d_obs_aligned)**2).mean().item():.3e}")
        plt.colorbar(im1, ax=axs[0])
        im2 = axs[1].imshow(d_obs_aligned, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[1].set_title("Target")
        plt.colorbar(im2, ax=axs[1])
        im3 = axs[2].imshow(d_proj, origin='lower', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axs[2].set_title(f"Proj Cost: {((d_proj-d_obs_aligned)**2).mean().item():.3e}")
        plt.colorbar(im3, ax=axs[2])
        im4 = axs[3].imshow(diff_ref, origin='lower', cmap='RdBu_r', vmin=vmin_d, vmax=vmax_d)
        axs[3].set_title("Ref - Target")
        plt.colorbar(im4, ax=axs[3])
        im5 = axs[4].imshow(diff_proj, origin='lower', cmap='RdBu_r', vmin=vmin_d, vmax=vmax_d)
        axs[4].set_title("Proj - Target")
        plt.colorbar(im5, ax=axs[4])
        axs[5].axis('off')
        fig.suptitle(f"{label} ({season}) - Mean Fields & Differences (GFO)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{var_name}_{season}_MeanFields_GFO.png"))
        plt.close()
        
        # PLOT 2: Cost
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        c_max = max(cost_ref.max(), cost_proj.max())
        im0 = axs[0].imshow(cost_ref, origin='lower', cmap='Reds', vmin=0, vmax=c_max)
        axs[0].set_title(f"Ref Cost")
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(cost_proj, origin='lower', cmap='Reds', vmin=0, vmax=c_max)
        axs[1].set_title(f"Proj Cost")
        plt.colorbar(im1, ax=axs[1])
        imp = cost_ref - cost_proj
        bound = max(abs(imp.min()), abs(imp.max()))
        im2 = axs[2].imshow(imp, origin='lower', cmap='RdBu_r', vmin=-bound, vmax=bound)
        axs[2].set_title(f"Improvement (Ref - Proj)")
        plt.colorbar(im2, ax=axs[2])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{var_name}_{season}_CostComparison_GFO.png"))
        plt.close()

    print("Diagnostics Generation Complete.\n")

# --- Optimized Parameter Output ---

def compute_field_cost(ref_data, pert_data_list, obs_data, weights_map):
    """
    Computes total cost using reconstructed fields (same method as diagnostics).
    Returns: (ref_cost, opt_cost)
    """
    total_ref_cost = 0.0
    total_opt_cost = 0.0
    
    for season in SEASONS:
        all_vars = list(VAR_MAP_GEOS.values()) + list(VAR_MAP_MIT.values())
        
        for var_name in all_vars:
            coll_type = 'GEOS' if var_name in VAR_MAP_GEOS.values() else 'MIT'
            
            # Get Ref/Obs
            if coll_type == 'GEOS':
                if season not in ref_data['GEOS']: continue
                d_ref = ref_data['GEOS'][season].get(var_name)
                d_obs = obs_data['GEOS'][season].get(var_name)
            else:
                if season not in ref_data['MIT']: continue
                d_ref = ref_data['MIT'][season].get(var_name)
                d_obs = obs_data['MIT'][season].get(var_name)

            if d_ref is None or d_obs is None: continue

            # Align
            d_obs_aligned = d_obs.interp_like(d_ref, method='nearest')
            
            # Sigma
            sigma = d_obs_aligned.std().values
            if sigma < 1e-12: sigma = 1.0
            
            # Ref Cost
            c_ref = ((d_ref - d_obs_aligned)**2).sum().item() / (sigma**2)
            total_ref_cost += c_ref
            
            # Optimized Cost (Projected)
            d_delta_sum = xr.zeros_like(d_ref)
            for p_data in pert_data_list:
                exp_id = p_data['ID']
                if exp_id not in weights_map: continue
                w = weights_map[exp_id]
                
                d_pert = None
                if coll_type == 'GEOS':
                     ds_tmp = p_data['GEOS'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                elif coll_type == 'MIT':
                     ds_tmp = p_data['MIT'].get(season)
                     if ds_tmp: d_pert = ds_tmp.get(var_name)
                     
                if d_pert is None: d_pert = d_ref

                if d_pert.shape != d_ref.shape:
                    d_pert = d_pert.interp_like(d_ref, method='nearest')
                d_delta_sum += w * (d_pert - d_ref)
            
            d_proj = d_ref + d_delta_sum
            c_opt = ((d_proj - d_obs_aligned)**2).sum().item() / (sigma**2)
            total_opt_cost += c_opt
            
    return total_ref_cost, total_opt_cost


def optimize_with_gfo(ref_data, pert_data_list, obs_data, param_df, weights_dict=None):
    print("\n--- Organizing Data for GFO Algorithm ---")
    
    # 1. Prepare P0 (Original Parameters) and P (Perturbed Matrix)
    # p0 = [val1, val2, ...]
    # p[i] = [val1, ..., val_i_pert, ...]
    
    p0 = param_df['Original'].values
    n_params = len(p0)
    
    p_matrix = [] # List of tuples/arrays
    
    # Map ExpID to its Index in the p_matrix list
    # Because pert_data_list might be unsorted or partial.
    # We must iterate through the loaded experiments in pert_data_list.
    
    exps_indices = [] # Indices of parameters that correspond to loaded experiments
    
    for p_data in pert_data_list:
        exp_id = p_data['ID']
        
        # Find which parameter corresponds to this experiment
        row = param_df[param_df['Exp'] == exp_id]
        if row.empty:
            print(f"Warn: Exp {exp_id} not in parameter definitions.")
            continue
            
        idx = row.index[0] # Index in DataFrame i.e., parameter index
        exps_indices.append(idx)
        
        # Construct p vector for this experiment
        p_vec = p0.copy()
        p_vec[idx] = row['Perturbed'].item()
        p_matrix.append(p_vec)
        
    p = tuple(p_matrix)
    
    print(f"  Initialized {len(p)} experiments for GFO.")
    if len(p) == 0: return

    # 2. Prepare Data Arrays (Y, X0, X, W)
    # They enter GFO as TUPLES of ARRAYS (one array per field).
    # Fields = (Season x Variable)
    
    y_list = []
    x0_list = []
    w_list = []
    
    # x_list is list of lists: x[experiment_index][field_index]
    x_matrix = [[] for _ in range(len(p))]
    
    accepted_fields = 0
    
    # Flatten Data Logic
    for season in SEASONS:
        # GEOS
        vars_geos = VAR_MAP_GEOS.values()
        for v_name in vars_geos:
            # Check availability in Ref and Obs
            if (season in ref_data['GEOS']) and (season in obs_data['GEOS']):
                d_ref = ref_data['GEOS'][season].get(v_name)
                d_obs = obs_data['GEOS'][season].get(v_name)
                
                if d_ref is not None and d_obs is not None:
                    # Align Obs to Ref
                    d_obs = d_obs.interp_like(d_ref, method='nearest')
                    
                    # Weights (Importance)
                    w_imp = 1.0
                    if weights_dict and v_name in weights_dict:
                        w_imp = weights_dict[v_name]
                        
                    # Flatten
                    arr_obs = d_obs.values.flatten()
                    arr_ref = d_ref.values.flatten()
                    arr_w   = np.full_like(arr_obs, w_imp)
                    
                    # Store Ref/Obs/Weight
                    y_list.append(arr_obs)
                    x0_list.append(arr_ref)
                    w_list.append(arr_w)
                    
                    # Extract for Perturbed Experiments
                    for i, p_data in enumerate(pert_data_list):
                        exp_id = p_data['ID']
                        # Check exist
                        ds_pert = p_data['GEOS'].get(season)
                        if ds_pert is not None and v_name in ds_pert:
                            d_pert = ds_pert[v_name]
                        else:
                            # If missing, assume equal to Ref (no sensitivity)
                            d_pert = d_ref
                            
                        # Align
                        if d_pert.shape != d_ref.shape:
                            d_pert = d_pert.interp_like(d_ref, method='nearest')
                            
                        x_matrix[i].append(d_pert.values.flatten())
                        
                    accepted_fields += 1

        # MIT
        vars_mit = VAR_MAP_MIT.values()
        for v_name in vars_mit:
             if (season in ref_data['MIT']) and (season in obs_data['MIT']):
                d_ref = ref_data['MIT'][season].get(v_name)
                d_obs = obs_data['MIT'][season].get(v_name)
                
                if d_ref is not None and d_obs is not None:
                    d_obs = d_obs.interp_like(d_ref, method='nearest')
                    w_imp = 1.0
                    if weights_dict and v_name in weights_dict:
                        w_imp = weights_dict[v_name]
                        
                    y_list.append(d_obs.values.flatten())
                    x0_list.append(d_ref.values.flatten())
                    w_list.append(np.full(d_ref.size, w_imp))
                    
                    for i, p_data in enumerate(pert_data_list):
                        ds_pert = p_data['MIT'].get(season)
                        if ds_pert is not None and v_name in ds_pert:
                            d_pert = ds_pert[v_name]
                        else:
                            d_pert = d_ref
                        if d_pert.shape != d_ref.shape:
                            d_pert = d_pert.interp_like(d_ref, method='nearest')
                        x_matrix[i].append(d_pert.values.flatten())
                    accepted_fields += 1

    print(f"  Organized {accepted_fields} data fields.")
    
    # Convert lists to Tuples/Arrays for GFO
    y_tuple = tuple(y_list)
    x0_tuple = tuple(x0_list)
    w_tuple = tuple(w_list)
    x_tuple = tuple( [tuple(row) for row in x_matrix] )
    
    # sanitize Mask Nans?
    # GFO class has 'stats' which uses nanmean.
    # 'estimate' uses sum(w*(y-dist)**2). If nan present, result is nan?
    # GFO line 261: ind = ((~np.isnan(Yd[:,0])) & (~np.isnan(R)))
    # It handles NaNs internally by masking.
    
    # 3. Running GFO
    print("\n>>> INITIALIZING GFO SOLVER <<<")
    solver = GFO(y_tuple, x0_tuple, p0, x_tuple, p, w_tuple)
    
    print(">>> ESTIMATING OPTIMAL PARAMETERS <<<")
    res = solver.estimate()
    
    if res is None:
        print("Optimization Failed.")
        return
    
    # Unpack new result structure
    gfo_results = res['results']
    diff_matrix = res['diff_matrix']
    old_params = res['old_params']
    
    # 4. Compute Reference Cost (Field-Based)
    # Use empty weight map to get ref cost
    empty_weights = {p_data['ID']: 0.0 for p_data in pert_data_list}
    ref_cost_field, _ = compute_field_cost(ref_data, pert_data_list, obs_data, empty_weights)
    
    print(f"\n{'='*80}")
    print(f"INITIAL COST (Field-Based): {ref_cost_field:.3f}")
    print(f"{'='*80}")
    
    all_results_for_csv = []
    
    for Q_val in [0, 0.5, 1.0]:
        r = gfo_results.get(Q_val)
        if r is None:
            print(f"\n[Q = {Q_val}] SKIPPED (Matrix Singular)")
            continue
            
        eta_weights = r['eta']
        p_opt = r['params']
        
        # Build weight map for this Q's eta values
        weight_map_q = {}
        for idx, p_idx in enumerate(exps_indices):
            p_row = param_df.iloc[p_idx]
            exp_id = p_row['Exp']
            weight_map_q[exp_id] = eta_weights[idx]
        
        # Compute field-based cost
        _, proj_cost_field = compute_field_cost(ref_data, pert_data_list, obs_data, weight_map_q)
        improvement_field = (1 - proj_cost_field / ref_cost_field) * 100 if ref_cost_field > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR Q = {Q_val} (Prior Uncertainty)")
        print(f"{'='*80}")
        print(f"Projected Cost: {proj_cost_field:.3f}  |  Improvement: {improvement_field:.2f}%")
        print("-"*80)
        
        # Build results list with sensitivity
        exp_results = []
        for idx, p_idx in enumerate(exps_indices):
            p_row = param_df.iloc[p_idx]
            exp_id = p_row['Exp']
            orig = p_row['Original']
            opt_val = p_opt[p_idx]
            eta_val = eta_weights[idx]
            change_pct = (opt_val - orig)/orig * 100 if orig != 0 else 0
            sensitivity = abs(eta_val)
            
            exp_results.append({
                'Exp': exp_id,
                'Parameter': p_row['Parameter'],
                'Original': orig,
                'Optimized': opt_val,
                'ChangePct': change_pct,
                'Eta': eta_val,
                'Sensitivity': sensitivity,
                'Q': Q_val
            })
        
        # Sort by Sensitivity (descending)
        exp_results.sort(key=lambda x: x['Sensitivity'], reverse=True)
        
        # Print Table Header
        print(f"{'Rank':<5} {'Exp':<13} {'Parameter':<18} {'Original':<12} {'Optimized':<12} {'Change%':<10} {'Sensitivity':<10}")
        print("-"*80)
        
        for rank, row in enumerate(exp_results, 1):
            print(f"{rank:<5} {row['Exp']:<13} {row['Parameter']:<18} {row['Original']:<12.4g} {row['Optimized']:<12.4g} {row['ChangePct']:+10.2f}% {row['Sensitivity']:<10.4f}")
        
        all_results_for_csv.extend(exp_results)
        
    # Save to CSV
    pd.DataFrame(all_results_for_csv).to_csv("optimization_results_GFO_Algo.csv", index=False)
    print(f"\n{'='*80}")
    print("Saved all results to optimization_results_GFO_Algo.csv")
    
    # Generate Diagnostics using Q=0 (unconstrained) results for visualization
    # If Q=0 failed, fallback to Q=0.5, then Q=1
    diag_q = 0 if gfo_results.get(0) else (0.5 if gfo_results.get(0.5) else 1.0)
    diag_result = gfo_results.get(diag_q)
    
    if diag_result:
        eta_for_diag = diag_result['eta']
        weight_map = {}
        for idx, p_idx in enumerate(exps_indices):
            p_row = param_df.iloc[p_idx]
            exp_id = p_row['Exp']
            weight_map[exp_id] = eta_for_diag[idx]
            
        print(f"\nGenerating Diagnostics using Q={diag_q} results...")
        generate_diagnostics(ref_data, pert_data_list, obs_data, weight_map, param_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--root_dir', type=str, default="/nobackupp27/afahad/GFEXPv11.7")
    args = parser.parse_args()
    
    ROOT_DIR = args.root_dir
    TARGET_DIR = TARGET_DIR_DEFAULT
    
    if args.test:
        ROOT_DIR = "MOCK_DATA/GFEXP"
        TARGET_DIR = generate_mock_data(ROOT_DIR)
        
    param_df = get_experiment_parameters()
    # Mock data loading step similar to original
    # For brevity, assuming data loading works as in original script
    # ... (Actual loading code would be here, effectively duplicated from greens_func_tuning.py)
    
    # Re-implement Loading Loop just to make script functional
    obs_data = load_targets(TARGET_DIR)
    ref_geos = {}
    ref_mit = {}
    for s in SEASONS:
        ref_geos[s] = load_geos_data(ROOT_DIR, "GF01_ref", s)
        ref_mit[s] = load_mitgcm_data(ROOT_DIR, "GF01_ref", s)
    ref_container = {'GEOS': ref_geos, 'MIT': ref_mit}
    
    pert_container_list = []
    for exp_id in param_df['Exp'].unique():
        print(f"Loading {exp_id}...")
        p_geos = {s: load_geos_data(ROOT_DIR, exp_id, s) for s in SEASONS}
        p_mit = {s: load_mitgcm_data(ROOT_DIR, exp_id, s) for s in SEASONS}
        pert_container_list.append({'ID': exp_id, 'GEOS': p_geos, 'MIT': p_mit})
        
    weights = None
    if args.weighted:
        weights = {
            'FRSEAICE': 1.0, 'H500': 0, 'LWGNET': 1.0, 'SWGNET': 1.0, 
            'SALT_300m': 0.8, 'THETA_300m': 0.8, 'THETA_0m': 4, 
            'SALT_0m': 0.8, 'T2M': 1.2, 'TS': 0, 'T300': 2, 'U200': 4
        }
        
    optimize_with_gfo(ref_container, pert_container_list, obs_data, param_df, weights)

if __name__ == "__main__":
    main()
