# Green's Function Parameter Tuning (GFO) — Developer Handoff Guide

This document is a concise, operational guide for taking over and implementing the Green's Function parameter tuning framework for the next version of the NASA GEOS model.

---

## 1. Storage & Experiment Locations (Cluster Paths)

### Root Workspace & Simulations
- **Experiment Base Directory**: `/nobackupp27/afahad/GFEXPv11.7`
- **Control / Unperturbed Run**: `/nobackupp27/afahad/GFEXPv11.7/GF01_ref`
- **Single-Parameter Perturbation Runs**: `/nobackupp27/afahad/GFEXPv11.7/GF02_TURN` through `GF22_CQF`
- **Tuned Forward Validation Runs**:
  - `GF_OPT`: First tuned verification simulation
  - `GF_OPT72`: 72-level configuration test
  - `GF_OPTUpd`: Updated parameter validation simulation

### Climatology File Structure Inside Each Experiment
Each experiment directory contains a `clim/` subfolder with pre-extracted seasonal means:
- **Atmosphere (GEOS)**:  
  `{ROOT_DIR}/{EXP_ID}/clim/geos_clim/{EXP_ID}.mean.{SEASON}.nc`
- **Ocean (MITgcm)**:  
  `{ROOT_DIR}/{EXP_ID}/clim/mitgcm.{VAR}.{DEPTH}.{SEASON}.*.nc`  
  *(e.g., `mitgcm.THETA.0m.DJF.2008-2017.nc`)*

### Observational Targets
- **Target Directory**: `/nobackupp27/afahad/GFEXPv11.7/target`
  - `MERRA2.seasonal.mean.DJF.nc` / `MERRA2.seasonal.mean.JJA.nc` (Atmosphere & radiation)
  - `ECCO_L4_Climatology_2008-2017.nc` (Ocean temperature & salinity)
  - `m2/`: Raw monthly MERRA-2 slices for time series validation

### Model Source & Setup
- Model build & run setup: `/nobackupp27/afahad/GFEXPv11.7/readme.txt`
- Binary setup script: `/nobackupp27/afahad/GFEXPv11.7/GEOSgcm/install/bin/gcm_setup`
- Initial conditions & MIT input: `GEOS_IC/` and `mit_input/`
- Custom source code modifications (if adding non-namelist parameters):
  - Catchment model (`CQfactor`): `GEOSgcm/src/Components/@GEOSgcm_GridComp/GEOSagcm_GridComp/GEOSphysics_GridComp/GEOSsurface_GridComp/GEOSland_GridComp/GEOScatch_GridComp/`
  - Surface layer (`VSLfactor`, `Charnok1`, `Charnok2`): `GEOSgcm/src/Shared/@GMAO_Shared/GEOS_Shared/surfacelayer.F90`

---

## 2. Codebase Organization

| File / Directory | Purpose |
| :--- | :--- |
| `scripts/greens_func_tuning.py` | **Primary tuning engine**: Loads fields, normalizes, builds sensitivity matrix $G$, and solves via Least Squares & Ridge regression ($Aw=b$). |
| `scripts/greens_func_tuning_GFO.py` | **Menemenlis et al. (2005) GFO solver**: Implements formal covariance weighting $R$ and prior uncertainty $Q \in \{0, 0.5, 1.0\}$. |
| `scripts/post.py` | **Validation post-processing**: Compares tuned runs (`GF_OPT`, `GF_OPTUpd`) against baseline `GF01_ref` and MERRA-2; generates spatial bias & diff maps. |
| `scripts/timeseries.py` | Extracts and plots global mean $T_S$, SST, and Nino 3.4 SST anomaly time series. |
| `processing/extract_clim.py` | **Climatology extractor**: Reads raw monthly GEOS collections (`geosgcm_surf`, `geosgcm_prog`), extracts pressure levels, averages 10-year DJF/JJA seasonal means, and saves to `clim/geos_clim/`. |
| `processing/process_merra2.py` | **MERRA-2 target processor**: Extracts $T_S$, $T_{200}$, $H_{500}$ from raw MERRA-2 2D/3D monthly files and builds target NetCDF files. |
| `processing/get_target_era5.py` | **ERA5 cloud retriever**: Streams and processes ERA5 reanalysis fields directly from Google Cloud ARCO-ERA5 Zarr store. |
| `processing/find_missing.py` | **File auditor**: Checks for missing months/years across all experiment directories before running climatology extraction. |
| `processing/check_longnames.py` & `check_vars_v2.py` | NetCDF variable name and coordinate verification utilities. |
| `processing/analyze_warming_pattern.py` | Computes and plots global warming patterns between reference and perturbed runs. |
| `processing/compare_GF01_GF22_TS.py` | Direct difference analysis between reference and specific experiment runs. |
| `processing/run_extract.sh` & `run_direct.sh` | SLURM batch execution scripts for climatology extraction on the cluster. |
| `processing/merra2_download_urls.txt` | Complete list of NASA GES DISC HTTP download links for MERRA-2 2D/3D collections. |
| `utils/create_html_viewer.py` | Scans `plots/` and rebuilds the portable `plots_viewer.html` gallery. |
| `results/` | Output CSV files with ranked parameter shifts (`optimization_results.csv`, `optimization_results_GFO_Algo.csv`). |
| `plots/` | All 156 generated diagnostic maps, cost comparisons, and validation plots. |

---

## 3. How to Add New Experiments (Parameters)

To introduce new model parameters in the next GEOS version:

1. **Run the Perturbed Simulation**:
   Create a new experiment directory (e.g., `GF23_NEWPARAM`) perturbing only that parameter. Run the model and generate seasonal climatologies in `clim/geos_clim/` and `clim/mitgcm.*`.

2. **Register in the Parameter Dictionary**:
   In `scripts/greens_func_tuning.py` and `scripts/greens_func_tuning_GFO.py`, edit the `get_experiment_parameters()` function:
   ```python
   def get_experiment_parameters():
       data = [
           # ... existing experiments ...
           {'Exp': 'GF23_NEWPARAM', 'Parameter': 'PARAM_NAME', 'Original': 1.0, 'Perturbed': 1.5},
       ]
       df = pd.DataFrame(data)
       df['Delta'] = df['Perturbed'] - df['Original']
       return df
   ```

3. **Execution**:
   The scripts automatically discover any experiment ID listed in this dataframe and build corresponding columns in the sensitivity matrix $G$.

---

## 4. How to Add New Target Variables

### A. Atmospheric / Surface Variables (GEOS)
1. Add to variable list and mapping in `scripts/greens_func_tuning.py` (around lines 17–33):
   ```python
   GEOS_VARS = ['FRSEAICE', 'T2M', 'T_300', 'U_200', 'H_500', 'LWGNET', 'SWGNET', 'NEW_VAR']

   VAR_MAP_GEOS = {
       # ...
       'NEW_VAR': 'NEW_VAR_NAME_IN_NETCDF'
   }
   ```
2. Ensure the observational target files (`MERRA2.seasonal.mean.{SEASON}.nc`) contain `NEW_VAR_NAME_IN_NETCDF`.

### B. Ocean Variables (MITgcm)
1. Add key to `MIT_VARS` list:
   ```python
   MIT_VARS = ['SALT_0m', 'SALT_300m', 'THETA_0m', 'THETA_300m', 'U_0m']
   ```
2. Update the filename mapping in `load_mitgcm_data()`:
   ```python
   file_map = {
       'SALT0': ('SALT', '0m'),
       # ...
       'U0': ('U', '0m')  # Expects filename pattern: mitgcm.U.0m.{season}*.nc
   }
   ```
3. Update `load_targets()` to extract the corresponding field from `ECCO_L4_Climatology_2008-2017.nc` (or target ocean dataset).

---

## 5. How Seasons Are Handled

Seasons are defined globally at the top of the tuning scripts:
```python
SEASONS = ['DJF', 'JJA']
```
- **To add more seasons** (e.g., transitional `MAM`, `SON`):
  1. Change to `SEASONS = ['DJF', 'MAM', 'JJA', 'SON']`.
  2. Ensure each experiment has corresponding `{EXP_ID}.mean.MAM.nc` and target observations have `MAM` slices.
- **To use individual monthly averages** (e.g., July / Dec):
  Change the list and point `load_geos_data()` / `load_targets()` to the monthly file naming convention.

---

## 6. How Normalization & Weights Are Handled

### Spatial Normalization (Automatic)
Because variables have different units (K, m, m/s, W/m²), each variable field is normalized by the spatial standard deviation ($\sigma_{\text{obs}}$) of the observation:
$$\tilde{y} = \frac{y}{\sigma_{\text{obs}}}, \quad \tilde{x} = \frac{x}{\sigma_{\text{obs}}}$$
This prevents large-magnitude variables (like geopotential height ~5000 m) from dominating small-magnitude ones (like sea ice fraction 0–1).

### Custom User Weights
In `scripts/greens_func_tuning.py`, enable `--weighted` and define manual scale factors:
```python
weights = {
    'T2M': 1.0,
    'U200': 1.0,
    'SWGNET': 0.5,
    'FRSEAICE': 0.2
}
```
The solver scales each variable's block by $\sqrt{W_v}$, so Cost $J = \sum_v W_v \frac{(x_v - y_v)^2}{\sigma_v^2}$.

### Inversion Regularization & Prior Uncertainty
- **Least Squares (`greens_func_tuning.py`)**:
  Applies Ridge parameter $\alpha \in [0, 0.1, 0.3]$. $\alpha$ is scaled by $N_{\text{obs}}$ to ensure grid-resolution independence:
  $$(G^T G + \alpha N_{\text{obs}} I) w = G^T d$$
- **Menemenlis GFO (`greens_func_tuning_GFO.py`)**:
  Uses prior covariance $Q$:
  $$P = \left(\text{diag}(1/Q^2) + G^T R^{-1} G\right)^{-1}$$
  - $Q = 0$: Unconstrained least-squares.
  - $Q = 1.0$: Restricts parameter shifts to $\approx 1$ perturbation magnitude (physically safe).
  - $Q = 0.5$: Conservative tuning.

---

## 7. How to Run & Verify

### Step 1: Mock Test Run (Fast Verification Without Full Data)
Verify pipeline integrity and plotting on any machine without needing model output:
```bash
python scripts/greens_func_tuning.py --test
python scripts/greens_func_tuning_GFO.py --test
```

### Step 2: Full Optimization on the Cluster
Run on the server with simulation directories mounted:
```bash
python scripts/greens_func_tuning.py --root_dir /nobackupp27/afahad/GFEXPv11.7
python scripts/greens_func_tuning_GFO.py --root_dir /nobackupp27/afahad/GFEXPv11.7
```
Outputs are written to:
- `results/optimization_results.csv`
- `results/optimization_results_GFO_Algo.csv`
- `plots/`

### Step 3: Forward Validation Post-Processing
After running a new forward simulation (e.g. `GF_OPT_v2`) using the optimal parameters:
1. Update `EXP_OPT = 'GF_OPT_v2'` in `scripts/post.py`.
2. Run validation:
   ```bash
   python scripts/post.py
   python scripts/timeseries.py
   ```
3. Update interactive plot viewer:
   ```bash
   python utils/create_html_viewer.py
   ```
   Open `plots_viewer.html` in a browser to inspect all results.
