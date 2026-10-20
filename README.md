# Green's Function Parameter Tuning for GEOS-MIT Coupled Climate Model

This repository contains the complete codebase, optimization algorithms, parameter sensitivity tables, and diagnostic plots for multi-objective **Green's Function Optimization (GFO)** applied to the **coupled NASA GEOS (Atmosphere) and MITgcm (Ocean)** general circulation model.

---

## Table of Contents
1. [Overview & Methodology](#overview--methodology)
2. [Target Variables & Observational Datasets](#target-variables--observational-datasets)
3. [Model Experiments & Parameter Definitions](#model-experiments--parameter-definitions)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Optimization Results & Parameter Ranking](#optimization-results--parameter-ranking)
6. [Results Walkthrough with Diagnostic Figures](#results-walkthrough-with-diagnostic-figures)
   - [Cost Function Reductions](#1-global-cost-function-reductions)
   - [Validation Simulations (GF_OPT vs Baseline)](#2-coupled-model-validation-gf01_ref-vs-gf_opt)
   - [Spatial Bias & Difference Maps](#3-spatial-bias-and-difference-maps)
   - [Atmosphere & Ocean Seasonal Diagnostics](#4-atmosphere--ocean-seasonal-diagnostics)
7. [Repository Structure](#repository-structure)
8. [Setup & Execution Guide (Handoff Instructions)](#setup--execution-guide-handoff-instructions)

---

## Overview & Methodology

Atmospheric and ocean climate models contain physical parameterizations (convection, cloud microphysics, turbulence, surface exchange, radiation) with empirical parameters that directly govern model climate fidelity. Standard manual tuning is computationally prohibitive in coupled systems due to non-linear feedbacks and high dimensionality.

Here, we implement the **Green's Function approach** (Menemenlis et al., 2005) to objectively calibrate 21 physical parameters simultaneously:
- **Baseline Control Experiment**: `GF01_ref` run using the coupled GEOS-MIT model.
- **Sensitivity Ensemble**: 21 single-parameter perturbation runs (`GF02_TURN` through `GF22_CQF`), perturbing one parameter at a time.
- **Climatological Targets**: Multi-year seasonal climatologies (**DJF** and **JJA**) combining satellite reanalyses (**MERRA-2**, **CERES**), in-situ datasets (**HadCRUT4**), and ocean state estimates (**ECCO L4**).
- **Inverse Formulation**: A linear response matrix (Green's Function matrix $G$) is built from the perturbation ensemble and inverted against observational misfits to find the optimal parameter set $\mathbf{p}_{\text{opt}}$.
- **Forward Model Validation**: Optimal parameter values were deployed in forward verification runs (**`GF_OPT`**, **`GF_OPT72`**, and **`GF_OPTUpd`**).

---

## Target Variables & Observational Datasets

The multi-objective cost function constrains both the atmosphere/surface state and the upper/subsurface ocean state:

| Variable | Description | Observational Dataset | Period / Climatology | Target Variable Name | Model Source Collection |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Sea ice fraction** | Polar sea ice cover | **MERRA-2** | 2009–2018 (DJF, JJA) | `FRSEAICE` | `geosgcm_surf` (`FRSEAICE`) |
| **Net surface shortwave radiation** | Downward net solar flux at surface | **CERES** / MERRA-2 | 2009–2018 (DJF, JJA) | `Surface_net_downward_shortwave_flux` (`SWGNET`) | `geosgcm_surf` (`SWGNET`) |
| **Net surface longwave radiation** | Absorbed longwave flux at surface | **CERES** / MERRA-2 | 2009–2018 (DJF, JJA) | `Surface_absorbed_longwave_radiation` (`LWGNET` / `SFCEM-LWS`) | `geosgcm_surf` (`LWGNET`) |
| **Upper-air Dynamics & Temp** | Zonal wind at 200 hPa, Temp at 300 hPa, Height at 500 hPa | **MERRA-2** | 2009–2018 (DJF, JJA) | `U200`, `T300`, `H500` | `geosgcm_prog` (`U`, `T`, `H`) |
| **2m Surface Air Temperature** | Near-surface land/ocean temperature | **HadCRUT4** / MERRA-2 | 2009–2018 (DJF, JJA) | `T2M` | `geosgcm_surf` (`T2M`) |
| **Sea Surface Temperature (SST)** | Surface ocean thermal state | **ECCO L4** | 2009–2018 (DJF, JJA) | `THETA` (0 m) | `mitgcm.THETA.0m` |
| **Sea Surface Salinity (SSS)** | Surface ocean salinity | **ECCO L4** | 2009–2018 (DJF, JJA) | `SALT` (0 m) | `mitgcm.SALT.0m` |
| **Subsurface Temperature (300m)**| Thermocline thermal structure | **ECCO L4** | 2009–2018 (DJF, JJA) | `THETA` (300 m) | `mitgcm.THETA.300m` |
| **Subsurface Salinity (300m)** | Thermocline halocline structure | **ECCO L4** | 2009–2018 (DJF, JJA) | `SALT` (300 m) | `mitgcm.SALT.300m` |

---

## Model Experiments & Parameter Definitions

The 21 sensitivity runs each perturbed one physical parameter from its default reference value:

| Exp ID | Parameter Name | Physical Description | Reference ($p_{\text{ref}}$) | Perturbed ($p_{\text{pert}}$) | Delta ($\Delta p$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **GF02_TURN** | `TURNRHCRIT` | Level where total moisture PDF switches from free atmosphere to PBL | 726.0 | 850.0 | +124.0 |
| **GF03_ENTR** | `ENTR_DP` | Deep convection entrainment rate | $1.0 \times 10^{-4}$ | $4.0 \times 10^{-4}$ | $+3.0 \times 10^{-4}$ |
| **GF04_WBF** | `WBFFACTOR` | Wegener-Bergeron-Findeisen mixed-phase cloud process scaling | 0.091 | 0.300 | +0.209 |
| **GF05_DCS** | `DCS` | Critical ice particle diameter for autoconversion (m) | $2.91 \times 10^{-4}$ | $2.50 \times 10^{-4}$ | $-4.1 \times 10^{-5}$ |
| **GF06_DUST** | `FDROP_DUST` | Cloud droplet activation fraction on mineral dust | 0.50 | 0.10 | -0.40 |
| **GF07_DCNV** | `DROPSZCNV` | Convective cloud droplet size | $2.5 \times 10^{-5}$ | $5.0 \times 10^{-5}$ | $+2.5 \times 10^{-5}$ |
| **GF08_FSOOT**| `FDROP_SOOT` | Cloud droplet activation fraction on soot aerosol | 0.05 | 0.09 | +0.04 |
| **GF09_MINCN**| `MINCDNC` | Minimum nucleated droplet concentration ($\text{cm}^{-3}$) | 52.0 | 25.0 | -27.0 |
| **GF10_TBRAD**| `TURB_BETA_RAD` | Cloud-top entrainment rate | 0.20 | 0.30 | +0.10 |
| **GF11_TRBKH**| `TURB_KHRADFAC` | Cloud-top radiative cooling turbulence scale factor | 0.85 | 0.50 | -0.35 |
| **GF12_MINRH**| `MINRHCRIT` | Free troposphere minimum critical relative humidity | 0.89 | 0.80 | -0.09 |
| **GF13_QRC** | `QRC_CRIT` | Critical relative humidity threshold for cloud/rain formation | $2.0 \times 10^{-4}$ | $4.0 \times 10^{-4}$ | $+2.0 \times 10^{-4}$ |
| **GF14_RKM** | `RKM` | Shallow convection entrainment rate | 8.0 | 12.0 | +4.0 |
| **GF15_AUT** | `AUT_SCALE` | Liquid cloud autoconversion rate scaling | 1.70 | 0.50 | -1.20 |
| **GF16_UIS** | `UISCALE` | Scaling for ice crystal terminal fall velocity | 1.00 | 0.75 | -0.25 |
| **GF17_CAHMAX**| `CICE_AH_MAX` | Sea ice albedo calculation threshold parameter | 0.30 | 0.20 | -0.10 |
| **GF18_ALBICEV**| `ALBICEV` | Visible band sea ice albedo | 0.78 | 0.73 | -0.05 |
| **GF19_BMDL** | `VSLfactor` | Scaling of ocean laminar viscous sublayer impact | 1.00 | 0.80 | -0.20 |
| **GF20_CH1** | `Charnok1` | Charnock ocean surface wind stress factor at high wind speed | $2.0303 \times 10^{-6}$ | $2.8425 \times 10^{-6}$ | $+8.12 \times 10^{-7}$ |
| **GF21_CH2** | `Charnok2` | Charnock ocean surface wind stress factor at low wind speed | $2.150 \times 10^{-3}$ | $3.150 \times 10^{-3}$ | $+1.00 \times 10^{-3}$ |
| **GF22_CQF** | `CQfactor` | Surface moisture exchange coefficient factor | 1.00 | 0.50 | -0.50 |

---

## Mathematical Formulation

Assuming small perturbations, the model response $\mathbf{y}(\mathbf{p})$ around the reference parameter vector $\mathbf{p}_0$ is linearized:
$$\mathbf{y}(\mathbf{p}) \approx \mathbf{y}(\mathbf{p}_0) + \mathbf{G} \, \Delta \mathbf{p}$$

where $\mathbf{G}$ is the Green's Function (sensitivity) matrix.

### Normalized Sensitivity Form
For numerical stability across disparate parameter dimensions, parameter updates are scaled by the perturbation step:
$$\lambda_i = \frac{p_i - p_{0, i}}{p_{\text{pert}, i} - p_{0, i}} \quad \implies \quad p_i = p_{0, i} + \lambda_i (p_{\text{pert}, i} - p_{0, i})$$

The columns of $\mathbf{G}$ are the differential model responses:
$$\mathbf{G}_{:, i} = \mathbf{x}_i - \mathbf{x}_0$$
where $\mathbf{x}_0$ is the reference state vector and $\mathbf{x}_i$ is the state vector from experiment $i$.

### Objective Function & Solvers
We implement two inversion formulations in this repository:

#### 1. Menemenlis et al. (2005) GFO Formulation ([`greens_func_tuning_GFO.py`](scripts/greens_func_tuning_GFO.py))
Minimizes parameter adjustment and model-data discrepancy with prior parameter uncertainty covariance $\mathbf{Q}$:
$$\eta = \left(\mathbf{Q}_d^{-1} + \mathbf{G}^T \mathbf{R}^{-1} \mathbf{G}\right)^{-1} \mathbf{G}^T \mathbf{R}^{-1} (\mathbf{y}_{\text{obs}} - \mathbf{x}_0)$$

where:
- $\mathbf{R}$ is the diagonal observational error covariance: $R_j = \sum w_j (y_j - x_{0, j})^2$.
- $\mathbf{Q}_d = \text{diag}(1/Q^2)$ imposes bounds on parameter changes. We evaluate $Q \in \{0, 0.5, 1.0\}$ ($Q=0$ is unconstrained).

#### 2. Regularized Least Squares / Ridge Regression ([`greens_func_tuning.py`](scripts/greens_func_tuning.py))
Solves:
$$\min_{\mathbf{w}} \|\mathbf{A} \mathbf{w} - \mathbf{b}\|^2 + \alpha \cdot N_{\text{obs}} \|\mathbf{w}\|^2$$
where $\mathbf{A} = \mathbf{R}^{-1/2} \mathbf{G}$ and $\mathbf{b} = \mathbf{R}^{-1/2} (\mathbf{y}_{\text{obs}} - \mathbf{x}_0)$, with $\alpha \in [0, 0.1, 0.3]$.

---

## Optimization Results & Parameter Ranking

### Least Squares Inversion Results ([`results/optimization_results.csv`](results/optimization_results.csv))

Parameters ranked by absolute relative change from baseline:

| Rank | Parameter | Exp ID | Baseline ($p_{\text{orig}}$) | Perturbed ($p_{\text{pert}}$) | Optimized ($p_{\text{opt}}$) | Weight ($w$) | Change (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`ENTR_DP`** | GF03_ENTR | $1.00 \times 10^{-4}$ | $4.00 \times 10^{-4}$ | **$3.78 \times 10^{-4}$** | +0.925 | **+277.5%** |
| **2** | **`WBFFACTOR`**| GF04_WBF | 0.091 | 0.300 | **0.232** | +0.675 | **+154.9%** |
| **3** | **`RKM`** | GF14_RKM | 8.00 | 12.00 | **15.28** | +1.819 | **+91.0%** |
| **4** | **`AUT_SCALE`**| GF15_AUT | 1.70 | 0.50 | **0.240** | +1.216 | **-85.9%** |
| **5** | **`QRC_CRIT`** | GF13_QRC | $2.00 \times 10^{-4}$ | $4.00 \times 10^{-4}$ | **$4.67 \times 10^{-5}$** | -0.767 | **-76.7%** |
| **6** | **`MINCDNC`** | GF09_MINCN | 52.0 | 25.0 | **66.03** | -0.520 | **+27.0%** |
| **7** | **`DROPSZCNV`**| GF07_DCNV | $2.50 \times 10^{-5}$ | $5.00 \times 10^{-5}$ | **$1.89 \times 10^{-5}$** | -0.243 | **-24.3%** |
| **8** | **`UISCALE`** | GF16_UIS | 1.00 | 0.75 | **1.23** | -0.914 | **+22.9%** |
| **9** | **`VSLfactor`**| GF19_BMDL | 1.00 | 0.80 | **1.19** | -0.928 | **+18.6%** |
| **10**| **`FDROP_DUST`**| GF06_DUST | 0.50 | 0.10 | **0.59** | -0.222 | **+17.7%** |
| **11**| **`Charnok1`** | GF20_CH1 | $2.03 \times 10^{-6}$ | $2.84 \times 10^{-6}$ | **$1.77 \times 10^{-6}$** | -0.314 | **-12.6%** |
| **12**| **`CQfactor`** | GF22_CQF | 1.00 | 0.50 | **0.89** | +0.225 | **-11.2%** |
| **13**| **`Charnok2`** | GF21_CH2 | 0.00215 | 0.00315 | **0.00231** | +0.159 | **+7.4%** |
| **14**| **`TURB_KHRADFAC`**| GF11_TRBKH | 0.85 | 0.50 | **0.82** | +0.089 | **-3.7%** |
| **15**| **`MINRHCRIT`**| GF12_MINRH | 0.89 | 0.80 | **0.86** | +0.336 | **-3.4%** |
| **16**| **`DCS`** | GF05_DCS | $2.90 \times 10^{-4}$ | $2.50 \times 10^{-4}$ | **$2.97 \times 10^{-4}$** | -0.177 | **+2.4%** |
| **17**| **`FDROP_SOOT`**| GF08_FSOOT| 0.05 | 0.09 | **0.049** | -0.014 | **-1.2%** |
| **18**| **`TURNRHCRIT`**| GF02_TURN | 726.0 | 850.0 | **734.3** | +0.067 | **+1.1%** |
| **19**| **`TURB_BETA_RAD`**| GF10_TBRAD| 0.20 | 0.30 | **0.201** | +0.015 | **+0.7%** |
| **20**| **`ALBICEV`** | GF18_ALBICEV| 0.78 | 0.73 | **0.776** | +0.081 | **-0.5%** |
| **21**| **`CICE_AH_MAX`**| GF17_CAHMAX| 0.30 | 0.20 | **0.299** | +0.013 | **-0.4%** |

### Menemenlis GFO Algorithm Comparison ([`results/optimization_results_GFO_Algo.csv`](results/optimization_results_GFO_Algo.csv))
Under bounded prior parameter uncertainties:
- **At $Q = 1.0$**: `RKM` (+51.1%), `AUT_SCALE` (-52.2%), `MINCDNC` (+25.8%), `ENTR_DP` (+115.0%), and `WBFFACTOR` (+44.0%) are consistently identified as the core sensitive drivers of coupled performance.
- **Physical Interpretation**:
  1. **Enhanced Deep & Shallow Entrainment (`ENTR_DP` $\uparrow$, `RKM` $\uparrow$)**: Weak entrainment causes excessive deep convective penetrations and tropospheric thermal bias. Increasing entrainment strengthens lateral environmental air mixing, properly adjusting upper troposphere heating and Hadley circulation strength.
  2. **Microphysics & Cloud Radiation (`WBFFACTOR` $\uparrow$, `AUT_SCALE` $\downarrow$, `MINCDNC` $\uparrow$)**: Increasing the Wegener-Bergeron-Findeisen process and reducing liquid autoconversion preserves supercooled liquid cloud water, reducing shortwave biases over Southern Ocean and mid-latitudes.

---

## Results Walkthrough with Diagnostic Figures

### 1. Global Cost Function Reductions

The optimization achieves substantial cost reductions across both the atmosphere and the ocean:

| Global Cost by Experiment | Global Cost by Variable |
| :---: | :---: |
| ![Cost Comparison Across Experiments](plots/Global_Cost_Breakdown_Experiments.png) | ![Cost Breakdown by Variable](plots/Global_Cost_Breakdown_Variables.png) |
| *Relative global normalized cost comparing Reference (1.0), Optimized (~0.67), and individual perturbation runs.* | *Normalized error improvement broken down across each target variable (Ref vs Optimized).* |

*(GFO Menemenlis formulation results: [Cost by Exp (GFO)](plots/Global_Cost_Breakdown_Experiments_GFO.png) | [Cost by Var (GFO)](plots/Global_Cost_Breakdown_Variables_GFO.png))*

---

### 2. Coupled Model Validation (`GF01_ref` vs `GF_OPT`)

The optimized parameters were implemented in forward simulation runs (`GF_OPT`, `GF_OPT72`, `GF_OPTUpd`) and compared against the baseline control and observational benchmarks:

#### Global Surface Temperature & Sea Surface Temperature
| Global Mean Surface Temperature (TS) | Global Sea Surface Temperature (SST) |
| :---: | :---: |
| ![Global TS Comparison](plots/Global_TS_Comparison.png) | ![Global SST Comparison](plots/Global_SST_Comparison.png) |
| *Global 2m surface temperature time series comparing GF01_ref, tuned runs, and MERRA-2.* | *Global SST time series showing stable adjustment toward ECCO/MERRA-2 observations.* |

#### Tropical Pacific Climate (ENSO & Nino 3.4)
| Nino 3.4 SST Anomaly Comparison | Warming Pattern Map |
| :---: | :---: |
| ![Nino 3.4 SST Comparison](plots/Nino34_Comparison.png) | ![Warming Pattern Map](plots/Warming_Pattern_Map.png) |
| *ENSO variability and amplitude preservation in GF_OPT relative to reference.* | *Global spatial temperature anomalies demonstrating reduced warm/cold biases.* |

---

### 3. Spatial Bias and Difference Maps

The following maps show the spatial structure of model improvements (Control Bias vs Tuned Bias):

#### Surface Temperature (`TS`)
![Surface Temperature Comparison](plots/Map_Comparison_TS.png)
*Left: Reference Control; Middle: Optimized; Right: Target Observations.*

![Surface Temperature Difference](plots/Map_Diff_TS.png)
*Spatial difference map ($\text{Model}_{\text{opt}} - \text{Model}_{\text{ref}}$).*

#### Polar Sea Ice Cover (`FRSEAICE`)
| Sea Ice Comparison | Sea Ice Difference Map |
| :---: | :---: |
| ![Sea Ice Comparison](plots/Map_Comparison_FRSEAICE.png) | ![Sea Ice Difference](plots/Map_Diff_FRSEAICE.png) |

#### Net Shortwave Radiation (`SWGNET`)
| Shortwave Radiation Comparison | Shortwave Difference Map |
| :---: | :---: |
| ![SWGNET Comparison](plots/Map_Comparison_SWGNET.png) | ![SWGNET Difference](plots/Map_Diff_SWGNET.png) |

#### Precipitation & Latent Heat Flux
| Total Precipitation (`PRECTOT`) | Latent Heat Flux (`LHFX`) |
| :---: | :---: |
| ![Precipitation Comparison](plots/Map_Comparison_PRECTOT.png) | ![Latent Heat Flux Comparison](plots/Map_Comparison_LHFX.png) |

---

### 4. Atmosphere & Ocean Seasonal Diagnostics

Seasonal (DJF and JJA) diagnostics for key atmospheric levels and ocean depths:

| Variable & Season | Mean Fields & Residual Differences | Regional Cost Comparison |
| :--- | :---: | :---: |
| **Surface Air Temp (T2M) - DJF** | ![T2M DJF Mean Fields](plots/T2M_DJF_MeanFields.png) | ![T2M DJF Cost](plots/T2M_DJF_CostComparison.png) |
| **Surface Air Temp (T2M) - JJA** | ![T2M JJA Mean Fields](plots/T2M_JJA_MeanFields.png) | ![T2M JJA Cost](plots/T2M_JJA_CostComparison.png) |
| **Zonal Wind 200 hPa (U200) - DJF**| ![U200 DJF Mean Fields](plots/U200_DJF_MeanFields.png) | ![U200 DJF Cost](plots/U200_DJF_CostComparison.png) |
| **Temp 300 hPa (T300) - DJF** | ![T300 DJF Mean Fields](plots/T300_DJF_MeanFields.png) | ![T300 DJF Cost](plots/T300_DJF_CostComparison.png) |
| **Ocean SST (THETA 0m) - DJF** | ![THETA 0m DJF Mean Fields](plots/THETA_0m_DJF_MeanFields.png) | ![THETA 0m DJF Cost](plots/THETA_0m_DJF_CostComparison.png) |
| **Subsurface Temp (THETA 300m) - DJF** | ![THETA 300m DJF Mean Fields](plots/THETA_300m_DJF_MeanFields.png) | ![THETA 300m DJF Cost](plots/THETA_300m_DJF_CostComparison.png) |
| **Ocean Salinity (SALT 0m) - DJF** | ![SALT 0m DJF Mean Fields](plots/SALT_0m_DJF_MeanFields.png) | ![SALT 0m DJF Cost](plots/SALT_0m_DJF_CostComparison.png) |

*To interactively inspect all 156 generated diagnostic plots across seasons and variables, open [`plots_viewer.html`](plots_viewer.html) in any web browser.*

---

## Repository Structure

```
greens_funtions_GEOS/
├── README.md                      # Complete project documentation and walkthrough
├── .gitignore                     # Git ignore rules (strictly excludes large data/NetCDF files)
├── requirements.txt               # Python package dependencies
├── plots_viewer.html              # Portable interactive HTML plot gallery (all 156 plots)
├── scripts/                       # Core tuning and analysis scripts
│   ├── greens_func_tuning.py      # Main coupled tuning (Least Squares & Ridge Regression)
│   ├── greens_func_tuning_GFO.py  # Menemenlis et al. (2005) GFO algorithm
│   ├── post.py                    # Post-processing & validation of GF_OPT vs GF01 & MERRA-2
│   ├── timeseries.py              # Time series analysis (Global TS, SST, Nino 3.4)
│   ├── plot_geos_timeseries.py    # GEOS time series plotting utilities
│   ├── plot_gf01_gfopt_comparison.py # Control vs Optimized comparison maps
│   ├── plot_sst_comparison_ocean_only.py # Ocean-only SST bias analysis
│   └── analyze_cloud_impact.py    # Cloud fraction analysis (CLDLO, CLDMD, CLDHI)
├── results/                       # Optimization result tables (CSV)
│   ├── optimization_results.csv   # Ranked parameter results (Least Squares)
│   └── optimization_results_GFO_Algo.csv # Ranked parameter results (Menemenlis GFO for Q=0, 0.5, 1.0)
├── utils/                         # Helper tools & viewer generators
│   ├── create_html_viewer.py      # Portable plot gallery builder
│   ├── create_html_viewer_base64.py
│   ├── create_html_viewer_artifact.py
│   └── debug_glob.py
└── plots/                         # All 156 diagnostic, comparison, and time series plots
    ├── Global_Cost_Breakdown_Experiments.png
    ├── Global_Cost_Breakdown_Variables.png
    ├── Global_TS_Comparison.png
    ├── Global_SST_Comparison.png
    ├── Nino34_Comparison.png
    ├── Warming_Pattern_Map.png
    ├── Map_Comparison_*.png
    ├── Map_Diff_*.png
    └── <VAR>_<SEASON>_*.png
```

---

## Setup & Execution Guide (Handoff Instructions)

### 1. Prerequisites
Install required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Running in Mock Test Mode (No Large Data Required)
To verify the entire optimization pipeline, data loading, linear inversion, and plot generation without requiring access to hundreds of gigabytes of simulation output:
```bash
python scripts/greens_func_tuning.py --test
```
or with the Menemenlis GFO algorithm:
```bash
python scripts/greens_func_tuning_GFO.py --test
```
This automatically creates synthetic multi-season coupled fields, runs the inversion, reports parameter sensitivities, and generates diagnostic figures in `plots/`.

### 3. Running with Full Model Simulation Data
When working on the compute cluster with the GEOS-MIT simulations mounted:
```bash
# Standard least squares / Ridge inversion:
python scripts/greens_func_tuning.py --root_dir /nobackupp27/afahad/GFEXPv11.7

# Menemenlis et al. (2005) GFO algorithm:
python scripts/greens_func_tuning_GFO.py --root_dir /nobackupp27/afahad/GFEXPv11.7
```

### 4. Running Post-Processing Validation
To regenerate the verification comparisons between `GF01_ref` and `GF_OPT`:
```bash
python scripts/post.py
python scripts/timeseries.py
```

### 5. Interactive Plot Gallery
Open `plots_viewer.html` directly in any web browser:
```bash
# On Mac:
open plots_viewer.html
# On Linux:
xdg-open plots_viewer.html
```
Use the dropdown or keyboard arrow keys (Left/Right, Up/Down) to browse through all 156 diagnostic figures.

---

## References
1. **Menemenlis, D., I. Fukumori, and T. Lee (2005)**: *Using Green's Functions to Calibrate an Ocean General Circulation Model.* Monthly Weather Review, 133(5), 1224–1240. [https://doi.org/10.1175/MWR2912.1](https://doi.org/10.1175/MWR2912.1)
2. **Gelaro, R., et al. (2017)**: *The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2).* Journal of Climate, 30(14), 5419–5454.
3. **Forget, G., et al. (2015)**: *ECCO version 4: an integrated framework for non-linear inverse modeling and global ocean state estimation.* Geoscientific Model Development, 8(10), 3071–3104.
