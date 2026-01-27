'''
src/features.py
Author: maia.advance, maymeridian
Description: Feature extraction. 
             Includes 'Robust Duration' (AGN Killer) to distinguish short TDEs from long AGNs.
'''

import numpy as np
import pandas as pd
import os
import warnings
import time
from datetime import timedelta
from joblib import Parallel, delayed
from scipy.stats import kurtosis

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from extinction import fitzpatrick99
from config import FILTER_WAVELENGTHS, PROCESSED_TRAINING_DATA_PATH, PROCESSED_TESTING_DATA_PATH, TRAIN_LOG_PATH, TEST_LOG_PATH

warnings.filterwarnings("ignore")

# --- HELPER FUNCTIONS ---

def get_log_data(dataset_type):
    if dataset_type == 'train': 
        return pd.read_csv(TRAIN_LOG_PATH)
    elif dataset_type == 'test': 
        return pd.read_csv(TEST_LOG_PATH)
    else: 
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def apply_deextinction(df, log_df):
    if 'Flux_Corrected' in df.columns: 
        return df
    
    if 'EBV' not in df.columns:
        if 'EBV' in log_df.columns:
            ebv_map = log_df.set_index('object_id')['EBV']
            df['EBV'] = df['object_id'].map(ebv_map)
        else:
            df['Flux_Corrected'] = df['Flux']
            df['Flux_err_Corrected'] = df['Flux_err']
            return df

    unique_filters = list(FILTER_WAVELENGTHS.keys())
    unique_wls = np.array([FILTER_WAVELENGTHS[f] for f in unique_filters], dtype=float)
    ext_factors = fitzpatrick99(unique_wls, 1.0)
    ext_map = dict(zip(unique_filters, ext_factors))

    a_lambda = df['Filter'].map(ext_map) * (df['EBV'] * 3.1)
    correction_factor = 10**(a_lambda / 2.5)
    df['Flux_Corrected'] = df['Flux'] * correction_factor
    df['Flux_err_Corrected'] = df['Flux_err'] * correction_factor
    return df

def apply_quality_cuts(lc_df):
    if 'SNR' not in lc_df.columns:
        safe_err = lc_df['Flux_err'].replace(0, 1e-5)
        lc_df['SNR'] = lc_df['Flux'] / safe_err

    valid_mask = (lc_df['Flux'] > 0) 
    counts = lc_df[valid_mask].groupby('object_id').size()
    keep_ids = counts[counts >= 2].index
    return lc_df[lc_df['object_id'].isin(keep_ids)].copy()

def fit_2d_gp(obj_df):
    if 'Flux_Corrected' in obj_df.columns:
        y = obj_df['Flux_Corrected'].values
        y_err = obj_df['Flux_err_Corrected'].values
    else:
        y = obj_df['Flux'].values
        y_err = obj_df['Flux_err'].values

    X = np.column_stack([obj_df['Time (MJD)'].values, obj_df['Filter'].map(FILTER_WAVELENGTHS).values])

    y_scale = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    y_norm = y / y_scale
    y_err_norm = y_err / y_scale

    kernel = ConstantKernel(1.0) * Matern(length_scale=[100, 6000], nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_norm**2, n_restarts_optimizer=0, random_state=42)
    gp.fit(X, y_norm)
    return gp, y_scale

# --- PHYSICS LOGIC ---

def calculate_tde_physics(t_grid, y_pred_g, peak_idx, peak_time, peak_flux):
    # 1. FADE PHYSICS (t^-5/3)
    post_peak_indices = np.where(t_grid > peak_time)[0]
    tde_power_law_error = 10.0 
    
    if len(post_peak_indices) > 5 and peak_flux > 0:
        y_real_fade = y_pred_g[post_peak_indices]
        t_fade = t_grid[post_peak_indices]
        dt = (t_fade - peak_time) + 10 
        y_ideal_tde = peak_flux * (dt / dt[0])**(-1.67)
        residuals = (y_real_fade - y_ideal_tde) / peak_flux
        tde_power_law_error = np.mean(residuals**2)

    # 2. RISE PHYSICS (t^2 Fireball)
    pre_peak_indices = np.where(t_grid < peak_time)[0]
    rise_fireball_error = 10.0
    
    if len(pre_peak_indices) > 5 and peak_flux > 0:
        y_real_rise = y_pred_g[pre_peak_indices]
        t_rise = t_grid[pre_peak_indices]
        
        if len(t_rise) > 3:
            try:
                coeffs = np.polyfit(t_rise, y_real_rise, 2)
                p = np.poly1d(coeffs)
                residuals_rise = (y_real_rise - p(t_rise)) / peak_flux
                rise_fireball_error = np.mean(residuals_rise**2)
            except Exception:
                pass

    # 3. Shape Stats
    fade_correlation = 0.0
    if len(post_peak_indices) > 2:
        fade_correlation = np.corrcoef(t_grid[post_peak_indices], y_pred_g[post_peak_indices])[0, 1]

    half_max = peak_flux / 2.0
    rise_idx_candidates = np.where((y_pred_g[:peak_idx] <= half_max))[0]
    t_half_rise = t_grid[rise_idx_candidates[-1]] if len(rise_idx_candidates) > 0 else t_grid[0]
        
    fade_idx_candidates = np.where((y_pred_g[peak_idx:] <= half_max))[0]
    t_half_fade = t_grid[peak_idx + fade_idx_candidates[0]] if len(fade_idx_candidates) > 0 else t_grid[-1]
        
    fwhm = t_half_fade - t_half_rise
    
    return tde_power_law_error, rise_fireball_error, fade_correlation, fwhm

# --- MAIN EXTRACTION ---

def get_gp_features(obj_id, obj_df):
    try:
        gp, y_scale = fit_2d_gp(obj_df)
    except Exception:
        return None

    params = gp.kernel_.get_params()
    try:
        ls_time = params.get('k2__length_scale', [0,0])[0]
        ls_wave = params.get('k2__length_scale', [0,0])[1]
        amplitude = np.sqrt(params.get('k1__constant_value', 0)) * y_scale
    except Exception:
        ls_time, ls_wave, amplitude = 0, 0, 0

    if 'Flux_Corrected' in obj_df.columns:
        flux_data = obj_df['Flux_Corrected']
        flux_err = obj_df['Flux_err_Corrected']
    else:
        flux_data = obj_df['Flux']
        flux_err = obj_df['Flux_err']

    # --- ROBUST DURATION FEATURES ---
    # Instead of Max-Min (sensitive to outliers), we use 10th-90th percentile.
    significant_mask = flux_data > (3 * flux_err) # 3-sigma detections
    detection_times = obj_df.loc[significant_mask, 'Time (MJD)']
    
    total_survey_span = obj_df['Time (MJD)'].max() - obj_df['Time (MJD)'].min()
    
    if len(detection_times) > 4:
        # Robust Span: Time between 10th and 90th percentile of activity
        t_10 = np.percentile(detection_times, 10)
        t_90 = np.percentile(detection_times, 90)
        robust_duration = t_90 - t_10
        
        # Duty Cycle: What % of the survey was this object "ON"?
        # TDEs should be low (~10-20%). AGNs should be high (~80-100%).
        duty_cycle = robust_duration / total_survey_span if total_survey_span > 0 else 0
    else:
        robust_duration = 0.0
        duty_cycle = 0.0

    # Calculate Flux Kurtosis (Spikiness)
    flux_kurtosis = kurtosis(flux_data, fisher=True)

    # GP Predictions & Chi-Square
    X_obs = np.column_stack([obj_df['Time (MJD)'].values, obj_df['Filter'].map(FILTER_WAVELENGTHS).values])
    y_gp_pred = gp.predict(X_obs) * y_scale
    safe_err = np.where(flux_err <= 0, 1e-5, flux_err)
    chi_sq_terms = ((flux_data - y_gp_pred) / safe_err) ** 2
    reduced_chi_square = np.mean(chi_sq_terms)
    reduced_chi_square = min(reduced_chi_square, 1000.0)

    # GP Grid Features
    t_min, t_max = obj_df['Time (MJD)'].min(), obj_df['Time (MJD)'].max()
    t_grid = np.linspace(t_min, t_max, 100)
    g_wave = FILTER_WAVELENGTHS['g']
    X_pred_g = np.column_stack([t_grid, np.full_like(t_grid, g_wave)])
    y_pred_g = gp.predict(X_pred_g) * y_scale

    peak_idx = np.argmax(y_pred_g)
    peak_time = t_grid[peak_idx]
    peak_flux = y_pred_g[peak_idx]
    threshold = peak_flux / 2.512

    pre_peak = y_pred_g[:peak_idx]
    t_pre = t_grid[:peak_idx]

    if len(pre_peak) > 0 and np.any(pre_peak < threshold):
        drop_idx = np.where(pre_peak < threshold)[0][-1]
        rise_time = peak_time - t_pre[drop_idx]
    else:
        rise_time = peak_time - t_min

    post_peak = y_pred_g[peak_idx:]
    t_post = t_grid[peak_idx:]

    if len(post_peak) > 0 and np.any(post_peak < threshold):
        drop_idx = np.where(post_peak < threshold)[0][0]
        fade_time = t_post[drop_idx] - peak_time
    else:
        fade_time = t_max - peak_time

    tde_error, rise_error, fade_shape, fwhm = calculate_tde_physics(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)

    def get_val(t, band):
        val = gp.predict([[t, FILTER_WAVELENGTHS[band]]])[0] * y_scale
        return val if val > 0 else 1e-5

    ug_peak = -2.5 * np.log10(get_val(peak_time, 'u') / get_val(peak_time, 'g'))
    gr_peak = -2.5 * np.log10(get_val(peak_time, 'g') / get_val(peak_time, 'r'))
    ur_peak = -2.5 * np.log10(get_val(peak_time, 'u') / get_val(peak_time, 'r'))

    t_fade = peak_time + (fade_time/2)
    gr_fade = -2.5 * np.log10(get_val(t_fade, 'g') / get_val(t_fade, 'r'))
    color_cooling_rate = gr_fade - gr_peak 
    
    rise_fade_ratio = rise_time / fade_time if fade_time > 0 else 0
    area_under_curve = np.trapezoid(y_pred_g, t_grid)
    compactness = area_under_curve / peak_flux if peak_flux > 0 else 0
    rise_slope = amplitude / rise_time if rise_time > 1 else amplitude
    
    baseline_window = int(len(y_pred_g) * 0.15)
    baseline_flux = np.median(y_pred_g[:baseline_window])
    baseline_ratio = baseline_flux / peak_flux if peak_flux > 0 else 0

    return {
        'object_id': obj_id,             # Unique identifier for the transient

        # --- GP KERNEL PARAMETERS ---
        'amplitude': amplitude,          # Peak brightness of the fitted model (TDEs are very luminous)
        'ls_time': ls_time,              # Time Length Scale: How "fast" the lightcurve evolves. (TDEs ~months, AGNs ~years)
        'ls_wave': ls_wave,              # Wavelength Length Scale: Correlation between colors. (TDEs are blue in all bands)

        # --- TEMPORAL SHAPE ---
        'rise_time': rise_time,          # Days from detection to Peak. (TDEs rise sharply)
        'fade_time': fade_time,          # Days from Peak to fade out. (TDEs fade slowly over months)
        'fwhm': fwhm,                    # Full Width Half Max: How "sharp" is the peak? (TDEs are sharper than broad SNe)
        'rise_fade_ratio': rise_fade_ratio, # Rise/Fade: TDEs rise fast & fade slow (Ratio < 1). SNe are often more symmetric.
        'compactness': compactness,      # Area Under Curve / Peak: "Energy Density". Low for sharp bursts (TDE), High for blocks (AGN).
        'rise_slope': rise_slope,        # Amplitude / Rise Time: How violent is the explosion? (TDEs are very steep)

        # --- PHYSICS MODELS (The "TDE Detectors") ---
        'tde_power_law_error': tde_error,      # Error vs ideal t^-5/3 gravity decay. (Lower = More likely TDE)
        'rise_fireball_error': rise_error, # Error vs ideal t^2 fireball rise. (Lower = More likely TDE)
        'reduced_chi_square': reduced_chi_square,   # Smoothness. Low = Smooth (TDE/SN). High = Jittery/Stochastic (AGN).
        'fade_shape_correlation': fade_shape,  # Monotonicity. 1.0 = Fades strictly down. <0 = Bumps/Flares (likely AGN).
        'baseline_ratio': baseline_ratio,      # Flux before explosion / Peak Flux. (TDEs ≈ 0, AGNs > 0 because they are always on).

        # --- COLOR & TEMPERATURE ---
        'color_cooling_rate': color_cooling_rate, # Change in (g-r) over time. TDEs cool (get redder) as the debris expands.
        'ug_peak': ug_peak,              # (u - g) color: Measures Ultraviolet intensity. TDEs are extremely UV-bright (negative values).
        'gr_peak': gr_peak,              # (g - r) color: Visual color temperature.
        'ur_peak': ur_peak,              # (u - r) color: Max temp difference. Best differentiator for Blue TDEs vs Red SNe.

        # --- STATISTICAL MORPHOLOGY ---
        'flux_kurtosis': flux_kurtosis,  # "Spikiness" of the flux distribution.

        # --- THE "AGN KILLERS" (Robustness) ---
        'robust_duration': robust_duration, # Time between 10th & 90th percentile of energy. Ignores outlier noise spikes.
        'duty_cycle': duty_cycle            # % of survey time the object was active. TDEs ≈ 10-20% (One burst). AGNs ≈ 100% (Always on).
    }

def extract_features(lc_df, dataset_type='train'):
    total_start_time = time.time()
    
    cache_file = PROCESSED_TRAINING_DATA_PATH if dataset_type == 'train' else PROCESSED_TESTING_DATA_PATH
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_csv(cache_file)

    print(f"Extracting Features for {dataset_type}...")
    log_df = get_log_data(dataset_type)
    lc_df = apply_deextinction(lc_df, log_df)
    lc_clean = apply_quality_cuts(lc_df)
    
    unique_ids = lc_clean['object_id'].unique()
    
    print(f"Fitting 2D GPs on {len(unique_ids)} objects using ALL cores...")
    grouped_data = [group for _, group in lc_clean.groupby('object_id')]
    
    features_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(get_gp_features)(lc_clean.iloc[group.index[0]]['object_id'], group) 
        for group in grouped_data
    )
    
    features_list = [f for f in features_list if f is not None]
    features_df = pd.DataFrame(features_list).fillna(0)

    if 'Z' in log_df.columns:
        print("Merging Redshift & Calculating Luminosity...")
        meta = log_df[['object_id', 'Z', 'Z_err']].copy() if 'Z_err' in log_df.columns else log_df[['object_id', 'Z']].copy()
        meta = meta.rename(columns={'Z': 'redshift', 'Z_err': 'redshift_err'})
        
        features_df = features_df.merge(meta, on='object_id', how='left')
        
        safe_z = features_df['redshift'].clip(lower=0.001)
        safe_flux = features_df['amplitude'].clip(lower=0.001)
        
        features_df['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(safe_z)
        features_df['log_tde_error'] = np.log10(features_df['tde_power_law_error'] + 1e-9)

    if cache_file:
        print(f"Saving features to cache: {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        features_df.to_csv(cache_file, index=False)

    print(f"Completed in {str(timedelta(seconds=int(time.time() - total_start_time)))}")
    return features_df