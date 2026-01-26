'''
src/features.py
Author: maia.advance, maymeridian
Description: Feature extraction with LUMINOSITY, PHYSICS, and RATIO features.
'''

import numpy as np
import pandas as pd
import os
import warnings
import time
from datetime import timedelta
from joblib import Parallel, delayed

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
    """
    Calculates normalized physics errors.
    """
    # 1. Power Law Fit
    post_peak_indices = np.where(t_grid > peak_time)[0]
    tde_power_law_error = 10.0 # Default high penalty
    
    if len(post_peak_indices) > 5 and peak_flux > 0:
        y_real_fade = y_pred_g[post_peak_indices]
        t_fade = t_grid[post_peak_indices]
        
        dt = (t_fade - peak_time) + 10 
        y_ideal_tde = peak_flux * (dt / dt[0])**(-1.67)
        
        # Normalize by Peak Flux
        # This ensures bright objects aren't punished just for being bright
        residuals = (y_real_fade - y_ideal_tde) / peak_flux
        tde_power_law_error = np.mean(residuals**2)

    # 2. Fade Shape Correlation (Monotonic Check)
    fade_correlation = 0.0
    if len(post_peak_indices) > 2:
        fade_correlation = np.corrcoef(t_grid[post_peak_indices], y_pred_g[post_peak_indices])[0, 1]

    # 3. FWHM (Full Width Half Max) - "How fat is the curve?"
    # TDEs are often sharper than diffusing Supernovae
    half_max = peak_flux / 2.0
    
    # Find rise crossing
    rise_idx_candidates = np.where((y_pred_g[:peak_idx] <= half_max))[0]
    if len(rise_idx_candidates) > 0:
        t_half_rise = t_grid[rise_idx_candidates[-1]]
    else:
        t_half_rise = t_grid[0]
        
    # Find fade crossing
    fade_idx_candidates = np.where((y_pred_g[peak_idx:] <= half_max))[0]
    if len(fade_idx_candidates) > 0:
        t_half_fade = t_grid[peak_idx + fade_idx_candidates[0]]
    else:
        t_half_fade = t_grid[-1]
        
    fwhm = t_half_fade - t_half_rise

    return tde_power_law_error, fade_correlation, fwhm

# --- MAIN EXTRACTION ---

def get_gp_features(obj_id, obj_df):
    try:
        gp, y_scale = fit_2d_gp(obj_df)
    except Exception:
        return None

    # GP Parameters
    params = gp.kernel_.get_params()
    try:
        ls_time = params.get('k2__length_scale', [0,0])[0]
        ls_wave = params.get('k2__length_scale', [0,0])[1]
        amplitude = np.sqrt(params.get('k1__constant_value', 0)) * y_scale
    except Exception:
        ls_time, ls_wave, amplitude = 0, 0, 0

    t_min, t_max = obj_df['Time (MJD)'].min(), obj_df['Time (MJD)'].max()
    t_grid = np.linspace(t_min, t_max, 100)
    
    g_wave = FILTER_WAVELENGTHS['g']
    X_pred_g = np.column_stack([t_grid, np.full_like(t_grid, g_wave)])
    y_pred_g = gp.predict(X_pred_g) * y_scale

    peak_idx = np.argmax(y_pred_g)
    peak_time = t_grid[peak_idx]
    peak_flux = y_pred_g[peak_idx]
    threshold = peak_flux / 2.512

    # Rise/Fade
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

    # PHYSICS FEATURES
    tde_error, fade_shape, fwhm = calculate_tde_physics(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)

    # Colors
    def get_val(t, band):
        val = gp.predict([[t, FILTER_WAVELENGTHS[band]]])[0] * y_scale
        return val if val > 0 else 1e-5

    ug_peak = -2.5 * np.log10(get_val(peak_time, 'u') / get_val(peak_time, 'g'))
    gr_peak = -2.5 * np.log10(get_val(peak_time, 'g') / get_val(peak_time, 'r'))
    
    # Cooling Rate
    t_fade = peak_time + (fade_time/2)
    gr_fade = -2.5 * np.log10(get_val(t_fade, 'g') / get_val(t_fade, 'r'))
    color_cooling_rate = gr_fade - gr_peak 
    
    # RATIO FEATURES (NEW)
    rise_fade_ratio = rise_time / fade_time if fade_time > 0 else 0
    
    area_under_curve = np.trapz(y_pred_g, t_grid)
    compactness = area_under_curve / peak_flux if peak_flux > 0 else 0

    return {
        'object_id': obj_id,
        'amplitude': amplitude, # This is FLUX amplitude
        'ls_time': ls_time,
        'ls_wave': ls_wave,
        'rise_time': rise_time,
        'fade_time': fade_time,
        'fwhm': fwhm, 
        'rise_fade_ratio': rise_fade_ratio, 
        'compactness': compactness,
        'tde_power_law_error': tde_error, 
        'fade_shape_correlation': fade_shape,
        'color_cooling_rate': color_cooling_rate,
        'ug_peak': ug_peak,
        'gr_peak': gr_peak
    }

def extract_features(lc_df, dataset_type='train'):
    total_start_time = time.time()
    
    # 1. CACHE CHECK (Uncommented for normal use)
    cache_file = PROCESSED_TRAINING_DATA_PATH if dataset_type == 'train' else PROCESSED_TESTING_DATA_PATH
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_csv(cache_file)

    # 2. PREP
    print(f"Extracting Features for {dataset_type}...")
    log_df = get_log_data(dataset_type)
    lc_df = apply_deextinction(lc_df, log_df)
    lc_clean = apply_quality_cuts(lc_df)
    
    unique_ids = lc_clean['object_id'].unique()
    
    # 3. PARALLEL EXECUTION
    print(f"Fitting 2D GPs on {len(unique_ids)} objects using ALL cores...")
    grouped_data = [group for _, group in lc_clean.groupby('object_id')]
    
    features_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(get_gp_features)(lc_clean.iloc[group.index[0]]['object_id'], group) 
        for group in grouped_data
    )
    
    features_list = [f for f in features_list if f is not None]
    features_df = pd.DataFrame(features_list).fillna(0)

    # 4. MERGE REDSHIFT & CALCULATE ABSOLUTE MAGNITUDE
    if 'Z' in log_df.columns:
        print("Merging Redshift & Calculating Luminosity...")
        
        # Prepare Metadata
        meta = log_df[['object_id', 'Z', 'Z_err']].copy() if 'Z_err' in log_df.columns else log_df[['object_id', 'Z']].copy()
        meta = meta.rename(columns={'Z': 'redshift', 'Z_err': 'redshift_err'})
        
        # Merge
        features_df = features_df.merge(meta, on='object_id', how='left')
        
        # --- THE LUMINOSITY CALCULATION ---
        # M = m - 5*log10(D_L) + C
        # Since Flux ~ 10^(-0.4 * m), we can derive: M ~ -2.5*log10(Flux) - 5*log10(z)
        # This gives us a proxy for Intrinsic Brightness (Luminosity)
        
        # Clean redshift (avoid log(0) or log(neg))
        safe_z = features_df['redshift'].clip(lower=0.001)
        safe_flux = features_df['amplitude'].clip(lower=0.001)
        
        # Calculate Proxy Absolute Magnitude
        # (Negative values mean brighter in astronomy, but raw values work for ML)
        features_df['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(safe_z)
        
        # Log-Transform the Power Law Error (Squash the dynamic range)
        features_df['log_tde_error'] = np.log10(features_df['tde_power_law_error'] + 1e-9)

    # 5. SAVE
    if cache_file:
        print(f"Saving features to cache: {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        features_df.to_csv(cache_file, index=False)

    print(f"Completed in {str(timedelta(seconds=int(time.time() - total_start_time)))}")
    return features_df