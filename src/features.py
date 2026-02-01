'''
src/features.py
Description: Feature extraction.
'''

import numpy as np
import pandas as pd
import os
import warnings
import time
from datetime import timedelta
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew, spearmanr, linregress

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

# PHYSICS LOGIC

def calculate_template_matching(t_grid, y_pred_g, peak_idx, peak_time, peak_flux):
    """Normalized TDE Shape Matching ONLY."""
    post_peak_indices = np.where(t_grid > peak_time)[0]
    match_tde = 10.0 
    
    if len(post_peak_indices) > 5 and peak_flux > 0:
        y_fade = y_pred_g[post_peak_indices]
        t_fade = t_grid[post_peak_indices] - peak_time
        
        y_norm = y_fade / peak_flux
        half_idx = np.argmax(y_norm < 0.5)
        
        if half_idx > 0:
            t_half = t_fade[half_idx]
            if t_half > 0.1: 
                t_norm = t_fade / t_half 
                y_temp_tde = 1.0 / (1.0 + (t_norm * (2**(1/1.67) - 1)))**1.67
                mask = t_norm < 3.0
                if mask.sum() > 2:
                    match_tde = np.sqrt(np.mean((y_norm[mask] - y_temp_tde[mask])**2))
                    
    return match_tde

def calculate_physics_wars(t_grid, y_pred_g, peak_idx, peak_time, peak_flux):
    # 1. TDE FIT
    post_peak_indices = np.where(t_grid > peak_time)[0]
    tde_error = 10.0 
    linear_decay_slope = 0.0 
    
    if len(post_peak_indices) > 5 and peak_flux > 0:
        y_real_fade = y_pred_g[post_peak_indices]
        t_fade = t_grid[post_peak_indices]
        dt = (t_fade - peak_time) + 10 
        
        y_ideal_tde = peak_flux * (dt / dt[0])**(-1.67)
        residuals_tde = (y_real_fade - y_ideal_tde) / peak_flux
        tde_error = np.mean(residuals_tde**2)

        try:
            log_t = np.log(dt)
            log_y = np.log(y_real_fade + 1e-9)
            slope, _, _, _, _ = linregress(log_t, log_y)
            linear_decay_slope = slope
        except Exception:
            linear_decay_slope = 0.0

    # 2. RISE PHYSICS
    pre_peak_indices = np.where(t_grid < peak_time)[0]
    rise_fireball_error = 10.0
    pre_peak_var = 0.0 
    
    if len(pre_peak_indices) > 5 and peak_flux > 0:
        y_real_rise = y_pred_g[pre_peak_indices]
        t_rise = t_grid[pre_peak_indices]
        if len(t_rise) > 3:
            try:
                coeffs = np.polyfit(t_rise, y_real_rise, 2)
                p = np.poly1d(coeffs)
                residuals_rise = (y_real_rise - p(t_rise)) / peak_flux
                rise_fireball_error = np.mean(residuals_rise**2)
                pre_peak_var = np.var(residuals_rise)
            except Exception:
                pass

    fade_correlation = 0.0
    if len(post_peak_indices) > 2:
        fade_correlation = np.corrcoef(t_grid[post_peak_indices], y_pred_g[post_peak_indices])[0, 1]

    half_max = peak_flux / 2.0
    rise_idx_candidates = np.where((y_pred_g[:peak_idx] <= half_max))[0]
    t_half_rise = t_grid[rise_idx_candidates[-1]] if len(rise_idx_candidates) > 0 else t_grid[0]
    fade_idx_candidates = np.where((y_pred_g[peak_idx:] <= half_max))[0]
    t_half_fade = t_grid[peak_idx + fade_idx_candidates[0]] if len(fade_idx_candidates) > 0 else t_grid[-1]
    fwhm = t_half_fade - t_half_rise
    
    return tde_error, linear_decay_slope, rise_fireball_error, fade_correlation, fwhm, pre_peak_var

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

    significant_negative = (flux_data < -3 * flux_err)
    negative_flux_fraction = significant_negative.sum() / len(flux_data) if len(flux_data) > 0 else 0.0

    significant_mask = flux_data > (3 * flux_err) 
    detection_times = obj_df.loc[significant_mask, 'Time (MJD)']
    total_survey_span = obj_df['Time (MJD)'].max() - obj_df['Time (MJD)'].min()
    
    if len(detection_times) > 4:
        t_10 = np.percentile(detection_times, 10)
        t_90 = np.percentile(detection_times, 90)
        robust_duration = t_90 - t_10
        duty_cycle = robust_duration / total_survey_span if total_survey_span > 0 else 0
    else:
        robust_duration = 0.0
        duty_cycle = 0.0

    flux_kurtosis = kurtosis(flux_data, fisher=True)
    flux_skew = skew(flux_data) 

    X_obs = np.column_stack([obj_df['Time (MJD)'].values, obj_df['Filter'].map(FILTER_WAVELENGTHS).values])
    y_gp_pred = gp.predict(X_obs) * y_scale
    safe_err = np.where(flux_err <= 0, 1e-5, flux_err)
    chi_sq_terms = ((flux_data - y_gp_pred) / safe_err) ** 2
    reduced_chi_square = np.mean(chi_sq_terms)
    reduced_chi_square = min(reduced_chi_square, 1000.0)

    # 100 Points Grid
    t_min, t_max = obj_df['Time (MJD)'].min(), obj_df['Time (MJD)'].max()
    t_grid = np.linspace(t_min, t_max, 100)
    g_wave = FILTER_WAVELENGTHS['g']
    X_pred_g = np.column_stack([t_grid, np.full_like(t_grid, g_wave)])
    y_pred_g = gp.predict(X_pred_g) * y_scale

    peak_idx = np.argmax(y_pred_g)
    peak_time = t_grid[peak_idx]
    peak_flux = y_pred_g[peak_idx]
    threshold = peak_flux / 2.512

    # --- 3. PERCENTILE RATIOS ---
    positive_flux = y_pred_g[y_pred_g > 0]
    
    if len(positive_flux) > 0:
        perc_20 = np.percentile(positive_flux, 20)
        perc_50 = np.percentile(positive_flux, 50)
        perc_80 = np.percentile(positive_flux, 80)
        percentile_ratio_20_50 = perc_20 / (perc_50 + 1e-9)
        percentile_ratio_80_max = perc_80 / (peak_flux + 1e-9)
    else:
        percentile_ratio_20_50 = 0.0
        percentile_ratio_80_max = 0.0

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

    # calculate physics
    tde_error, linear_decay_slope, rise_error, fade_shape, fwhm, pre_peak_var = calculate_physics_wars(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)
    match_tde = calculate_template_matching(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)

    def get_val(t, band):
        val = gp.predict([[t, FILTER_WAVELENGTHS[band]]])[0] * y_scale
        return val if val > 0 else 1e-5

    val_u = get_val(peak_time, 'u')
    val_g = get_val(peak_time, 'g')
    val_r = get_val(peak_time, 'r')
    
    flux_ratio_ug = val_u / val_g 
    flux_ratio_gr = val_g / val_r 

    ug_peak = -2.5 * np.log10(val_u / val_g)
    gr_peak = -2.5 * np.log10(val_g / val_r)
    ur_peak = -2.5 * np.log10(val_u / val_r)

    # COLOR FEATURES
    t_samples = np.linspace(peak_time, peak_time + fade_time, 5)
    g_samples = [get_val(t, 'g') for t in t_samples]
    r_samples = [get_val(t, 'r') for t in t_samples]
    gr_colors = [-2.5 * np.log10(g/r) for g, r in zip(g_samples, r_samples)]
    
    mean_color_gr = np.mean(gr_colors)
    std_color_gr = np.std(gr_colors)
    
    color_monotonicity, _ = spearmanr(np.arange(5), gr_colors)
    if np.isnan(color_monotonicity):
         color_monotonicity = 0.0
    
    try:
        slope, _, _, _, _ = linregress(np.arange(5), gr_colors)
        color_slope_gr = slope
    except Exception:
        color_slope_gr = 0.0

    t_fade_pt = peak_time + (fade_time/2)
    gr_fade = -2.5 * np.log10(get_val(t_fade_pt, 'g') / get_val(t_fade_pt, 'r'))
    color_cooling_rate = gr_fade - gr_peak 
    
    def get_area(band):
        y_band = gp.predict(np.column_stack([t_grid, np.full_like(t_grid, FILTER_WAVELENGTHS[band])])) * y_scale
        return np.trapezoid(y_band, t_grid)
    
    total_area = get_area('u') + get_area('g') + get_area('r') + get_area('i')
    
    rise_fade_ratio = rise_time / fade_time if fade_time > 0 else 0
    area_under_curve = np.trapezoid(y_pred_g, t_grid)
    compactness = area_under_curve / peak_flux if peak_flux > 0 else 0
    rise_slope = amplitude / rise_time if rise_time > 1 else amplitude
    
    baseline_window = int(len(y_pred_g) * 0.15)
    baseline_flux = np.median(y_pred_g[:baseline_window])
    baseline_ratio = baseline_flux / peak_flux if peak_flux > 0 else 0

    return {
        'object_id': obj_id,
        'amplitude': amplitude,
        'ls_time': ls_time,
        'ls_wave': ls_wave,
        'rise_time': rise_time,
        'fade_time': fade_time,
        'fwhm': fwhm,
        'rise_fade_ratio': rise_fade_ratio,
        'compactness': compactness,
        'rise_slope': rise_slope,
        
        # Physics
        'tde_power_law_error': tde_error,
        'template_chisq_tde': match_tde,
        'linear_decay_slope': linear_decay_slope, 
        
        'mean_color_gr': mean_color_gr,
        'std_color_gr': std_color_gr,
        
        'total_radiated_energy_proxy': total_area, 
        'color_monotonicity': color_monotonicity, 
        'negative_flux_fraction': negative_flux_fraction,
        'percentile_ratio_20_50': percentile_ratio_20_50, 
        'percentile_ratio_80_max': percentile_ratio_80_max, 
        'rise_fireball_error': rise_error,
        'pre_peak_var': pre_peak_var, 
        'reduced_chi_square': reduced_chi_square,
        'fade_shape_correlation': fade_shape,
        'baseline_ratio': baseline_ratio,
        'color_cooling_rate': color_cooling_rate,
        'color_slope_gr': color_slope_gr, 
        'ug_peak': ug_peak,
        'gr_peak': gr_peak,
        'ur_peak': ur_peak,
        'flux_ratio_ug': flux_ratio_ug,
        'flux_ratio_gr': flux_ratio_gr,
        'flux_kurtosis': flux_kurtosis,
        'flux_skew': flux_skew, 
        'robust_duration': robust_duration,
        'duty_cycle': duty_cycle
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
    
    print(f"Fitting 2D GPs on {len(unique_ids)} objects using all cores...")
    grouped_data = [group for _, group in lc_clean.groupby('object_id')]
    
    features_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(get_gp_features)(lc_clean.iloc[group.index[0]]['object_id'], group) 
        for group in grouped_data
    )
    
    features_list = [f for f in features_list if f is not None]
    features_df = pd.DataFrame(features_list).fillna(0)

    if 'Z' in log_df.columns:
        print("Merging Redshift & Calculating Luminosity...")
        if 'Z_err' in log_df.columns:
            meta = log_df[['object_id', 'Z', 'Z_err']].copy()
            meta = meta.rename(columns={'Z': 'redshift', 'Z_err': 'redshift_err'})
        else:
            meta = log_df[['object_id', 'Z']].copy()
            meta = meta.rename(columns={'Z': 'redshift'})
        
        features_df = features_df.merge(meta, on='object_id', how='left')
        
        safe_z = features_df['redshift'].clip(lower=0.0)
        time_dilation_factor = 1.0 + safe_z
        
        features_df['rest_rise_time'] = features_df['rise_time'] / time_dilation_factor
        features_df['rest_fade_time'] = features_df['fade_time'] / time_dilation_factor
        features_df['rest_fwhm'] = features_df['fwhm'] / time_dilation_factor

        safe_flux = features_df['amplitude'].clip(lower=0.001)
        features_df['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(safe_z + 0.001)
        features_df['total_radiated_energy'] = features_df['total_radiated_energy_proxy'] * (safe_z + 0.001)**2
        features_df['log_tde_error'] = np.log10(features_df['tde_power_law_error'] + 1e-9)

    if cache_file:
        print(f"Saving features to cache: {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        features_df.to_csv(cache_file, index=False)

    print(f"Completed in {str(timedelta(seconds=int(time.time() - total_start_time)))}")
    return features_df
