'''
src/features.py
Author: maia.advance, maymeridian
Description: Fully automated feature extraction. FAST VERSION (Parallelized).
'''

import numpy as np
import pandas as pd
import os
import warnings
import time
from datetime import timedelta
from joblib import Parallel, delayed

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from extinction import fitzpatrick99
from config import FILTER_WAVELENGTHS, PROCESSED_TRAINING_DATA_PATH, PROCESSED_TESTING_DATA_PATH, TRAIN_LOG_PATH, TEST_LOG_PATH

# Suppress GP convergence warnings to keep terminal clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

    print("Applying De-extinction...")
    if 'EBV' not in df.columns:
        if 'EBV' in log_df.columns:
            ebv_map = log_df.set_index('object_id')['EBV']
            df['EBV'] = df['object_id'].map(ebv_map)
        else:
            print("Warning: EBV column not found. Skipping de-extinction.")
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
    print("Applying Quality Cuts...")
    if 'SNR' not in lc_df.columns:
        safe_err = lc_df['Flux_err'].replace(0, 1e-5)
        lc_df['SNR'] = lc_df['Flux'] / safe_err

    valid_mask = (lc_df['Flux'] > 0) 
    valid_points = lc_df[valid_mask]
    counts = valid_points.groupby('object_id').size()
    keep_ids = counts[counts >= 2].index # Keep objects with at least 2 points

    print(f"Retained {len(keep_ids)} objects out of {lc_df['object_id'].nunique()}.")
    return lc_df[lc_df['object_id'].isin(keep_ids)].copy()

def fit_2d_gp(obj_df):
    # Auto-fallback to raw flux if corrected is missing
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

    # Kernel setup
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * Matern(length_scale=[100, 6000], length_scale_bounds=[(1e-2, 1e5), (1e-2, 1e5)], nu=1.5)

    # SPEED OPTIMIZATION: n_restarts_optimizer=0 (was 2)
    # This makes fitting ~3x faster by avoiding "retry" attempts.
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_norm**2, n_restarts_optimizer=0, random_state=42)
    gp.fit(X, y_norm)

    return gp, y_scale

def get_gp_features(obj_id, obj_df):
    # Wrap in try-except to prevent one bad object from crashing the parallel loop
    try:
        gp, y_scale = fit_2d_gp(obj_df)
    except Exception:
        return None

    params = gp.kernel_.get_params()
    try:
        length_scales = params['k2__length_scale']
        amplitude = np.sqrt(params['k1__constant_value']) * y_scale
    except KeyError:
        try:
             length_scales = params['k1__length_scale']
             amplitude = np.sqrt(params['k2__constant_value']) * y_scale
        except (KeyError, TypeError, ValueError): 
             length_scales = [0, 0]
             amplitude = 0

    ls_time = length_scales[0] if len(length_scales) > 0 else 0
    ls_wave = length_scales[1] if len(length_scales) > 1 else 0

    # Grid Prediction
    t_min, t_max = obj_df['Time (MJD)'].min(), obj_df['Time (MJD)'].max()
    t_grid = np.linspace(t_min, t_max, 100)
    g_wave = FILTER_WAVELENGTHS['g']
    
    X_pred_g = np.column_stack([t_grid, np.full_like(t_grid, g_wave)])
    y_pred_g, _ = gp.predict(X_pred_g, return_std=True)
    y_pred_g *= y_scale

    peak_idx = np.argmax(y_pred_g)
    peak_time = t_grid[peak_idx]
    peak_flux = y_pred_g[peak_idx] 
    threshold = peak_flux / 2.512
    
    # Rise/Fade Calculation
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

    # Colors
    def get_val(t, band):
        val = gp.predict([[t, FILTER_WAVELENGTHS[band]]])[0] * y_scale
        return val if val > 0 else 1e-5

    ug_peak = -2.5 * np.log10(get_val(peak_time, 'u') / get_val(peak_time, 'g'))
    gr_peak = -2.5 * np.log10(get_val(peak_time, 'g') / get_val(peak_time, 'r'))
    ri_peak = -2.5 * np.log10(get_val(peak_time, 'r') / get_val(peak_time, 'i'))
    
    t_pre_mid = peak_time - (rise_time / 2)
    t_post_mid = peak_time + (fade_time / 2)

    gr_pre = -2.5 * np.log10(get_val(t_pre_mid, 'g') / get_val(t_pre_mid, 'r'))
    gr_post = -2.5 * np.log10(get_val(t_post_mid, 'g') / get_val(t_post_mid, 'r'))
    ri_pre = -2.5 * np.log10(get_val(t_pre_mid, 'r') / get_val(t_pre_mid, 'i'))
    ri_post = -2.5 * np.log10(get_val(t_post_mid, 'r') / get_val(t_post_mid, 'i'))

    return {
        'object_id': obj_id,
        'amplitude': amplitude,
        'length_scale_time': ls_time,
        'length_scale_wave': ls_wave,
        'rise_time': rise_time,
        'fade_time': fade_time,
        'ug_peak': ug_peak,
        'mean_gr_pre': gr_pre,
        'mean_gr_post': gr_post,
        'mean_ri_pre': ri_pre,
        'mean_ri_post': ri_post,
        'slope_gr_pre': (gr_peak - gr_pre) / (rise_time/2) if rise_time > 0 else 0,
        'slope_gr_post': (gr_post - gr_peak) / (fade_time/2) if fade_time > 0 else 0,
        'slope_ri_pre': (ri_peak - ri_pre) / (rise_time/2) if rise_time > 0 else 0,
        'slope_ri_post': (ri_post - ri_peak) / (fade_time/2) if fade_time > 0 else 0
    }

def extract_features(lc_df, dataset_type='train'):
    total_start_time = time.time()
    
    # 1. CACHE CHECK
    cache_file = PROCESSED_TRAINING_DATA_PATH if dataset_type == 'train' else PROCESSED_TESTING_DATA_PATH
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        features_df = pd.read_csv(cache_file)
        print(f"Loaded {len(features_df)} entries.")
        return features_df

    # 2. PREP
    print(f"Extracting Features for {dataset_type}...")
    log_df = get_log_data(dataset_type)
    lc_df = apply_deextinction(lc_df, log_df)
    lc_clean = apply_quality_cuts(lc_df)
    
    unique_ids = lc_clean['object_id'].unique()
    total_objects = len(unique_ids)
    
    # 3. PARALLEL EXECUTION (The Speedup!)
    print(f"Fitting 2D GPs on {total_objects} objects using ALL available cores...")
    
    # Prepare data for parallel execution (Group by object_id)
    # This prevents searching the massive DataFrame inside every loop iteration
    grouped_data = [group for _, group in lc_clean.groupby('object_id')]
    
    # n_jobs=-1 uses all available CPU cores
    features_list = Parallel(n_jobs=-1, verbose=5)(
        delayed(get_gp_features)(lc_clean.iloc[group.index[0]]['object_id'], group) 
        for group in grouped_data
    )
    
    # Filter out Nones (failed fits)
    features_list = [f for f in features_list if f is not None]

    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0)

    # 4. MERGE METADATA
    if 'Z' in log_df.columns:
        print("Merging Redshift...")
        cols_to_merge = ['object_id', 'Z']
        if 'Z_err' in log_df.columns:
            cols_to_merge.append('Z_err')
            
        meta = log_df[cols_to_merge].copy()
        rename_map = {'Z': 'redshift', 'Z_err': 'redshift_err'}
        meta = meta.rename(columns=rename_map)
        features_df = features_df.merge(meta, on='object_id', how='left')
    
    # 5. SAVE
    if cache_file:
        print(f"Saving features to cache: {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        features_df.to_csv(cache_file, index=False)

    total_time = time.time() - total_start_time
    print(f"Feature Extraction Completed in {str(timedelta(seconds=int(total_time)))}")

    return features_df