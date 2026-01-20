'''
src/features.py
Author: maia.advance, maymeridian
Description: Feature extraction including Flux, SNR, Colors, and Metadata.
'''

import numpy as np
import pandas as pd
from extinction import fitzpatrick99
from config import FILTER_WAVELENGTHS

def apply_deextinction(df, log_df):
    """
    Applies the 'Jurassic Park' de-extinction logic.
    """
    print("Applying De-extinction...")

    # 1. Map EBV (Dust extinction)
    if 'EBV' not in df.columns:
        ebv_map = log_df.set_index('object_id')['EBV']
        df['EBV'] = df['object_id'].map(ebv_map)

    # 2. Calculate Correction Factors
    unique_filters = list(FILTER_WAVELENGTHS.keys())
    unique_wls = np.array([FILTER_WAVELENGTHS[f] for f in unique_filters], dtype=float)
    
    ext_factors = fitzpatrick99(unique_wls, 1.0)
    ext_map = dict(zip(unique_filters, ext_factors))

    a_lambda = df['Filter'].map(ext_map) * (df['EBV'] * 3.1)

    # 3. Apply Correction
    df['Flux_Corrected'] = df['Flux'] * 10**(a_lambda / 2.5)

    return df

def extract_features(lc_df, log_df):
    """
    Turns lightcurves into a single row of features, including:
    - Flux Stats
    - SNR Stats
    - Colors (Temperature)
    - Duration
    - Redshift (Z)
    """
    print("Extracting Features (Flux, SNR, Colors, Z)...")

    # 1. Pre-calculation
    safe_err = lc_df['Flux_err'].replace(0, 1e-5) 
    lc_df['SNR'] = lc_df['Flux'] / safe_err

    # 2. Basic Stats (Flux & SNR) (Group by object and filter)
    flux_feats = lc_df.groupby(['object_id', 'Filter'])['Flux_Corrected'].agg(['mean', 'max', 'std']).unstack()
    flux_feats.columns = [f'flux_{stat}_{filt}' for stat, filt in flux_feats.columns]

    snr_feats = lc_df.groupby(['object_id', 'Filter'])['SNR'].agg(['max', 'mean']).unstack()
    snr_feats.columns = [f'snr_{stat}_{filt}' for stat, filt in snr_feats.columns]

    # 3. Temporal Features (Duration) (Length of Object Visibility (Max time - Min time))
    duration_df = lc_df.groupby('object_id')['Time (MJD)'].agg(np.ptp) # ptp = peak to peak (max - min)
    duration_df.name = 'duration'

    # 4. Merge Basic Features 
    features = pd.concat([flux_feats, snr_feats, duration_df], axis=1)
    features = features.fillna(0) # Fill missing filters with 0

    # 5. Color Features (The "Temperature")
    # Color = Magnitude difference, or in Flux space: Ratio.
    # Flux differences for stability: (Mean Flux u - Mean Flux r)
    # TDEs are blue, so u and g should be bright compared to r, i, z.
    
    # If an object is missing 'u', its flux is 0 (from fillna above), so color is -r (valid).
    if 'flux_mean_u' in features.columns and 'flux_mean_r' in features.columns:
        features['color_u_r'] = features['flux_mean_u'] - features['flux_mean_r']
    
    if 'flux_mean_g' in features.columns and 'flux_mean_i' in features.columns:
        features['color_g_i'] = features['flux_mean_g'] - features['flux_mean_i']

    # 6. Add Metadata (Redshift)
    # Merge Z and Z_err from the log file.
    # Reset index to make 'object_id' a column again for merging
    features = features.reset_index()
    meta_features = log_df[['object_id', 'Z', 'Z_err']].copy()
    
    # Z_err might be NaN in training data, fill with 0
    meta_features['Z_err'] = meta_features['Z_err'].fillna(0)
    
    # Finally, merge the features
    final_features = features.merge(meta_features, on='object_id', how='left')

    return final_features