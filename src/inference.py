'''
src/inference.py
Author: maia.advance, maymeridian
Description: inference model for classification of single objects. 
'''
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from config import MODEL_PATH, MODELS_DIR, SCORE_PATH
from src.features import get_gp_features, apply_deextinction, apply_quality_cuts
import src.data_loader


def load_model(): 
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run training first!")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    thresh_path = os.path.join(os.path.dirname(SCORE_PATH), 'threshold.txt')
    threshold = 0.4
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            try:
                threshold = float(f.read().strip())
                print(f"Loaded optimized decision threshold: {threshold}")
            except ValueError:
                print("Warning: Could not read threshold file. Using default.")
    else:
        print("Warning: Threshold file not found. Using default (0.4).")

    return model, threshold


def process_single_object(lc_df, z, ebv):
    lc_df['EBV'] = float(ebv) 
    lc_clean = apply_deextinction(lc_df, log_df=None)
    lc_clean = apply_quality_cuts(lc_clean)

    # GP extraction returns dict, convert to df for model
    obj_id = lc_clean['object_id'].iloc[0]
    feature_dict = get_gp_features(obj_id, lc_clean)
    features = pd.DataFrame([feature_dict])

    # rest frame physics
    z = max(float(z), 0.0)
    dilation = 1.0 + z
    
    features['redshift'] = z
    features['rest_rise_time'] = features['rise_time'] / dilation
    features['rest_fade_time'] = features['fade_time'] / dilation
    features['rest_fwhm'] = features['fwhm'] / dilation

    # amplitude correction
    safe_flux = features['amplitude'].clip(lower=0.001)
    features['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(z + 0.001)
    
    features['total_radiated_energy'] = features['total_radiated_energy_proxy'] * (z + 0.001)**2
    features['log_tde_error'] = np.log10(features['tde_power_law_error'] + 1e-9)
    
    return features


def predict(lc_df, z, ebv):
    required_cols = {'Time (MJD)', 'Flux', 'Flux_err', 'Filter', 'object_id'}

    if not required_cols.issubset(lc_df.columns):
        return 0, 0.0

    features = process_single_object(lc_df, z, ebv)
    X = features.drop(columns=['object_id'])
    # prediction
    y_prob = model.predict_proba(X)[:, 1]
    # apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred, y_prob


    