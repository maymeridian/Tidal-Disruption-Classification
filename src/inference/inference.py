'''
src/inference.py
Author: maia.advance, maymeridian
Description: inference model for classification of single objects. 
'''

import pandas as pd
import numpy as np
import joblib
import os
from config import MODEL_PATH, SCORE_PATH
from machine_learning.features import get_gp_features, apply_deextinction, apply_quality_cuts
from machine_learning.model_factory import MORPHOLOGY_FEATURES, PHYSICS_FEATURES


def load_inference_model(): 
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


def process_single_object(lc_df, z, z_err, ebv):

    lc_df['EBV'] = float(ebv) 
    lc_clean = apply_deextinction(lc_df, log_df=None)
    lc_clean = apply_quality_cuts(lc_clean)

    # GP extraction returns dict, convert to df for model
    obj_id = lc_clean['object_id'].iloc[0]
    feature_dict = get_gp_features(obj_id, lc_clean)

    if feature_dict is None:
        print(f"GP Extraction failed for {obj_id}")
        return None

    features = pd.DataFrame([feature_dict]).fillna(0)
    
    # Rest frame physics
    safe_z = max(float(z), 0.0)
    dilation = 1.0 + safe_z

    features['redshift'] = safe_z
    features['redshift_err'] = float(z_err) 
    features['rest_rise_time'] = features['rise_time'] / dilation
    features['rest_fade_time'] = features['fade_time'] / dilation
    features['rest_fwhm'] = features['fwhm'] / dilation

    # Amplitude correction
    safe_flux = features['amplitude'].clip(lower=0.001)
    features['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(z + 0.001)

    features['total_radiated_energy'] = features['total_radiated_energy_proxy'] * (z + 0.001)**2
    features['log_tde_error'] = np.log10(features['tde_power_law_error'] + 1e-9)

    return features.fillna(0)


def predict_single_object(model, X):
    p_base = model.models['base'].predict_proba(X)[:, 1]
    p_morph, p_phys = p_base 

    if 'morphology' in model.models:
        cols = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
        p_morph = model.models['morphology'].predict_proba(X[cols])[:, 1]

    if 'physics' in model.models:
        cols = [c for c in PHYSICS_FEATURES if c in X.columns]
        p_phys = model.models['physics'].predict_proba(X[cols])[:, 1]

    p_mlp = model.models['mlp'].predict_proba(X)[:, 1]
    p_knn = model.models['knn'].predict_proba(X)[:, 1]

    final_prob = (0.48 * p_base) + (0.16 * p_morph) + (0.16 * p_phys) + (0.10 * p_mlp) + (0.10 * p_knn)

    return float(final_prob[0])
