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
from config import MODEL_PATH, MODELS_DIR
from src.features import get_gp_features, apply_deextinction, apply_quality_cuts
import src.data_loader


class InferenceModel: 
    def __init__(self):     
        model, threshold = self.load_configuration()
        self.model = model
        self.threshold = threshold
    

    def load_configuration(self): 
        threshold_path = os.path.join(MODELS_DIR, 'threshold.txt')
        threshold = 0.4
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f: 
                threshold = float(f.read().strip())

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        return model, threshold


    def _process_single_object(self, lc_df, z, ebv):

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


    def predict(self, lc_df, z, ebv):
        required_cols = {'Time (MJD)', 'Flux', 'Flux_err', 'Filter', 'object_id'}

        if not required_cols.issubset(lc_df.columns):
            return self._build_result(0, 0.0, "missing_columns")

        features = self._process_single_object(lc_df, z, ebv)
        X = features.drop(columns=['object_id'])
        # prediction
        y_prob = self.model.predict_proba(X)[:, 1]
        # apply threshold
        y_pred = (y_prob >= threshold).astype(int)

        return y_pred, y_prob