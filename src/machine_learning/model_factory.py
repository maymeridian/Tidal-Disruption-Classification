'''
src/machine_learning/model_factory.py
Author: maia.advance, maymeridian
Description: THE EXPERT PANEL ENSEMBLE.
             - Model A: Generalist (All Features)
             - Model B: Morphologist (Shape/Time Features ONLY)
             - Model C: Physicist (Color/Error Features ONLY)
'''

import numpy as np
import json
import os
from catboost import CatBoostClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from config import MODEL_CONFIG, MODELS_DIR

# --- DEFINING THE EXPERTISE ---
# We define the specific columns each expert is allowed to see.
MORPHOLOGY_FEATURES = [
    'rise_time', 'fade_time', 'fwhm', 'rise_fade_ratio', 'compactness', 
    'rise_slope', 'flux_kurtosis', 'robust_duration', 'duty_cycle', 
    'ls_time', 'amplitude'
]

PHYSICS_FEATURES = [
    'tde_power_law_error', 'rise_fireball_error', 'reduced_chi_square', 
    'fade_shape_correlation', 'baseline_ratio', 'color_cooling_rate', 
    'ug_peak', 'gr_peak', 'ur_peak', 'ls_wave', 'redshift', 
    'absolute_magnitude_proxy', 'log_tde_error'
]

class ExpertPanelEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, scale_pos_weight=1.0):
        self.scale_pos_weight = scale_pos_weight
        self.models = [] 
        self.feature_importances_ = None 

    def fit(self, X, y):
        self.models = []
        seed = MODEL_CONFIG['random_seed']
        
        # Load Tuned Params (We use the same tuning for all, which is safe enough)
        base_params = {
            'iterations': 1000, 'depth': 5, 'learning_rate': 0.02,
            'l2_leaf_reg': 10, 'rsm': 0.5, 'loss_function': 'Logloss',
            'verbose': 0, 'allow_writing_files': False,
            'random_seed': seed, 'scale_pos_weight': self.scale_pos_weight
        }
        json_path = os.path.join(MODELS_DIR, 'best_params.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    tuned = json.load(f)
                if 'scale_pos_weight' in tuned: 
                    del tuned['scale_pos_weight']
                base_params.update(tuned)
            except Exception: 
                pass

        # --- 1. THE GENERALIST (All Features) ---
        m1 = CatBoostClassifier(**base_params)
        m1.fit(X, y)
        self.models.append(('generalist', m1, list(X.columns)))
        self.feature_importances_ = m1.feature_importances_ # Log Generalist importance

        # --- 2. THE MORPHOLOGIST (Shape Only) ---
        # Filter X to only shape columns that exist in the dataframe
        shape_cols = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
        if shape_cols:
            m2 = CatBoostClassifier(**base_params)
            m2.fit(X[shape_cols], y)
            self.models.append(('morphologist', m2, shape_cols))

        # --- 3. THE PHYSICIST (Physics Only) ---
        # Filter X to only physics columns
        phys_cols = [c for c in PHYSICS_FEATURES if c in X.columns]
        if phys_cols:
            m3 = CatBoostClassifier(**base_params)
            m3.fit(X[phys_cols], y)
            self.models.append(('physicist', m3, phys_cols))
        
        return self

    def predict_proba(self, X):
        # Weighted Ensemble:
        # Generalist: 60% (Trust the one that sees everything)
        # Morphologist: 20%
        # Physicist: 20%
        
        probs_total = np.zeros(len(X))
        
        # 1. Generalist
        p1 = self.models[0][1].predict_proba(X[self.models[0][2]])[:, 1]
        probs_total += (0.60 * p1)
        
        # 2. Morphologist
        if len(self.models) > 1:
            p2 = self.models[1][1].predict_proba(X[self.models[1][2]])[:, 1]
            probs_total += (0.20 * p2)
            
        # 3. Physicist
        if len(self.models) > 2:
            p3 = self.models[2][1].predict_proba(X[self.models[2][2]])[:, 1]
            probs_total += (0.20 * p3)
            
        return np.vstack([1 - probs_total, probs_total]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# --- FACTORY ---
def get_model(model_name, scale_pos_weight=1.0):
    if model_name == 'catboost':
        return ExpertPanelEnsemble(scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError("Only 'catboost' is supported.")

def train_with_cv(model_name, X, y):
    print("\n--- Running EXPERT PANEL Ensemble ---")
    print("    (Generalist [60%] + Morphologist [20%] + Physicist [20%])")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_CONFIG['random_seed'])
    cv_scores = []
    best_thresholds = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        n_pos = y_train.sum()
        scale_weight = (len(y_train) - n_pos) / n_pos if n_pos > 0 else 1.0

        model = get_model(model_name, scale_pos_weight=scale_weight)
        model.fit(X_train, y_train)
        
        probs_val = model.predict_proba(X_val)[:, 1]
        
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.2, 0.8, 0.02):
            s = f1_score(y_val, (probs_val >= t).astype(int), zero_division=0) 
            if s > best_f1: 
                best_f1 = s
                best_t = t
        
        val_tdes = y_val.sum()
        print(f"   Fold {fold}: F1={best_f1:.4f} (Thresh={best_t:.2f}) - Val TDEs: {val_tdes}")
        
        cv_scores.append(best_f1)
        best_thresholds.append(best_t)
        fold += 1

    avg_f1 = np.mean(cv_scores)
    avg_thresh = np.mean(best_thresholds)
    
    print(f"\n   Average Panel F1: {avg_f1:.4f}")
    
    # Final Train
    print("\n--- Training Final Panel on 100% Data ---")
    n_pos_all = y.sum()
    final_weight = (len(y) - n_pos_all) / n_pos_all
    
    final_model = get_model(model_name, scale_pos_weight=final_weight)
    final_model.fit(X, y)
    
    return final_model, avg_f1, avg_thresh