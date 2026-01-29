'''
src/machine_learning/model_factory.py
Description: Hybrid Ensemble Classifier.
             - Architecture: Weighted Ensemble of Gradient Boosting (CatBoost), 
               Neural Networks (MLP), and Nearest Neighbors (KNN).
             - Strategy: 
                1. Base Model (48%): CatBoost trained on all features.
                2. Domain Models (32%): CatBoost trained on specific morphology/physics subsets.
                3. Support Models (20%): MLP and KNN trained on all features for non-linear diversity.
             - Features: Uses physics-corrected (rest-frame) metrics.
'''

import numpy as np
import json
import os
from catboost import CatBoostClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from config import MODEL_CONFIG, MODELS_DIR

# --- FEATURE SUBSETS ---

# Subset 1: Morphology & Temporal Evolution Features
MORPHOLOGY_FEATURES = [
    'rest_rise_time', 
    'rest_fade_time', 
    'rest_fwhm', 
    'ls_time',             
    'rise_fade_ratio', 
    'compactness', 
    'rise_slope', 
    'flux_kurtosis', 
    'robust_duration', 
    'duty_cycle', 
    'pre_peak_var', 
    'amplitude'
]

# Subset 2: Physics & Color Metrics
PHYSICS_FEATURES = [
    'tde_power_law_error', 
    'rise_fireball_error', 
    'reduced_chi_square', 
    'ls_wave',             
    'fade_shape_correlation', 
    'baseline_ratio', 
    'color_cooling_rate', 
    'color_slope_gr', 
    'flux_ratio_ug', 
    'flux_ratio_gr', 
    'ug_peak', 'gr_peak', 'ur_peak', 
    'redshift', 
    'absolute_magnitude_proxy', 
    'log_tde_error'
]

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, scale_pos_weight=1.0):
        self.scale_pos_weight = scale_pos_weight
        self.models = {} 
        self.feature_importances_ = None

    def fit(self, X, y):
        seed = MODEL_CONFIG['random_seed']
        
        # --- GRADIENT BOOSTING PARAMETERS ---
        cb_params = {
            'iterations': 1000, 'depth': 5, 'learning_rate': 0.02,
            'l2_leaf_reg': 10, 'rsm': 0.5, 'loss_function': 'Logloss',
            'verbose': 0, 'allow_writing_files': False,
            'random_seed': seed, 'scale_pos_weight': self.scale_pos_weight
        }
        
        # Load optimized hyperparameters if available
        json_path = os.path.join(MODELS_DIR, 'best_params.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: 
                    tuned = json.load(f)
                if 'scale_pos_weight' in tuned: 
                    del tuned['scale_pos_weight']
                cb_params.update(tuned)
            except Exception: 
                pass

        # 1. Base Model (CatBoost - All Features)
        self.models['base'] = CatBoostClassifier(**cb_params)
        self.models['base'].fit(X, y)
        self.feature_importances_ = self.models['base'].feature_importances_

        # 2. Morphology Sub-Model (CatBoost - Shape Features)
        cols_morph = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
        if cols_morph:
            self.models['morphology'] = CatBoostClassifier(**cb_params)
            self.models['morphology'].fit(X[cols_morph], y)

        # 3. Physics Sub-Model (CatBoost - Physics Features)
        cols_phys = [c for c in PHYSICS_FEATURES if c in X.columns]
        if cols_phys:
            self.models['physics'] = CatBoostClassifier(**cb_params)
            self.models['physics'].fit(X[cols_phys], y)

        # 4. Support Model A: Multi-Layer Perceptron (Neural Network)
        self.models['mlp'] = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('net', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', 
                                  solver='adam', alpha=0.01, max_iter=600, 
                                  random_state=seed))
        ])
        self.models['mlp'].fit(X, y)

        # 5. Support Model B: K-Nearest Neighbors
        self.models['knn'] = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', p=2))
        ])
        self.models['knn'].fit(X, y)
        
        return self

    def predict_proba(self, X):
        # 1. Generate Component Predictions
        p_base = self.models['base'].predict_proba(X)[:, 1]
        
        p_morph = p_base # Fallback
        if 'morphology' in self.models:
            cols = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
            p_morph = self.models['morphology'].predict_proba(X[cols])[:, 1]
            
        p_phys = p_base # Fallback
        if 'physics' in self.models:
            cols = [c for c in PHYSICS_FEATURES if c in X.columns]
            p_phys = self.models['physics'].predict_proba(X[cols])[:, 1]
            
        p_mlp = self.models['mlp'].predict_proba(X)[:, 1]
        p_knn = self.models['knn'].predict_proba(X)[:, 1]
        
        # 2. Ensemble Aggregation (Weighted Average)
        # Weights logic: 
        # - Gradient Boosting (80% Total): Split 60% Base, 20% Morph, 20% Phys
        # - Support Models (20% Total): Split 50% MLP, 50% KNN
        
        final_prob = (0.48 * p_base) + \
                     (0.16 * p_morph) + \
                     (0.16 * p_phys) + \
                     (0.10 * p_mlp) + \
                     (0.10 * p_knn)
        
        return np.vstack([1 - final_prob, final_prob]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# --- FACTORY ---
def get_model(model_name, scale_pos_weight=1.0):
    if model_name == 'catboost':
        return EnsembleClassifier(scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError("Only 'catboost' is supported.")

def train_with_cv(model_name, X, y):
    print("\n--- Initializing Hybrid Ensemble Classifier ---")
    print("    Configuration: Gradient Boosting (Core) + MLP/KNN (Support)")
    
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
        print(f"   Fold {fold}: F1={best_f1:.4f} (Thresh={best_t:.2f}) - Validation Positives: {val_tdes}")
        
        cv_scores.append(best_f1)
        best_thresholds.append(best_t)
        fold += 1

    avg_f1 = np.mean(cv_scores)
    avg_thresh = np.mean(best_thresholds)
    
    print(f"\n   Average Ensemble F1: {avg_f1:.4f}")
    
    # Final Train
    print("\n--- Training Final Production Model ---")
    n_pos_all = y.sum()
    final_weight = (len(y) - n_pos_all) / n_pos_all
    
    final_model = get_model(model_name, scale_pos_weight=final_weight)
    final_model.fit(X, y)
    
    return final_model, avg_f1, avg_thresh