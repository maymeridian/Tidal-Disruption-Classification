'''
src/machine_learning/model_factory.py
Author: maia.advance, maymeridian
Description: Factory pattern. Automatically loads 'best_params.json' if available.
'''

import numpy as np
import json
import os
from catboost import CatBoostClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from config import MODEL_CONFIG, MODELS_DIR

def get_model(model_name, scale_pos_weight=1.0):
    """
    Returns an initialized model.
    Prioritizes tuned parameters from 'models/best_params.json'.
    """
    seed = MODEL_CONFIG['random_seed']

    if model_name == 'catboost':
        
        # 1. DEFAULT PARAMS (Safe Fallback)
        params = {
            'iterations': 1000,
            'depth': 5,
            'learning_rate': 0.02,
            'l2_leaf_reg': 10,
            'rsm': 0.5,
            'loss_function': 'Logloss',
            'random_seed': seed,
            'verbose': 0,
            'allow_writing_files': False,
            'scale_pos_weight': scale_pos_weight
        }

        # 2. CHECK FOR OPTIMIZED PARAMS
        json_path = os.path.join(MODELS_DIR, 'best_params.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    tuned_params = json.load(f)
                
                # Update defaults with tuned values
                # Note: We override scale_pos_weight only if it was tuned
                # But typically for CV training we want the dynamic weight passed in arguments
                # So we keep the dynamic weight unless you decide otherwise.
                # For now, let's load everything BUT scale_pos_weight to let train_with_cv control the balance.
                
                if 'scale_pos_weight' in tuned_params:
                    del tuned_params['scale_pos_weight'] 
                    
                params.update(tuned_params)
                # print("  (Using optimized parameters from best_params.json)") 
            except Exception as e:
                print(f"Warning: Could not load best_params.json: {e}")

        # Ensure dynamic weight is applied (Factory argument overrides JSON if conflict)
        params['scale_pos_weight'] = scale_pos_weight

        return CatBoostClassifier(**params)
        
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Only 'catboost' is supported.")

def train_with_cv(model_name, X, y):
    """
    Runs 5-Fold Stratified CV using Weighted CatBoost.
    """
    print(f"\n--- Running 5-Fold CV ({model_name}) ---")
    
    # Check if we are using tuned params
    json_path = os.path.join(MODELS_DIR, 'best_params.json')
    if os.path.exists(json_path):
        print("✓ Optimization active: Using tuned parameters.")
    else:
        print("⚠ Optimization inactive: Using default parameters.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_CONFIG['random_seed'])
    
    cv_scores = []
    best_thresholds = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # CALCULATE WEIGHT 
        n_pos = y_train_fold.sum()
        n_neg = len(y_train_fold) - n_pos
        scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = get_model(model_name, scale_pos_weight=scale_weight)
        model.fit(X_train_fold, y_train_fold)
        
        # Optimize Threshold
        probs_val = model.predict_proba(X_val_fold)[:, 1]
        best_f1_fold = 0.0
        best_thresh_fold = 0.5
        
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds_fold = (probs_val >= thresh).astype(int)
            # type: ignore to silence linter
            score = f1_score(y_val_fold, preds_fold, zero_division=0) # type: ignore

            if score > best_f1_fold:
                best_f1_fold = score
                best_thresh_fold = thresh
        
        val_tdes = y_val_fold.sum()
        print(f"   Fold {fold}: F1={best_f1_fold:.4f} (Thresh={best_thresh_fold:.2f}) - Val TDEs: {val_tdes}")
        
        cv_scores.append(best_f1_fold)
        best_thresholds.append(best_thresh_fold)
        fold += 1

    avg_f1 = np.mean(cv_scores)
    avg_thresh = np.mean(best_thresholds)
    
    print(f"\n   Average CV F1: {avg_f1:.4f}")
    print(f"   Optimized Threshold: {avg_thresh:.2f}")

    # FINAL PRODUCTION TRAINING
    print("\n--- Training Final Production Model (100% Data) ---")
    
    n_pos_all = y.sum()
    n_neg_all = len(y) - n_pos_all
    final_weight = n_neg_all / n_pos_all
    print(f"   Final Scale Weight: {final_weight:.2f}")

    final_model = get_model(model_name, scale_pos_weight=final_weight)
    final_model.fit(X, y)
    
    return final_model, avg_f1, avg_thresh