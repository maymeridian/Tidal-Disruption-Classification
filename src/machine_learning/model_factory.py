'''
src/machine_learning/model_factory.py
Author: maia.advance, maymeridian
Description: Factory pattern using Class Weights (Best for tiny datasets).
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from config import MODEL_CONFIG

def get_model(model_name, scale_pos_weight=1.0):
    """
    Returns an initialized model.
    """
    seed = MODEL_CONFIG['random_seed']

    if model_name == 'xgboost':
        return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, eval_metric='logloss', scale_pos_weight=scale_pos_weight, min_child_weight=2, random_state=seed)
        
    elif model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
    
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=seed)
        
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

def train_with_cv(model_name, X, y):
    """
    Runs 5-Fold Stratified CV using Class Weights (No Oversampling).
    """
    print(f"\n--- Running 5-Fold CV with Class Weights ({model_name}) ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_CONFIG['random_seed'])
    
    cv_scores = []
    best_thresholds = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # CALCULATE DYNAMIC WEIGHT
        # If we have 150 non-TDEs and 6 TDEs: Weight = 150 / 6 = 25.0
        # This tells the model: "Finding a TDE is 25x more important."
        n_pos = y_train_fold.sum()
        n_neg = len(y_train_fold) - n_pos
        scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Train on raw data with heavy weights
        model = get_model(model_name, scale_pos_weight=scale_weight)
        model.fit(X_train_fold, y_train_fold)
        
        # Optimize Threshold
        probs_val = model.predict_proba(X_val_fold)[:, 1]
        best_f1_fold = 0.0
        best_thresh_fold = 0.5
        
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds_fold = (probs_val >= thresh).astype(int)
            score = f1_score(y_val_fold, preds_fold, zero_division=0)
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
    
    # Calculate final weight on full dataset
    n_pos_all = y.sum()
    n_neg_all = len(y) - n_pos_all
    final_weight = n_neg_all / n_pos_all
    print(f"   Final Scale Weight: {final_weight:.2f}")

    final_model = get_model(model_name, scale_pos_weight=final_weight)
    final_model.fit(X, y)
    
    return final_model, avg_f1, avg_thresh