'''
src/machine_learning/tune.py
Author: maia.advance, maymeridian
Description: Automated Hyperparameter Tuning. Accepts command line args for trial count.
'''

import sys
import os
import argparse

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import optuna
import json 
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.data_loader import get_prepared_dataset
from config import MODELS_DIR

def objective(trial):
    X, y = get_prepared_dataset(dataset_type='train')
    
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
        'rsm': trial.suggest_float('rsm', 0.4, 0.9), 
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
        'loss_function': 'Logloss'
    }
    
    params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 15.0, 25.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        best_f1 = 0
        for t in np.arange(0.2, 0.8, 0.05):
            preds = (probs >= t).astype(int)
            # type: ignore
            score = f1_score(y_val, preds, zero_division=0) 
            if score > best_f1: 
                best_f1 = score
        
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores)

if __name__ == "__main__":
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30, help='Number of trials to run')
    args = parser.parse_args()

    print(f"--- Starting Optuna Search ({args.trials} Trials) ---")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction='maximize')
    
    # Use the argument here
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print(study.best_params)
    print(f"Best CV F1: {study.best_value:.4f}")
    print("="*50)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, 'best_params.json')
    
    with open(save_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    print(f"✓ Saved best parameters to {save_path}")