'''
pipeline/tune.py
Author: maia.advance, maymeridian
Description: ADVANCED Hyperparameter Tuning.
             - Early Stopping: DISABLED (Matches original high-score behavior).
             - Data Loading: OPTIMIZED (Loads once).
             - rsm: ENABLED.
'''

import os
import json
import numpy as np
import optuna

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from io_handler.io_handler import get_prepared_dataset
from config import MODELS_DIR

def run_optimization(n_trials):
    print("--- Loading Data into RAM ---")
    X, y = get_prepared_dataset(dataset_type='train')
    print(f"Data Loaded: {X.shape} samples. Starting Tuning...")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 800, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'border_count': trial.suggest_int('border_count', 128, 254),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10),
            'rsm': trial.suggest_float('rsm', 0.4, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Lossguide', 'Depthwise']),

            # Fixed Settings
            'random_seed': 42,
            'verbose': 0,
            'allow_writing_files': False,
            'loss_function': 'Logloss',
            'thread_count': -1
        }

        if params['grow_policy'] == 'SymmetricTree':
            params['depth'] = trial.suggest_int('depth', 4, 8)
            params['boosting_type'] = 'Ordered'
        else:
            params['depth'] = trial.suggest_int('depth', 4, 12)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 5, 50)
            params['boosting_type'] = 'Plain'

        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 10.0, 30.0)

        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(**params)

            model.fit(X_train, y_train, verbose=False)

            probs = model.predict_proba(X_val)[:, 1]

            best_f1_fold = 0

            for t in np.arange(0.2, 0.8, 0.05):
                preds = (probs >= t).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1_fold:
                    best_f1_fold = score

            f1_scores.append(best_f1_fold)

        return np.mean(f1_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print("\nTuning finished successfully.")
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving best results...")

    if len(study.trials) > 0:
        print("\n" + "="*50)
        print("BEST PARAMETERS:")
        print(study.best_params)
        print(f"Best CV F1: {study.best_value:.4f}")
        print("="*50)

        os.makedirs(MODELS_DIR, exist_ok=True)
        save_path = os.path.join(MODELS_DIR, 'best_params.json')

        final_params = {
            'loss_function': 'Logloss',
            'verbose': 0,
            'allow_writing_files': False,
            'random_seed': 42
        }
        final_params.update(study.best_params)

        with open(save_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        print(f"Saved to {save_path}")


def run_tuning(n_trials=30, force=False):
    """
    Checks if best_params.json exists. If not (or if forced), runs the Optuna study.
    """
    params_path = os.path.join(MODELS_DIR, 'best_params.json')

    # If the file exists and we aren't forcing a retune, just skip
    if os.path.exists(params_path) and not force:
        print("\n[✓] Optimized parameters already exist. Skipping tuning.")
        return

    # Handle the print statements based on whether it was forced or automatic
    if force:
        print(f"\n=== STAGE 0: TUNING (FORCED, {n_trials} Trials) ===")
    else:
        print("\n[!] No optimized parameters found.")
        print(f"--- Initiating Auto-Tuning ({n_trials} trials) ---")

    # Run the actual optimization block
    try:
        run_optimization(n_trials=n_trials)
        print("\n[✓] Tuning Complete.")
    except Exception as e:
        print(f"\n[X] Tuning Failed: {e}. Falling back to default parameters.")