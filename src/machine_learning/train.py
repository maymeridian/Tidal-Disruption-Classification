'''
src/machine_learning/train.py
Author: maia.advance, maymeridian
Description: Training pipeline. Cleaned to focus on the robust 0.64 strategy.
'''

import os
import joblib
from datetime import datetime

from src.data_loader import get_prepared_dataset
from src.machine_learning.model_factory import train_with_cv
from config import MODELS_DIR, MODEL_PATH, SCORE_PATH, MODEL_CONFIG


def run_training(model_name=None):
    """
    Executes the training pipeline.
    """
    if model_name is None:
        model_name = MODEL_CONFIG['default_model']

    print(f"--- Starting Pipeline with Model: {model_name} ---")

    # 1. GET DATA
    X_train, y_train = get_prepared_dataset('train')

    # 2. RUN TRAINING (Using 5-Fold CV + Class Weights + Auto-Tuning)
    model, score, threshold = train_with_cv(model_name, X_train, y_train)

    # 3. Feature Importance (Diagnostic)
    print("\n--- Feature Importance (Top 10) ---")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_

        if len(importance) == len(X_train.columns):
            # Store the args
            sort_args = {'key': lambda x: x[1], 'reverse': True}
            data = zip(X_train.columns, importance)
            top_features = sorted(data, **sort_args)[:10]

            # Unpack them using **
            for name, imp in top_features:
                print(f"{name}: {imp:.4f}")

    # 4. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"\nProduction model saved to {MODEL_PATH}")

    # Save Threshold (Critical for predict.py)
    thresh_path = os.path.join(MODELS_DIR, 'threshold.txt')
    with open(thresh_path, 'w') as f:
        f.write(str(threshold))
    print(f"Optimized threshold ({threshold:.2f}) saved to {thresh_path}")

    # Save Archive with Timestamp
    date_str = datetime.now().strftime("%Y-%m-%d")
    archive_filename = f"{model_name}_{date_str}_{score:.4f}.pkl"
    joblib.dump(model, os.path.join(MODELS_DIR, archive_filename))

    # Save Score
    with open(SCORE_PATH, 'w') as f:
        f.write(str(score))
