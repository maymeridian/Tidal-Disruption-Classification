'''
src/machine_learning/train.py
Author: maia.advance, maymeridian
Description: Training pipeline. Clean, focused, and integrated with data_loader.
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
    X, y = get_prepared_dataset('train')

    # 2. RUN TRAINING (Using 5-Fold CV + Class Weights)
    model, score, threshold = train_with_cv(model_name, X, y)

    # 3. Feature Importance (Diagnostic)
    print("\n--- Feature Importance (Top 10) ---")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        if len(importance) == len(X.columns):
            for name, imp in sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)[:10]:
                print(f"{name}: {imp:.4f}")

    # 4. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"\nProduction model saved to {MODEL_PATH}")

    # Save Threshold (Critical for predict.py)
    thresh_path = os.path.join(os.path.dirname(SCORE_PATH), 'threshold.txt')
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