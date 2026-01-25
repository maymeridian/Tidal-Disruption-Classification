'''
src/machine_learning/train.py
Author: maia.advance, maymeridian
Description: Training pipeline for TDE Classifier using K-Fold CV and SMOTE.
'''

import pandas as pd
import os
import joblib
from datetime import datetime

from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features
from src.machine_learning.model_factory import train_with_cv 
from config import DATA_DIR, MODELS_DIR, MODEL_PATH, SCORE_PATH, TRAIN_LOG_PATH, MODEL_CONFIG

def run_training(model_name=None):
    """
    Executes the training pipeline.
    Uses model_factory.train_with_cv to handle SMOTE and Cross-Validation logic.
    """
    if model_name is None:
        model_name = MODEL_CONFIG['default_model']

    print(f"--- Starting Pipeline with Model: {model_name} ---")

    # 1. Load & Prep Data
    print("Loading Data...")
    train_log = pd.read_csv(TRAIN_LOG_PATH)
    lc_df = load_lightcurves(dataset_type='train')
    
    # Preprocessing
    lc_df = apply_deextinction(lc_df, train_log)
    features_df = extract_features(lc_df, train_log)
    full_df = features_df.merge(train_log[['object_id', 'target']], on='object_id')

    X = full_df.drop(columns=['object_id', 'target'])
    y = full_df['target']

    # 2. RUN TRAINING (Using 5-Fold CV + SMOTE)
    # This delegates the imbalance handling to the factory
    model, score, threshold = train_with_cv(model_name, X, y)

    # 3. Feature Importance (Diagnostic)
    print("\n--- Feature Importance (Top 10) ---")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
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