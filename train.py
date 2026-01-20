'''
train.py
Author: maia.advance, maymeridian
Description: 
'''

import pandas as pd
import os
import joblib
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features
from src.model_factory import get_model
from config import DATA_DIR, MODELS_DIR, MODEL_PATH, SCORE_PATH, TRAIN_LOG, CURRENT_MODEL_NAME, RANDOM_SEED, TEST_SIZE

def main():
    # 0. Argument Parsing
    parser = argparse.ArgumentParser(description="Train TDE Classifier")
    parser.add_argument('--model', type=str, default=CURRENT_MODEL_NAME, help="Model type to train (e.g., 'random_forest', 'logistic_regression')")
    args = parser.parse_args()
    
    # Use the argument if provided, otherwise fall back to config default
    selected_model_name = args.model
    print(f"--- Starting Pipeline with Model: {selected_model_name} ---")

    # 1. Load Log Data
    print("Loading Train Log...")

    log_path = os.path.join(DATA_DIR, TRAIN_LOG)
    train_log = pd.read_csv(log_path)

    # 2. Load Lightcurves
    lc_df = load_lightcurves(train_log, dataset_type='train')

    # 3. Preprocessing
    lc_df = apply_deextinction(lc_df, train_log)

    # 4. Feature Engineering
    features_df = extract_features(lc_df, train_log)

    # 5. Merge with Target Labels
    full_df = features_df.merge(train_log[['object_id', 'target']], on='object_id')

    # 6. Prepare X and y
    X = full_df.drop(columns=['object_id', 'target'])
    y = full_df['target']

    # 7. Split & Train
    print("Training Model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_SEED)

    # Get model from factory using the SELECTED name
    try:
        model = get_model(selected_model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    model.fit(X_train, y_train)
    
    # 8. Evaluate
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred)

    print(f"\nValidation F1 Score ({selected_model_name}): {score:.4f}")
    print(classification_report(y_val, y_pred))

    # 9. Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save Production Model
    joblib.dump(model, MODEL_PATH)
    print(f"Latest model saved to {MODEL_PATH}")

    # Save Archive Model
    date_str = datetime.now().strftime("%Y-%m-%d")
    archive_filename = f"{selected_model_name}_{date_str}_{score:.4f}.pkl"
    archive_path = os.path.join(MODELS_DIR, archive_filename)
    
    joblib.dump(model, archive_path)
    print(f"Archived model saved to {archive_path}")

    # 10. Save Score
    with open(SCORE_PATH, 'w') as f:
        f.write(str(score))

if __name__ == "__main__":
    main()