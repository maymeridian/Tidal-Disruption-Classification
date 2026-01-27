'''
src/machine_learning/predict.py
Author: maia.advance, maymeridian
Description: Generates predictions using the saved model and optimized threshold.
             Updated to use centralized data loading via get_prepared_dataset.
'''

import pandas as pd
import os
import joblib
from datetime import datetime

from src.data_loader import get_prepared_dataset 
from config import MODEL_PATH, SCORE_PATH, RESULTS_DIR, TEST_LOG_PATH

def run_prediction():
    """
    Executes the prediction pipeline using the saved model at MODEL_PATH.
    Applies the optimized threshold found during training.
    """
    
    # 1. Load Test Features & IDs
    print("Loading Test Lightcurves...")
    X_test, ids = get_prepared_dataset('test')

    # 2. Load Model 
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run training first!")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 3. Load Optimized Threshold
    thresh_path = os.path.join(os.path.dirname(SCORE_PATH), 'threshold.txt')
    threshold = 0.5 # Default fallback
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            try:
                threshold = float(f.read().strip())
                print(f"Loaded optimized decision threshold: {threshold}")
            except ValueError:
                print("Warning: Could not read threshold file. Using default 0.5.")
    else:
        print("Warning: Threshold file not found. Using default 0.5.")

    # 4. Predict
    if hasattr(model, "feature_names_"):
        missing_cols = set(model.feature_names_) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[model.feature_names_] # Reorder to match model

    print("Generating predictions...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Apply Threshold
    y_preds = (y_probs >= threshold).astype(int)

    # 5. Create Partial Submission DataFrame
    submission_partial = pd.DataFrame({'object_id': ids, 'prediction': y_preds, 'probability': y_probs})

    # 6. Merge with Full Test Log
    print("Finalizing submission file...")
    test_log = pd.read_csv(TEST_LOG_PATH)

    # Left merge ensures every object in the original test set is present
    final_submission = test_log[['object_id']].merge(submission_partial, on='object_id', how='left')
    
    # Fill NaN (objects we filtered out) with 0
    final_submission['prediction'] = final_submission['prediction'].fillna(0).astype(int)
    final_submission['probability'] = final_submission['probability'].fillna(0.0)

    # 7. Construct Output Filename
    f1_score_str = "0.000"
    if os.path.exists(SCORE_PATH):
        with open(SCORE_PATH, 'r') as f:
            raw_score = float(f.read().strip())
            f1_score_str = f"{raw_score:.4f}"

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"submission_{date_str}_{f1_score_str}.csv"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, filename)

    # Save final submission
    final_submission[['object_id', 'prediction']].to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    
    # Diagnostic: How many TDEs did we find?
    num_found = final_submission['prediction'].sum()
    print(f"Total TDEs predicted in Test Set: {num_found}")