'''
src/machine_learning/predict.py
Author: maia.advance, maymeridian
Description: Generates predictions using the saved model and optimized threshold.
'''

import pandas as pd
import os
import joblib
from datetime import datetime

from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features
from config import DATA_DIR, MODEL_PATH, SCORE_PATH, RESULTS_DIR, TEST_LOG_PATH

def run_prediction():
    """
    Executes the prediction pipeline using the saved model at MODEL_PATH.
    Applies the optimized threshold found during training.
    """
    # 1. Load Test Log
    print("Loading Test Log...")
    test_log = pd.read_csv(TEST_LOG_PATH)

    # 2. Load Test Lightcurves
    lc_df = load_lightcurves(dataset_type='test')

    # 3. Preprocessing
    lc_df = apply_deextinction(lc_df, test_log)

    # 4. Feature Engineering 
    features_df = extract_features(lc_df, test_log)

    # 5. Load Model 
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run training first!")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 6. Load Optimized Threshold
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

    # 7. Predict
    X_test = features_df.drop(columns=['object_id'])
    print("Generating predictions...")
    
    # Get Probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Apply Threshold
    y_preds = (y_probs >= threshold).astype(int)

    # 8. Create Submission DataFrame
    submission = pd.DataFrame({
        'object_id': features_df['object_id'], 
        'prediction': y_preds,
        'probability': y_probs  # Useful for debugging
    })

    # Include all objects from test_log (fill missing with 0)
    final_submission = test_log[['object_id']].merge(submission, on='object_id', how='left')
    final_submission['prediction'] = final_submission['prediction'].fillna(0).astype(int)
        
    # 9. Construct Output Filename
    f1_score_str = "0.000"
    if os.path.exists(SCORE_PATH):
        with open(SCORE_PATH, 'r') as f:
            raw_score = float(f.read().strip())
            f1_score_str = f"{raw_score:.4f}"

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"submission_{date_str}_{f1_score_str}.csv"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, filename)

    # Save only required columns for submission (usually just ID and prediction)
    # Adjust columns if the challenge requires probability
    final_submission[['object_id', 'prediction']].to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    
    # Diagnostic: How many TDEs did we find?
    num_found = final_submission['prediction'].sum()
    print(f"Total TDEs predicted in Test Set: {num_found}")