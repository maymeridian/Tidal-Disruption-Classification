'''
predict.py
Author: maia.advance, maymeridian
Description: Generates predictions and saves a formatted submission file with the date and F1 score.
'''

import pandas as pd
import os
import joblib
from datetime import datetime

from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features
from config import DATA_DIR, MODEL_PATH, SCORE_PATH, SUBMISSIONS_DIR, TEST_LOG

def main():
    # 1. Load Test Log
    print("Loading Test Log...")

    log_path = os.path.join(DATA_DIR, TEST_LOG)
    test_log = pd.read_csv(log_path)

    # 2. Load Test Lightcurves
    lc_df = load_lightcurves(test_log, dataset_type='test')

    # 3. Preprocessing (Must match train.py)
    lc_df = apply_deextinction(lc_df, test_log)

    # 4. Feature Engineering 
    features_df = extract_features(lc_df, test_log)

    # 5. Load Model 
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train.py first!")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 6. Predict (ensure columns match exactly what the model expects (drop object_id)
    X_test = features_df.drop(columns=['object_id'])

    print("Generating predictions...")
    predictions = model.predict(X_test)

    # 7. Create Submission DataFrame
    submission = pd.DataFrame({'object_id': features_df['object_id'], 'prediction': predictions})

    # Include all objects from test_log, even if some had no lightcurve data
    final_submission = test_log[['object_id']].merge(submission, on='object_id', how='left')
        
    # Fill missing predictions with 0 (safe assumption for rare events like TDEs)
    final_submission['prediction'] = final_submission['prediction'].fillna(0).astype(int)
        
    # 8. Construct Output Filename
    f1_score_str = "0.000"

    if os.path.exists(SCORE_PATH):
        with open(SCORE_PATH, 'r') as f:
            # Read score and format to 4 decimal places just in case
            raw_score = float(f.read().strip())
            f1_score_str = f"{raw_score:.4f}"
    else:
        print(f"Warning: Score file not found at {SCORE_PATH}. Using 0.000 in filename.")

    # Get current date formatted as YYYY-MM-DD
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Create the filename: submission_2023-10-27_0.8543.csv
    filename = f"submission_{date_str}_{f1_score_str}.csv"
    
    # Ensure the submissions directory exists
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    output_path = os.path.join(SUBMISSIONS_DIR, filename)

    # Save submission to folder
    final_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()