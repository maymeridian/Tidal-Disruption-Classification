'''
src/data_loader.py
Author: maia.advance, maymeridian
Description: Efficient loading of lightcurve data with caching and auto-log detection.
'''

import pandas as pd 
import os
from config import DATA_DIR, TRAIN_LOG_PATH, TEST_LOG_PATH

def load_lightcurves(dataset_type='train', data_dir=DATA_DIR):
    """
    Loads lightcurve data. Automatically determines which log to use based on dataset_type.
    
    Args:
        dataset_type (str): 'train' or 'test'.
        data_dir (str): Root data directory. Defaults to DATA_DIR from config.py.
    """
    
    # 1. Construct the path for the optimized "All" file
    combined_filename = f"{dataset_type}_all_full_lightcurves.csv"
    combined_path = os.path.join(data_dir, combined_filename)

    # 2. Early Return: Check if the combined file already exists
    if os.path.exists(combined_path):
        print(f"Found cached dataset: {combined_path}")
        print("Loading directly (skipping split folders)...")
        return pd.read_csv(combined_path)

    # 3. If not found, we need the log to find the splits.
    print(f"Cached file not found. Building {combined_filename} from splits...")

    if dataset_type == 'train':
        log_path = TRAIN_LOG_PATH
    elif dataset_type == 'test':
        log_path = TEST_LOG_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")

    log_df = pd.read_csv(log_path)
    
    lightcurve_frames = []
    unique_splits = log_df['split'].unique()

    print(f"Processing {len(unique_splits)} split folders for {dataset_type}...")

    for split_name in unique_splits:
        chunk_name = f"{dataset_type}_full_lightcurves.csv"
        chunk_path = os.path.join(data_dir, split_name, chunk_name)

        if os.path.exists(chunk_path):
            df_chunk = pd.read_csv(chunk_path)
            lightcurve_frames.append(df_chunk)
        else:
            print(f"Warning: Chunk file not found {chunk_path}")
    
    if not lightcurve_frames:
        raise FileNotFoundError(f"No lightcurve files were loaded from {data_dir}!")

    # 4. Combine all frames
    combined_df = pd.concat(lightcurve_frames, ignore_index=True)

    # 5. Save the combined file for future runs (Cache it)
    print(f"Saving combined dataset to {combined_path}...")
    combined_df.to_csv(combined_path, index=False)
    print("Save complete.")

    return combined_df