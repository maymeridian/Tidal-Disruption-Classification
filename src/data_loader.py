'''
src/data_loader.py
Author: maia.advance, maymeridian
Description: efficient loading of lightcurve data with caching.
'''

import pandas as pd 
import os
from config import DATA_DIR

def load_lightcurves(log_df, dataset_type='train', data_dir=DATA_DIR):
    """
    Loads lightcurve data. 
    
    OPTIMIZATION:
    1. Checks if a combined file (e.g., 'train_all_full_lightcurves.csv') exists in data_dir.
    2. If YES: Loads and returns that file immediately (Fast).
    3. If NO: Iterates through split folders, combines them, saves the combined file to disk, 
       and then returns the data.
    
    Args:
        log_df (pd.DataFrame): The train or test log containing 'split' info.
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

    # 3. If not found, build it from the split folders
    print(f"Cached file not found. Building {combined_filename} from splits...")
    
    lightcurve_frames = []
    unique_splits = log_df['split'].unique()

    print(f"Processing {len(unique_splits)} split folders for {dataset_type}...")

    for split_name in unique_splits:
        # Path: data/split_##/train_full_lightcurves.csv
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