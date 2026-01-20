'''
src/data_loader.py
Author: maia.advance, maymeridian
Description: 
'''

import pandas as pd 
import os
from config import DATA_DIR

def load_lightcurves(log_df, dataset_type='train', data_dir=DATA_DIR):
    """
    Iterate through the data/split_## folders defined in log_df.
    
    Args:
        log_df (pd.DataFrame): The train or test log containing 'split' info.
        dataset_type (str): 'train' or 'test'.
        data_dir (str): Root data directory. Defaults to DATA_DIR from config.py.
    """ 
    lightcurve_frames = []

    unique_splits = log_df['split'].unique()

    print(f"Loading {dataset_type} lightcurves from {len(unique_splits)} split folders...")

    for split_name in unique_splits:
        # Path: data/split_##/train_full_lightcurves.csv
        file_name = f"{dataset_type}_full_lightcurves.csv"
        file_path = os.path.join(data_dir, split_name, file_name)

        if os.path.exists(file_path):
            df_chunk = pd.read_csv(file_path)
            lightcurve_frames.append(df_chunk)
        else:
            print(f"Warning: File not found {file_path}")
    
    if not lightcurve_frames:
        raise FileNotFoundError(f"No lightcurve files were loaded from {data_dir}!")

    return pd.concat(lightcurve_frames, ignore_index=True)