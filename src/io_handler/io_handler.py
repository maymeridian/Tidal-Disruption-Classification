'''
src/io_handler/io_handler.py
Author: maia.advance, maymeridian
Description: Efficient loading of lightcurve data with caching and automated dataset preparation.
'''

import pandas as pd
import os

from machine.features import extract_features, get_log_data
from config import DATA_DIR, PROCESSED_TRAINING_DATA_PATH, PROCESSED_TESTING_DATA_PATH


def load_features(dataset_type='train'):
    """
    Single source of truth for loading processed features.
    If features do not exist, it automatically triggers the extraction pipeline.
    """
    if dataset_type == 'train':
        feature_path = PROCESSED_TRAINING_DATA_PATH
    elif dataset_type == 'test':
        feature_path = PROCESSED_TESTING_DATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # 1. Check if we need to run the heavy extraction
    if not os.path.exists(feature_path):
        print(f"Features not found at {feature_path}. Initiating extraction...")

        lc_df = load_lightcurves(dataset_type=dataset_type)
        df = extract_features(lc_df, dataset_type=dataset_type) 
    else:
        df = pd.read_csv(feature_path)

    # 2. Load metadata using the helper in features.py
    meta = get_log_data(dataset_type)

    # 3. Standardize ID column
    if 'object' in meta.columns: 
        meta.rename(columns={'object': 'object_id'}, inplace=True)
    if 'object' in df.columns: 
        df.rename(columns={'object': 'object_id'}, inplace=True)

    # 4. Merge target labels (mostly for training/graphing data)
    if dataset_type == 'train' and 'target' not in df.columns:
        targets = meta[['object_id', 'target']].drop_duplicates()
        df = df.merge(targets, on='object_id', how='inner')

    return df, meta


def load_lightcurves(dataset_type='train', data_dir=DATA_DIR):
    """
    Loads raw lightcurve data. Automatically determines which log to use based on dataset_type.
    Uses caching to avoid re-stitching split files on every run.
    """
    combined_filename = f"combined_curves/{dataset_type}_all_full_lightcurves.csv"
    combined_path = os.path.join(data_dir, combined_filename)

    if os.path.exists(combined_path):
        print(f"Found cached dataset: {combined_path}")
        return pd.read_csv(combined_path)

    print(f"Cached file not found. Building {combined_filename} from splits...")
    log_df = get_log_data(dataset_type)

    lightcurve_frames = []
    unique_splits = log_df['split'].unique()

    for split_name in unique_splits:
        chunk_name = f"{dataset_type}_full_lightcurves.csv"
        chunk_path = os.path.join(data_dir, split_name, chunk_name)

        if os.path.exists(chunk_path):
            df_chunk = pd.read_csv(chunk_path)
            lightcurve_frames.append(df_chunk)

    if not lightcurve_frames:
        raise FileNotFoundError(f"No lightcurve files were loaded from {data_dir}!")

    combined_df = pd.concat(lightcurve_frames, ignore_index=True)
    combined_df.to_csv(combined_path, index=False)
    return combined_df


def get_prepared_dataset(dataset_type='train'):
    """
    Prepares the final numeric dataset specifically for Machine Learning models.
    """
    if dataset_type == 'train':
        print("--- Preparing Training Dataset ---")
        full_df, _ = load_features(dataset_type='train')
        X = full_df.drop(columns=['object_id', 'target'])
        y = full_df['target']
        return X, y
        
    elif dataset_type == 'test':
        print("--- Preparing Testing Dataset ---")
        full_df, _ = load_features(dataset_type='test')
        X = full_df.drop(columns=['object_id'])
        ids = full_df['object_id']
        return X, ids
    
    return None