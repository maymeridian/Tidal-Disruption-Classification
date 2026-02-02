import pandas as pd
import os
import sys

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'datasets', 'processed_data', 'processed_training_data.csv')
LOG_PATH = os.path.join(ROOT_DIR, 'datasets', 'log_data', 'train_log.csv')
CURVE_PATH = os.path.join(ROOT_DIR, 'datasets', 'combined_curves', 'train_all_full_lightcurves.csv')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'plots')

def load_processed_features():
    """Loads feature matrix and merges with targets."""
    print(f"   -> Loading Features from {os.path.basename(DATA_PATH)}...")
    try:
        df = pd.read_csv(DATA_PATH)
        meta = pd.read_csv(LOG_PATH)
        
        # Standardize 'object_id'
        if 'object' in meta.columns: 
            meta.rename(columns={'object': 'object_id'}, inplace=True)
        if 'object' in df.columns: 
            df.rename(columns={'object': 'object_id'}, inplace=True)
        
        # Merge Labels
        if 'target' not in df.columns:
            targets = meta[['object_id', 'target']].drop_duplicates()
            df = df.merge(targets, on='object_id', how='inner')
            
        df['Class'] = df['target'].apply(lambda x: 'TDE' if str(x) in ['1', '1.0', 'TDE'] else 'Other')
        return df
    except Exception as e:
        print(f"[!] Feature Load Error: {e}")
        sys.exit(1)

def load_lightcurves():
    """Loads raw light curves and standardizes column names."""
    print(f"   -> Loading Light Curves from {os.path.basename(CURVE_PATH)}...")
    try:
        df = pd.read_csv(CURVE_PATH)
        meta = pd.read_csv(LOG_PATH)
        
        # 1. Standardize 'object_id'
        if 'object' in df.columns: 
            df.rename(columns={'object': 'object_id'}, inplace=True)
        if 'object' in meta.columns: 
            meta.rename(columns={'object': 'object_id'}, inplace=True)

        # 2. Standardize 'flux' (Handle 'Flux', 'flux_Jy', etc.)
        flux_map = {
            'Flux': 'flux', 'flux_Jy': 'flux', 'mag': 'flux', 'magnitude': 'flux',
            'Flux_err': 'flux_err', 'flux_err_Jy': 'flux_err', 'mag_err': 'flux_err',
            'Time (MJD)': 'mjd', 'time': 'mjd', 'MJD': 'mjd',
            'Filter': 'filter', 'band': 'filter', 'passband': 'filter'
        }
        df.rename(columns=flux_map, inplace=True)

        # Check if it worked
        if 'flux' not in df.columns:
            print(f"[!] CRITICAL: Could not find 'flux' column. Available: {list(df.columns)}")
            sys.exit(1)

        # 3. Merge Labels
        targets = meta[['object_id', 'target']].drop_duplicates()
        df = df.merge(targets, on='object_id', how='inner')
        return df
    except Exception as e:
        print(f"[!] Lightcurve Load Error: {e}")
        sys.exit(1)

def ensure_output_dir(subfolder=None):
    path = OUTPUT_DIR
    if subfolder:
        path = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    return path