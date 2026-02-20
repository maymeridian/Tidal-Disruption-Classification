import pandas as pd
import os
import sys

from config import PROCESSED_TRAINING_DATA_PATH, TRAIN_LOG_PATH, TRAIN_DATA_PATH, PLOTS_DIR
from graphing.plot_transients import generate_transient_plots
from graphing.plot_features import generate_feature_plots

def load_processed_features():
    print(f"   -> Loading Features from {os.path.basename(PROCESSED_TRAINING_DATA_PATH)}...")
    try:
        df = pd.read_csv(PROCESSED_TRAINING_DATA_PATH)
        meta = pd.read_csv(TRAIN_LOG_PATH)
        
        if 'object' in meta.columns: 
            meta.rename(columns={'object': 'object_id'}, inplace=True)
        if 'object' in df.columns: 
            df.rename(columns={'object': 'object_id'}, inplace=True)
        
        if 'target' not in df.columns:
            targets = meta[['object_id', 'target']].drop_duplicates()
            df = df.merge(targets, on='object_id', how='inner')
            
        df['Class'] = df['target'].apply(lambda x: 'TDE' if str(x) in ['1', '1.0', 'TDE'] else 'Other')
        return df
    except Exception as e:
        print(f"[!] Feature Load Error: {e}")
        sys.exit(1)

def load_lightcurves():
    print(f"   -> Loading Light Curves from {os.path.basename(TRAIN_DATA_PATH)}...")
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        meta = pd.read_csv(TRAIN_LOG_PATH)
        
        if 'object' in df.columns: 
            df.rename(columns={'object': 'object_id'}, inplace=True)
        if 'object' in meta.columns: 
            meta.rename(columns={'object': 'object_id'}, inplace=True)

        flux_map = {
            'Flux': 'flux', 'flux_Jy': 'flux', 'mag': 'flux', 'magnitude': 'flux',
            'Flux_err': 'flux_err', 'flux_err_Jy': 'flux_err', 'mag_err': 'flux_err',
            'Time (MJD)': 'mjd', 'time': 'mjd', 'MJD': 'mjd',
            'Filter': 'filter', 'band': 'filter', 'passband': 'filter'
        }
        df.rename(columns=flux_map, inplace=True)

        if 'flux' not in df.columns:
            print(f"[!] CRITICAL: Could not find 'flux' column. Available: {list(df.columns)}")
            sys.exit(1)

        # Merge Labels AND SpecType so plotting scripts don't need to load meta again
        cols_to_merge = ['object_id', 'target']
        if 'SpecType' in meta.columns:
            cols_to_merge.append('SpecType')
            
        targets = meta[cols_to_merge].drop_duplicates()
        df = df.merge(targets, on='object_id', how='inner')
        return df
    except Exception as e:
        print(f"[!] Lightcurve Load Error: {e}")
        sys.exit(1)

def ensure_output_dir(subfolder=None):
    path = PLOTS_DIR
    if subfolder:
        path = os.path.join(PLOTS_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def run_graphing():
    print("\n=== STAGE 3: GRAPHING ===")
    
    try:
        # 1. Load data ONCE into RAM
        df_curves = load_lightcurves()
        df_feats = load_processed_features()
        
        # 2. Setup Directories
        transient_dir = ensure_output_dir("transient_examples")
        anatomy_dir = ensure_output_dir("feature_anatomy")
        dist_dir = ensure_output_dir("feature_distributions")
        
        # 3. Pass data down to the graphing functions
        print("--- Generating Transient Examples ---")
        generate_transient_plots(df_curves, transient_dir)
        
        print("--- Generating Feature & Anatomy Plots ---")
        generate_feature_plots(df_curves, df_feats, anatomy_dir, dist_dir)
        
        print("\n[✓] All plotting tasks complete.")
    except Exception as e:
        print(f"\n[X] Graphing failed: {e}")