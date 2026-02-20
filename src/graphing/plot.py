'''
pipeline/plot.py
Author: maia.advance, maymeridian
Description: Master graphing script. Orchestrates data loading and generates transient, anatomy, and feature distribution plots.
'''

import pandas as pd
import os
import sys

from config import PLOTS_DIR, TRAIN_LOG_PATH
from graphing.plot_transients import generate_transient_plots
from graphing.plot_features import generate_feature_plots

# Import the raw data loaders from your single source of truth
from io_handler.io_handler import load_lightcurves as io_load_lightcurves
from io_handler.io_handler import load_features as io_load_features


def load_graphing_features():
    """Loads features and formats them specifically for plotting legends."""
    print("   -> Formatting Features for Graphing...")
    try:
        # Pull from the single source of truth
        df, _ = io_load_features(dataset_type='train')
            
        # Add the string label for the plots
        df['Class'] = df['target'].apply(lambda x: 'TDE' if str(x) in ['1', '1.0', 'TDE'] else 'Other')
        return df
    except Exception as e:
        print(f"[!] Feature Formatting Error: {e}")
        sys.exit(1)


def load_graphing_lightcurves():
    """Uses io_handler to get raw data, then standardizes columns for seaborn/matplotlib."""
    print("   -> Formatting Lightcurves for Graphing...")
    try:
        # Pull from the single source of truth
        df = io_load_lightcurves(dataset_type='train')
        meta = pd.read_csv(TRAIN_LOG_PATH)
        
        if 'object' in df.columns: 
            df.rename(columns={'object': 'object_id'}, inplace=True)
        if 'object' in meta.columns: 
            meta.rename(columns={'object': 'object_id'}, inplace=True)

        # Standardize columns for graphing scripts
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

        # Merge Labels AND SpecType for the plot titles
        cols_to_merge = ['object_id', 'target']
        if 'SpecType' in meta.columns:
            cols_to_merge.append('SpecType')
            
        targets = meta[cols_to_merge].drop_duplicates()
        df = df.merge(targets, on='object_id', how='inner')
        return df
    except Exception as e:
        print(f"[!] Lightcurve Formatting Error: {e}")
        sys.exit(1)


def ensure_output_dir(subfolder=None):
    """Safely creates output directories inside the main plots folder."""
    path = PLOTS_DIR
    if subfolder:
        path = os.path.join(PLOTS_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def run_graphing():
    """Master orchestrator for the graphing stage."""
    print("\n=== STAGE 3: GRAPHING ===")
    
    try:
        # 1. Load formatted data ONCE
        df_curves = load_graphing_lightcurves()
        df_feats = load_graphing_features()
        
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