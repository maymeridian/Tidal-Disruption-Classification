'''
main.py
Author: maia.advance, maymeridian
Description: Master script. Orchestrates Tuning, Training, and Prediction.
'''

import argparse
import os
import subprocess
import sys

from src.machine_learning.train import run_training
from src.machine_learning.predict import run_prediction
from config import MODEL_CONFIG, MODELS_DIR

def check_and_tune(n_trials=30):
    """
    Checks if best_params.json exists. If not, runs tune.py.
    """
    params_path = os.path.join(MODELS_DIR, 'best_params.json')
    
    if not os.path.exists(params_path):
        print("\n[!] No optimized parameters found.")
        print(f"--- Initiating Auto-Tuning ({n_trials} trials) ---")
        
        tune_script = os.path.join("src", "machine_learning", "tune.py")
        try:
            # Pass the trial count to the script
            subprocess.run([sys.executable, tune_script, '--trials', str(n_trials)], check=True)
            print("\n[✓] Tuning Complete.")
        except subprocess.CalledProcessError:
            print("\n[X] Tuning Failed. Falling back to default parameters.")

def main():
    parser = argparse.ArgumentParser(description="TDE Classifier Pipeline")
    
    # Actions
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--tune', action='store_true', help="Force re-tuning")
    
    # Options
    parser.add_argument('--model', type=str, default=MODEL_CONFIG.get('default_model', 'catboost'))
    parser.add_argument('--trials', type=int, default=30, help="Number of trials for Optuna tuning (default: 30)")

    args = parser.parse_args()

    print(f"--- Starting Pipeline with Model: {args.model} ---")

    # 1. TUNING
    if args.tune:
        print(f"\n=== STAGE 0: TUNING (FORCED, {args.trials} Trials) ===")
        tune_script = os.path.join("src", "machine_learning", "tune.py")
        try:
            subprocess.run([sys.executable, tune_script, '--trials', str(args.trials)], check=True)
        except subprocess.CalledProcessError:
            print("[X] Tuning failed.")
            
    elif args.train:
        check_and_tune(n_trials=args.trials)

    # 2. TRAINING
    if args.train:
        print("\n=== STAGE 1: TRAINING ===")
        run_training(model_name=args.model)
        
    # 3. PREDICTION
    if args.predict:
        print("\n=== STAGE 2: PREDICTION ===")
        run_prediction()

    if not args.train and not args.predict and not args.tune:
        print("No action specified.")
        print("Usage: python main.py --train --predict")

if __name__ == "__main__":
    main()