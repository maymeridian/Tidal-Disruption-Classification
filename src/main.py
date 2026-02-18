'''
main.py
Author: maia.advance, maymeridian
Description: Master script. Orchestrates Tuning, Training, Prediction, and Graphing.
'''

import argparse
import os
import subprocess
import sys
import unittest

from pipeline.train import run_training
from pipeline.predict import run_prediction
from inference.inference_test import TestInference
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


def run_graphing():
    print("\n=== STAGE: GRAPHING ===")
    
    scripts = [
        ("Feature Statistics", "src/graphing/plot_statistics.py"),
        ("Transient Examples", "src/graphing/plot_transients.py"),
        ("Feature Distributions", "src/graphing/plot_features.py")
    ]
    
    for name, path in scripts:
        if os.path.exists(path):
            print(f"--- Running {name} ---")
            subprocess.run([sys.executable, path], check=True)
        else:
            print(f"[!] Missing script: {path}")

    print("\n[✓] All plotting tasks complete.")


def main():
    parser = argparse.ArgumentParser(description="TDE Classifier Pipeline")

    # Actions
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--tune', action='store_true', help="Force re-tuning")
    parser.add_argument('--graph', action='store_true', help="Generate report plots")
    parser.add_argument('--inference', action='store_true', help="Run inference test")

    # Options
    parser.add_argument('--model', type=str, default=MODEL_CONFIG.get('default_model', 'catboost'))
    parser.add_argument('--trials', type=int, default=30, help="Number of trials for Optuna tuning (default: 30)")

    args = parser.parse_args()

    # Tuning
    if args.tune:
        print(f"\n=== STAGE 0: TUNING (FORCED, {args.trials} Trials) ===")
        tune_script = os.path.join("pipeline", "tune.py")
        try:
            subprocess.run([sys.executable, tune_script, '--trials', str(args.trials)], check=True)
        except subprocess.CalledProcessError:
            print("[X] Tuning failed.")

    elif args.train:
        check_and_tune(n_trials=args.trials)

    # Training
    if args.train:
        print("\n=== STAGE 1: TRAINING ===")
        run_training(model_name=args.model)

    # Prediction
    if args.predict:
        print("\n=== STAGE 2: PREDICTION ===")
        run_prediction()

    # Graphing
    if args.graph:
        run_graphing()

    # Inference
    if args.inference:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestInference)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("All tests passed!")
        else:
            print("Some tests failed.")

    # Unknown
    if not any([args.train, args.predict, args.tune, args.graph, args.inference]):
        print("No action specified.")
        print("Usage: python main.py --train --predict --graph")

if __name__ == "__main__":
    main()
