'''
src/main.py
Author: maia.advance, maymeridian
Description: Master script. Orchestrates Tuning, Training, Prediction, Inference, and Graphing.
'''

import argparse
import sys
import unittest

from pipeline.train import run_training
from pipeline.predict import run_prediction
from pipeline.tune import run_tuning
from graphing.plot import run_graphing
from inference.inference_test import TestInference
from config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description="TDE Classifier Pipeline")

    # Actions
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--tune', action='store_true', help="Force re-tuning")
    parser.add_argument('--plot', action='store_true', help="Generate report plots")
    parser.add_argument('--inference', action='store_true', help="Run inference test")

    # Options
    parser.add_argument('--model', type=str, default=MODEL_CONFIG.get('default_model', 'catboost'))
    parser.add_argument('--trials', type=int, default=30, help="# of trials for Optuna (default: 30)")

    args = parser.parse_args()

    # Tuning
    if args.tune:
        run_tuning(n_trials=args.trials, force=True)
    elif args.train:
        run_tuning(n_trials=args.trials, force=False)

    # Training
    if args.train:
        print("\n=== STAGE 1: TRAINING ===")
        run_training(model_name=args.model)

    # Prediction
    if args.predict:
        print("\n=== STAGE 2: PREDICTION ===")
        run_prediction()

    # Graphing
    if args.plot:
        run_graphing()

    # Inference
    if args.inference:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestInference)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if not result.wasSuccessful():
            print("Some tests failed.")
            sys.exit(1)
        else:
            print("All tests passed!")

    # Unknown
    if not any([args.train, args.predict, args.tune, args.plot, args.inference]):
        print("No action specified.")
        print("Usage: python main.py --train --predict")

if __name__ == "__main__":
    main()
