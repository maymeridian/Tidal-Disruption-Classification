'''
main.py
Author: maia.advance, maymeridian
Description: Master script to orchestrate training and prediction pipelines.
'''

import argparse

from src.machine_learning.train import run_training
from src.machine_learning.predict import run_prediction
from config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description="TDE Classifier Pipeline Orchestrator")
    
    # Action arguments
    parser.add_argument('--train', action='store_true', help="Run the training pipeline")
    parser.add_argument('--predict', action='store_true', help="Run the prediction pipeline")
    
    # Configuration arguments
    parser.add_argument('--model', type=str, default=MODEL_CONFIG['default_model'], 
                        help=f"Model to use (default: {MODEL_CONFIG['default_model']})")

    args = parser.parse_args()

    # Logic to handle user requests
    if args.train:
        print("\n=== STAGE 1: TRAINING ===")
        run_training(model_name=args.model)
        
    if args.predict:
        print("\n=== STAGE 2: PREDICTION ===")
        run_prediction()

    if not args.train and not args.predict:
        print("No action specified.")
        print("Usage Example: python main.py --train --predict")

if __name__ == "__main__":
    main()