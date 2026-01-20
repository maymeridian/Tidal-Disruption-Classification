'''
config.py
Author: maia.advance, maymeridian
Description: 
'''

import os

#############################
# --- DIRECTORY PATHS ---
#############################

# Automatically finds the root folder of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

#############################
# --- FILE PATHS ---
#############################

# The model file itself
MODEL_PATH = os.path.join(MODELS_DIR, 'tde_classifier.pkl')

# Metadata file to pass the F1 score from train.py to predict.py
SCORE_PATH = os.path.join(MODELS_DIR, 'latest_score.txt')

# Log file names (expected to be inside DATA_DIR)
TRAIN_LOG = 'train_log.csv'
TEST_LOG = 'test_log.csv'

#############################
# --- SCIENCE CONSTANTS ---
#############################

# Effective wavelengths (Angstroms) for LSST filters (u, g, r, i, z, y)
FILTER_WAVELENGTHS = {
    'u': 3641, 
    'g': 4704, 
    'r': 6155, 
    'i': 7504, 
    'z': 8695, 
    'y': 10056
}

#############################
# --- MODEL CONFIGURATION ---
#############################

# Options: 'logistic_regression', 'random_forest'
CURRENT_MODEL_NAME = 'logistic_regression'
RANDOM_SEED = 42
TEST_SIZE = 0.2