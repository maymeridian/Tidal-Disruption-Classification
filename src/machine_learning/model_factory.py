'''
src/machine_learning/model_factory.py
Author: maia.advance, maymeridian
Description: Factory pattern for initializing machine learning models.
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from config import MODEL_CONFIG

def get_model(model_name, y_train=None):
    """
    Returns an initialized model based on the model_name string.
    
    Args:
        model_name (str): Name of the model to initialize.
        y_train (pd.Series): Training labels. REQUIRED for calculating class weights.
    """
    print(f"Initializing model: {model_name}")

    seed = MODEL_CONFIG['random_seed']

    # Calculate Scale Weight for XGBoost (Ratio of Negative / Positive)
    scale_pos_weight = 1.0

    if y_train is not None:
        num_neg = len(y_train) - y_train.sum()
        num_pos = y_train.sum()
        
        if num_pos > 0:
            scale_pos_weight = num_neg / num_pos
            print(f"   -> Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)

    elif model_name == 'xgboost':
        return XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.09, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=seed)
        
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")