'''
src/model_factory.py
Author: maia.advance, maymeridian
Description: 
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from config import RANDOM_SEED

def get_model(model_name):
    """
    Returns an initialized model based on the model_name string.
    """
    print(f"Initializing model: {model_name}")

    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state = RANDOM_SEED)
    
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state = RANDOM_SEED)

    elif model_name == 'xgboost':
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state = RANDOM_SEED)

    elif model_name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=10, class_weight='balanced', random_state = RANDOM_SEED)
        
    # elif model_name == 'muon_optimizer':

    else:
        raise ValueError(f"Model '{model_name}' not recognized in model_factory.py")