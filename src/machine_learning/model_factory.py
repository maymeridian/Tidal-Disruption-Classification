'''
src/model_factory.py
Author: maia.advance, maymeridian
Description: Factory pattern for initializing machine learning models.
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Import the configuration dictionary
from config import MODEL_CONFIG

def get_model(model_name):
    """
    Returns an initialized model based on the model_name string.
    Uses hyperparameters defined in config.py.
    """
    print(f"Initializing model: {model_name}")

    # Extract the seed from the config dictionary
    seed = MODEL_CONFIG['random_seed']

    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
    
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=seed)

    elif model_name == 'xgboost':
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=seed)

    elif model_name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=10, class_weight='balanced', random_state=seed)
        
    else:
        raise ValueError(f"Model '{model_name}' not recognized in src/model_factory.py")