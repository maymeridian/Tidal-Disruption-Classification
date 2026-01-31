'''
src/machine_learning/model_factory.py
Author: maia.advance, maymeridian
Description: Robust ensemble combining physics-guided CatBoost
             models with non-linear neural and neighbor-based support
'''

import numpy as np
import json
import os
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from config import MODEL_CONFIG, MODELS_DIR

# --- FEATURE SUBSETS ---

# Subset 1: Morphology & Temporal Evolution Features
# Focused on the "Shape" of the light curve (Rise/Fade/FWHM).
MORPHOLOGY_FEATURES = [
    'rest_rise_time',
    'rest_fade_time',
    'rest_fwhm',
    'ls_time',
    'rise_fade_ratio',
    'compactness',
    'rise_slope',
    'flux_kurtosis',
    'robust_duration',
    'duty_cycle',
    'pre_peak_var',
    'amplitude'
]

# Subset 2: Physics & Color Metrics
# Focused on the "DNA" of the event (Temperature, Decay Laws, Energy).
PHYSICS_FEATURES = [
    'tde_power_law_error',
    'template_chisq_tde',
    'linear_decay_slope',
    'mean_color_gr',
    'std_color_gr',
    'mean_color_gr_pre',
    'color_slope_gr_pre',
    'blue_energy_fraction',
    'total_radiated_energy',
    'color_monotonicity',
    'negative_flux_fraction',
    'rise_fireball_error',
    'reduced_chi_square',
    'ls_wave',
    'fade_shape_correlation',
    'baseline_ratio',
    'color_cooling_rate',
    'color_slope_gr',
    'flux_ratio_ug',
    'flux_ratio_gr',
    'ug_peak',
    'gr_peak',
    'ur_peak',
    'redshift',
    'absolute_magnitude_proxy',
    'log_tde_error'
]


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted Ensemble Classifier combining Gradient Boosting, NN, and KNN.
    Designed to balance the high accuracy of Trees with the
    generalization of geometric and neural models.
    """
    def __init__(self, scale_pos_weight=1.0):
        self.scale_pos_weight = scale_pos_weight
        self.models = {}
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Trains the constituent models of the ensemble.
        """
        seed = MODEL_CONFIG['random_seed']

        # --- GRADIENT BOOSTING PARAMETERS ---
        # Optimized for stability (lower learning rate, depth 5)
        cb_params = {
            'iterations': 1000, 'depth': 5, 'learning_rate': 0.02,
            'l2_leaf_reg': 10, 'rsm': 0.5, 'loss_function': 'Logloss',
            'verbose': 0, 'allow_writing_files': False,
            'random_seed': seed, 'scale_pos_weight': self.scale_pos_weight
        }

        # Load optimized hyperparameters if available (from external tuning)
        json_path = os.path.join(MODELS_DIR, 'best_params.json')

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    tuned = json.load(f)

                if 'scale_pos_weight' in tuned:
                    del tuned['scale_pos_weight']  # Handled dynamically

                cb_params.update(tuned)
            except Exception:
                pass

        # 1. Base Model (CatBoost - All Features)
        # The primary driver of performance.
        self.models['base'] = CatBoostClassifier(**cb_params)
        self.models['base'].fit(X, y)
        self.feature_importances_ = self.models['base'].feature_importances_

        # 2. Morphology Sub-Model (CatBoost - Shape Features)
        # Forces model to look at shape, even if physics features are strong.
        cols_morph = [c for c in MORPHOLOGY_FEATURES if c in X.columns]

        if cols_morph:
            self.models['morphology'] = CatBoostClassifier(**cb_params)
            self.models['morphology'].fit(X[cols_morph], y)

        # 3. Physics Sub-Model (CatBoost - Physics Features)
        # Forces model to validate against physical laws (TDE power law, etc.).
        cols_phys = [c for c in PHYSICS_FEATURES if c in X.columns]
        if cols_phys:
            self.models['physics'] = CatBoostClassifier(**cb_params)
            self.models['physics'].fit(X[cols_phys], y)

        # 4. Support Model A: Multi-Layer Perceptron (Neural Network)
        # Captures non-linear interactions missed by tree splits.
        mlp_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'max_iter': 600,
            'random_state': seed
        }

        self.models['mlp'] = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('net', MLPClassifier(**mlp_params))
        ])
        self.models['mlp'].fit(X, y)

        # 5. Support Model B: K-Nearest Neighbors
        # Manifold learning: identifies objects clustered near TDEs.
        knn_clf = KNeighborsClassifier(n_neighbors=15, weights='distance', p=2)

        self.models['knn'] = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('knn', knn_clf)
        ])

        self.models['knn'].fit(X, y)

        return self

    def predict_proba(self, X):
        """
        Generates weighted probability estimates.
        """
        # 1. Generate Component Predictions
        p_base = self.models['base'].predict_proba(X)[:, 1]

        p_morph = p_base  # Fallback
        if 'morphology' in self.models:
            cols = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
            p_morph = self.models['morphology'].predict_proba(X[cols])[:, 1]

        p_phys = p_base  # Fallback
        if 'physics' in self.models:
            cols = [c for c in PHYSICS_FEATURES if c in X.columns]
            p_phys = self.models['physics'].predict_proba(X[cols])[:, 1]

        p_mlp = self.models['mlp'].predict_proba(X)[:, 1]
        p_knn = self.models['knn'].predict_proba(X)[:, 1]

        # 2. Ensemble Aggregation (Weighted Average)
        # Weights logic:
        # - Gradient Boosting (80% Total): Split 48% Base, 16% Morph, 16% Phys.
        #   (Base is the strongest, but subsets provide regularization).
        # - Support Models (20% Total): Split 10% MLP, 10% KNN.
        #   (Provides diversity to stabilize predictions on noisy data).

        final_prob = (0.48 * p_base) + \
                     (0.16 * p_morph) + \
                     (0.16 * p_phys) + \
                     (0.10 * p_mlp) + \
                     (0.10 * p_knn)

        return np.vstack([1 - final_prob, final_prob]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]

        return (probs >= 0.5).astype(int)


# --- FACTORY ---
def get_model(model_name, scale_pos_weight=1.0):
    if model_name == 'catboost':
        return EnsembleClassifier(scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError("Only 'catboost' is supported.")


def train_with_cv(model_name, X, y):
    """
    Performs 5-Fold Stratified Cross-Validation to validate model stability
    and optimize the decision threshold before final training.
    """
    print("\n--- Initializing Hybrid Ensemble Classifier ---")
    print("    Configuration: Gradient Boosting (Core) + MLP/KNN (Support)")

    seed = MODEL_CONFIG['random_seed']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = []
    best_thresholds = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Calculate dynamic class weights for imbalance handling
        n_pos = y_train.sum()
        scale_weight = (len(y_train) - n_pos) / n_pos if n_pos > 0 else 1.0

        model = get_model(model_name, scale_pos_weight=scale_weight)
        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)[:, 1]

        # Threshold Optimization
        best_f1 = 0.0
        best_t = 0.5

        for t in np.arange(0.2, 0.8, 0.02):
            s = f1_score(y_val, (probs_val >= t).astype(int), zero_division=0)

            if s > best_f1:
                best_f1 = s
                best_t = t

        val_tdes = y_val.sum()
        print(f"   Fold {fold}: F1={best_f1:.4f} (Thresh={best_t:.2f}) "
              f"- Validation Positives: {val_tdes}")

        cv_scores.append(best_f1)
        best_thresholds.append(best_t)
        fold += 1

    avg_f1 = np.mean(cv_scores)
    avg_thresh = np.mean(best_thresholds)

    print(f"\n   Average Ensemble F1: {avg_f1:.4f}")

    # Final Training on 100% of Data
    print("\n--- Training Final Production Model ---")

    n_pos_all = y.sum()
    final_weight = (len(y) - n_pos_all) / n_pos_all

    final_model = get_model(model_name, scale_pos_weight=final_weight)
    final_model.fit(X, y)

    return final_model, avg_f1, avg_thresh
