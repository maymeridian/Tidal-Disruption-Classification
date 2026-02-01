import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
import scipy.special
from catboost import CatBoostClassifier, Pool
from config import MODEL_CONFIG
from src.data_loader import get_prepared_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from IPython.display import clear_output
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import copy
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.stats import rankdata
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=1150
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-8
BATCH_SIZE = 64
EMBED_DIMENSION = 64
SEED = 42


# For Grid search : 

# 1. Sigmoid
# f(x) = 1 / (1 + e^-x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_activation(act_cls, dim=None):
    """Instantiates activation function, handling dim args if needed."""
    try:
        return act_cls(dim)
    except TypeError:
        return act_cls()
    except Exception:
        return nn.ReLU()


def clean_data_robust(df):

    # 1. Fix redshift_err if present
    if 'redshift_err' in df.columns:
        df['redshift_err'] = pd.to_numeric(df['redshift_err'], errors='coerce')
        df['redshift_err'] = df['redshift_err'].replace([np.inf, -np.inf], np.nan)
        
        # If the whole column is NaN, fill with 0.0 instead of dropping
        if df['redshift_err'].isna().all():
            print("Cleaning: 'redshift_err' is empty. Filling with 0.0 (Preserving column for model).")
            df['redshift_err'] = 0.0
        else:
            fill_val = df['redshift_err'].mean()
            fill_val = 0.0 if pd.isna(fill_val) else fill_val
            df['redshift_err'] = df['redshift_err'].fillna(fill_val)
            
    return df
# ==========================================
# 2. ACTIVATION FUNCTIONS
# ==========================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SReLU(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.t_l = nn.Parameter(torch.zeros(in_features))
        self.a_l = nn.Parameter(torch.zeros(in_features))
        self.t_r = nn.Parameter(torch.zeros(in_features))
        self.a_r = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        out = torch.where(x <= self.t_l, self.t_l + self.a_l * (x - self.t_l), x)
        out = torch.where(out >= self.t_r, self.t_r + self.a_r * (out - self.t_r), out)
        return out

class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)

# ==========================================
# 3. LOSS FUNCTIONS (Fixed)
# ==========================================

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight_value=1.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight_value]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        return self.criterion(logits, targets.float())

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        y_true = targets.float()
        tp = (probs * y_true).sum(dim=0)
        fp = (probs * (1 - y_true)).sum(dim=0)
        fn = ((1 - probs) * y_true).sum(dim=0)
        f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow((1 - tversky), self.gamma)

class LogitAdjustmentLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        prior = torch.tensor(cls_num_list).float()
        self.prior = prior / prior.sum()
        self.tau = tau
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Requires (B, 2) logits
        if logits.shape[1] == 1:
            # Trick to convert binary logits to (B,2) for CE
            logits = torch.cat([-logits, logits], dim=1)
            
        log_prior = torch.log(self.prior + 1e-12).to(logits.device)
        adjusted_logits = logits + self.tau * log_prior
        return self.criterion(adjusted_logits, targets.long())

def get_loss_function(name, counts=[100, 100]):
    name = name.lower()
    if name == 'bce': return nn.BCEWithLogitsLoss()
    if name == 'weighted_bce': return WeightedBCELoss(pos_weight_value=counts[0]/(counts[1]+1e-5))
    if name == 'focal': return FocalLoss()
    if name == 'soft_f1': return SoftF1Loss()
    if name == 'focal_tversky': return FocalTverskyLoss()
    if name == 'logit_adj': return LogitAdjustmentLoss(counts)
    return nn.BCEWithLogitsLoss()

# ==========================================
# 4. MODEL ARCHITECTURES
# ==========================================

# --- A. Tabular ResNet ---
class ResBlock(nn.Module):
    def __init__(self, dim, dropout, act_cls):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = get_activation(act_cls, dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = get_activation(act_cls, dim)

    def forward(self, x):
        res = x
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x + res

class TabularResNet(nn.Module):
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64, num_blocks=3, dropout=0.2, act_cls=nn.ReLU):
        super().__init__()
        # Basic handling: Treat everything as numeric for ResNet 
        # (Assuming embeddings are pre-flattened or data is purely numeric for this impl)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout, act_cls) for _ in range(num_blocks)])
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_bn(x)
        x = self.output_layer(x)
        return x

# --- B. Deep & Cross Network (DCN) ---
class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x0, xi):
        # x_{l+1} = x0 * (xi . w + b) + xi
        x_l_w = torch.sum(xi * self.weight, dim=1, keepdim=True)
        return x0 * x_l_w + self.bias + xi

class DeepCrossNet(nn.Module):
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64, num_cross=3, num_deep=3, act_cls=nn.ReLU):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross)])
        
        deep_layers = []
        in_d = input_dim
        for _ in range(num_deep):
            deep_layers.append(nn.Linear(in_d, hidden_dim))
            deep_layers.append(nn.BatchNorm1d(hidden_dim))
            deep_layers.append(get_activation(act_cls, hidden_dim))
            deep_layers.append(nn.Dropout(0.2))
            in_d = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)
        self.final = nn.Linear(input_dim + hidden_dim, 1)

    def forward(self, x):
        x0 = x
        xi = x
        for layer in self.cross_layers:
            xi = layer(x0, xi)
        xd = self.deep_net(x0)
        combined = torch.cat([xi, xd], dim=1)
        return self.final(combined)

# --- C. FT-Transformer (Simplified) ---
class FTTransformer(nn.Module):
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64, depth=3, heads=4, dropout=0.2, act_cls=nn.ReLU):
        super().__init__()
        # Note: This is a simplified version expecting numerical inputs. 
        # Full FT-Transformer requires per-feature embeddings. 
        # Here we project the whole input to d_model for simplicity in this specific pipeline.
        self.feature_projector = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=heads, 
            dim_feedforward=hidden_dim*2, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # B, Features -> B, 1, D
        x = self.feature_projector(x).unsqueeze(1)
        
        # Prepend CLS token
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Take CLS token output
        return self.final(x[:, 0, :])
        

class TimeScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim * 32)
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.bn = nn.BatchNorm1d(32)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32 * 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x_emb = self.proj(x).view(x.shape[0], self.input_dim, 32).permute(0, 2, 1)
        c1 = self.act(self.conv1(x_emb))
        c2 = self.act(self.conv2(x_emb))
        c4 = self.act(self.conv4(x_emb))
        p1 = self.global_pool(c1).squeeze(-1)
        p2 = self.global_pool(c2).squeeze(-1)
        p4 = self.global_pool(c4).squeeze(-1)
        return self.head(self.dropout(torch.cat([p1, p2, p4], dim=1)))

def train_predict_nn(X_train, y_train, X_val, X_test):
    input_dim = X_train.shape[1]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_train.values))
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = TimeScaleCNN(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    
    best_loss = float('inf')
    patience = 5
    counter = 0
    model.train()
    for epoch in range(2500): 
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
            
    model.eval()
    with torch.no_grad():
        val_preds = torch.sigmoid(model(torch.FloatTensor(X_val_s).to(device))).cpu().numpy().flatten()
        test_preds = torch.sigmoid(model(torch.FloatTensor(X_test_s).to(device))).cpu().numpy().flatten()
    return val_preds, test_preds

# ==========================================
# 2. ULTIMATE BLENDER (Extended)
# ==========================================

class UltimateBlender:
    def __init__(self, catboost_path):
        self.catboost_path = catboost_path
        self.anchor_params = None
        self.S_train = {} # Val preds
        self.S_test = {}  # Test preds
        self.y_val = None

    def load_anchor_params(self):
        loaded = joblib.load(self.catboost_path)
        self.anchor_params = loaded.get_params()

    # --- TUNING HELPERS ---
    def tune_xgboost(self, X_tr, y_tr, X_va, y_va, trials=30):
        print(f"  > Tuning XGBoost ({trials} trials) [Estimators up to 2000]...")
        def objective(trial):
            param = {
                # INCREASED RANGE
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5, 25),
                'booster': 'dart',
                'n_jobs': -1, 'random_state': 42, 'verbosity': 0
            }
            model = xgb.XGBClassifier(**param)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            return roc_auc_score(y_va, preds)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        return xgb.XGBClassifier(**study.best_params)

    def tune_lightgbm(self, X_tr, y_tr, X_va, y_va, trials=30):
        print(f"  > Tuning LightGBM ({trials} trials) [Estimators up to 2000]...")
        def objective(trial):
            param = {
                # INCREASED RANGE
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'class_weight': 'balanced',
                'extra_trees': True,
                'n_jobs': -1, 'random_state': 42, 'verbose': -1
            }
            model = lgb.LGBMClassifier(**param)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            return roc_auc_score(y_va, preds)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        return study.best_params

    # --- NEW: RF TUNER ---
    def tune_rf(self, X_tr, y_tr, X_va, y_va, trials=50):
        print(f"  > Tuning Random Forest ({trials} trials)...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': 'balanced',
                'n_jobs': -1, 'random_state': 42
            }
            model = RandomForestClassifier(**param)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            return roc_auc_score(y_va, preds)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        return RandomForestClassifier(**study.best_params)

    # --- EXECUTION ---
    def run_training(self, X_train, y_train, X_val, y_val, X_test):
        self.y_val = y_val
        
        # 1. ANCHOR (CV BAGGING)
        print("\n--- 1. Anchor (CatBoost CV Bagging) ---")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        anchor_val = np.zeros(len(X_val))
        anchor_test = np.zeros(len(X_test))
        
        for fold, (idx_tr, _) in enumerate(skf.split(X_train, y_train)):
            X_fold, y_fold = X_train.iloc[idx_tr], y_train.iloc[idx_tr]
            model = CatBoostClassifier(**self.anchor_params)
            model.fit(X_fold, y_fold, verbose=0)
            anchor_val += model.predict_proba(X_val)[:, 1] / 5
            anchor_test += model.predict_proba(X_test)[:, 1] / 5
            
        self.S_train['Anchor'] = anchor_val
        self.S_test['Anchor'] = anchor_test
        print(f"  Anchor F1 (Val): {self.get_f1(anchor_val):.4f}")

        print("\n--- 2. XGBoost (Deep Search) ---")
        xgb_model = self.tune_xgboost(X_train, y_train, X_val, y_val)
        xgb_model.fit(X_train, y_train)
        self.S_train['XGB'] = xgb_model.predict_proba(X_val)[:, 1]
        self.S_test['XGB'] = xgb_model.predict_proba(X_test)[:, 1]
        print(f"  XGB F1 (Val): {self.get_f1(self.S_train['XGB']):.4f}")

        print("\n--- 3. LightGBM (Deep Search) ---")
        lgb_params = self.tune_lightgbm(X_train, y_train, X_val, y_val)
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        self.S_train['LGB'] = lgb_model.predict_proba(X_val)[:, 1]
        self.S_test['LGB'] = lgb_model.predict_proba(X_test)[:, 1]
        print(f"  LGB F1 (Val): {self.get_f1(self.S_train['LGB']):.4f}")

        # 4. NEURAL NET
        print("\n--- 4. TimeScaleCNN ---")
        nn_val, nn_test = train_predict_nn(X_train, y_train, X_val, X_test)
        self.S_train['NN'] = nn_val
        self.S_test['NN'] = nn_test
        print(f"  NN F1 (Val): {self.get_f1(nn_val):.4f}")

        # 5. RANDOM FOREST (TUNED + FIXED PRINTING)
        print("\n--- 5. Random Forest (Tuned) ---")
        rf_model = self.tune_rf(X_train, y_train, X_val, y_val)
        rf_model.fit(X_train, y_train)
        self.S_train['RF'] = rf_model.predict_proba(X_val)[:, 1]
        self.S_test['RF'] = rf_model.predict_proba(X_test)[:, 1]
        # FIXED: Now printing score
        print(f"  RF F1 (Val): {self.get_f1(self.S_train['RF']):.4f}")

    def get_f1(self, probs, t=0.5):
        best = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            score = f1_score(self.y_val, (probs > thresh).astype(int))
            if score > best: best = score
        return best

    def optimize_and_submit(self, submission_ids):
        print("\n--- Stacking & Threshold Optimization ---")
        X_stack_val = pd.DataFrame(self.S_train)
        X_stack_test = pd.DataFrame(self.S_test)
        
        def objective(trial):
            C = trial.suggest_float("C", 0.001, 10.0, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            model = LogisticRegression(C=C, class_weight='balanced', solver=solver, max_iter=1500)
            
            # Inner CV
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            try:
                for t_idx, v_idx in skf.split(X_stack_val, self.y_val):
                    model.fit(X_stack_val.iloc[t_idx], self.y_val.iloc[t_idx])
                    preds = model.predict_proba(X_stack_val.iloc[v_idx])[:, 1]
                    scores.append(roc_auc_score(self.y_val.iloc[v_idx], preds))
                return np.mean(scores)
            except:
                return 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        
        print(f"Best Meta Params: {study.best_params}")
        meta = LogisticRegression(**study.best_params, class_weight='balanced', max_iter=1500)
        meta.fit(X_stack_val, self.y_val)
        
        val_probs = meta.predict_proba(X_stack_val)[:, 1]
        test_probs = meta.predict_proba(X_stack_test)[:, 1]
        
        # --- FINE-GRAINED THRESHOLD ---
        best_t = 0.5
        best_f1 = 0.0
        # Increased precision (10x steps)
        for t in np.arange(0.01, 0.99, 0.0001):
            score = f1_score(self.y_val, (val_probs > t).astype(int))
            if score > best_f1:
                best_f1 = score
                best_t = t
                
        print(f"\n>>> FINAL OPTIMIZED F1: {best_f1:.4f} (at threshold {best_t:.4f})")
        print(f">>> FINAL AUC: {roc_auc_score(self.y_val, val_probs):.4f}")

        submission = pd.DataFrame({
            'object_id': submission_ids,
            'probability': test_probs,
            'predicted_class': (test_probs > best_t).astype(int)
        })
        return submission

if __name__ == "__main__":
    from src.data_loader import get_prepared_dataset
    
    X, y = get_prepared_dataset('train')
    X = clean_data_robust(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    X_test, test_ids = get_prepared_dataset('test')
    X_test = clean_data_robust(X_test)
    X_test = X_test[X_train.columns]

    MODEL_PATH = "/home/maia-adv/Downloads/catboost_2026-01-29_0.7329.pkl"
    
    blender = UltimateBlender(MODEL_PATH)
    blender.load_anchor_params()
    blender.run_training(X_train, y_train, X_val, y_val, X_test)
    
    sub = blender.optimize_and_submit(test_ids)
    sub[['object_id', 'probability']].to_csv("submission_final_tuned.csv", index=False)
    print("Saved submission_final_tuned.csv")