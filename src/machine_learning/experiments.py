
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, auc, log_loss, precision_recall_curve
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import MODEL_CONFIG
from src.machine_learning.model_factory import MORPHOLOGY_FEATURES, PHYSICS_FEATURES
from src.data_loader import get_prepared_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = MODEL_CONFIG['random_seed']


def get_activation(act_cls, dim=None):
    """Instantiates activation function, handling dim args if needed."""
    try:
        return act_cls(dim)
    except TypeError:
        return act_cls()
    except Exception:
        return nn.ReLU()


def clean_data(df):

    if 'redshift_err' in df.columns:
        df['redshift_err'] = pd.to_numeric(df['redshift_err'], errors='coerce')
        df['redshift_err'] = df['redshift_err'].replace([np.inf,
                                                        -np.inf], np.nan)

        # if the whole column is NaN, then drop
        if df['redshift_err'].isna().all():
            df = df.drop(columns=['redshift_err'])
        else:
            fill_val = df['redshift_err'].mean()
            fill_val = 0.0 if pd.isna(fill_val) else fill_val
            df['redshift_err'] = df['redshift_err'].fillna(fill_val)

    return df

# --------------------
# ACTIVATION FUNCTIONS
# --------------------

# 1. Sigmoid
# f(x) = 1 / (1 + e^-x)

# 2. Gaussian
class GaussianActivation(nn.Module):
    def __init__(self, init_mu=0.0, init_sigma=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        return torch.exp(-((x - self.mu) / self.sigma) ** 2)


# 3. ReLU
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x)
        # or F.relu(x) or x.clamp(min=0)


# 4. Leaky ReLU
# f(x) = x if x > 0, else alpha * x
# Allows small negative gradient to prevent "dying ReLU"
class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)
        # or F.leaky_relu(x, negative_slope.alpha)


# 5. ELU - Exponential Linear Unit
class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


# 6. SELU - Scaled Exponential Linear Unit
# (Klambauer et al., 2017)
#  SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
class SELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        return self.scale * torch.where(
            x > 0,
            x,
            self.alpha * (torch.exp(x) - 1)
        )


# 7. PReLU - Parametric ReLU
# Like Leaky ReLU but alpha is learned during training
class PreLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_parameters) * init)

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)


# 8. S-Shaped ReLU (SReLU)
# (Jin et al., 2016)
# Piecewise linear function that learns the thresholds and slopes.
class SReLU(nn.Module):
    def __init__(self, in_features):
        super(SReLU, self).__init__()
        # 4 learnable parameters per channel
        self.t_l = nn.Parameter(torch.zeros(in_features))  # left threshold
        self.a_l = nn.Parameter(torch.zeros(in_features))  # left slope
        self.t_r = nn.Parameter(torch.zeros(in_features))  # right threshold
        self.a_r = nn.Parameter(torch.ones(in_features))
        #     ^  right slope (1 being identity)

    def forward(self, x):
        # Piecewise logic:
        # x <= t_l: t_l + a_l * (x - t_l)
        # t_l < x < t_r: x
        # x >= t_r: t_r + a_r * (x - t_r)

        out = torch.where(x <= self.t_l, self.t_l +
                          self.a_l * (x - self.t_l), x)

        out = torch.where(out >= self.t_r, self.t_r +
                          self.a_r * (out - self.t_r), out)
        return out


# 9. APLU - Adaptive Piecewise Linear Unit
# (Agostinelli et al., 2014)
class APLU(nn.Module):
    def __init__(self, in_features, S=2):
        super().__init__()
        self.S = S
        self.a = nn.Parameter(torch.randn(S, in_features) * 0.1)
        self.b = nn.Parameter(torch.randn(S, in_features) * 0.1)

    def forward(self, x):
        # f(x) = max(0, x) + sum(a_i * max(0, b_i - x))
        out = F.relu(x)
        for i in range(self.S):
            out += self.a[i] * F.relu(self.b[i] - x)
        return out


# 10. 'Mexican ReLU'
# https://www.researchgate.net/figure/Mexican-hat-type-activation-functions-3_fig1_268386570
class MexicanHat(nn.Module):
    def __init__(self):
        super().__init__()
        # (1 - x^2) * exp(-x^2 / 2)

    def forward(self, x):
        x_sq = x**2
        return (1 - x_sq) * torch.exp(-x_sq / 2)

# https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
# 11. Mish: Self Regularized Non-Monotonic Neural Activation Function
# -------------------------
# f(x) = x * tanh(sigma(x))
# where sigma(x) = ln(1 + e^x) is the softplus activation function
# -------------------------


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# https://arxiv.org/pdf/1710.05941v1
# 12. Swish - Smooth Non-Monotonic Activation Function
# a "self-gated" activation function
# -------------------------
# f(x) = x * sigma(x)
# where sigma is (1 + exp(-x))^-1
# -------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# https://arxiv.org/pdf/1606.08415
# 13. GELU: Gaussian Error Linear Units
# -------------------------
# G(x) = xP(X <= x) = xPhi(x) = x * (1/2) [ 1 + erf( x/sqrt(2) ) ]
# approximated by :
#  0.5x(1 + tanh[(sqrt(2/pi) * (x + 0.044715x^3))])   or   x*sigma(1.702x)
# -------------------------
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# https://arxiv.org/abs/1907.06732
# 14. PADÉ Activation Units
class PADE(nn.Module):
    def __init__(self, m=2, n=2):
        super().__init__()
        self.m = m
        self.n = n
        self.numerator_coeffs = nn.Parameter(torch.randn(m+1) * 0.1)
        self.denominator_coeffs = nn.Parameter(torch.randn(n) * 0.1)

    def forward(self, x):
        numerator = torch.zeros_like(x)
        for i, coeff in enumerate(self.numerator_coeffs):
            numerator += coeff * (x**i)

        denominator = torch.ones_like(x)
        for i, coeff in enumerate(self.denominator_coeffs):
            denominator += coeff * (x ** (i+1))

        return numerator / (denominator + 1e-12)


# normally used for finding periodics, this is just a funny addition
class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)


# --------------
# Loss Functions
# --------------

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
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                      reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean() \
            if self.reduction == 'mean' else focal_loss.sum()


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

        tversky = (tp + self.smooth) / (tp + self.alpha * fp +
                                        self.beta * fn + self.smooth)

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


def get_loss_function(name, counts=[100, 100], pos_weight=None):
    name = name.lower()
    if name == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == 'weighted_bce':
        return WeightedBCELoss(pos_weight_value=counts[0]/(counts[1]+1e-5))
    if name == 'focal':
        return FocalLoss()
    if name == 'soft_f1':
        return SoftF1Loss()
    if name == 'focal_tversky':
        return FocalTverskyLoss()
    if name == 'logit_adj':
        return LogitAdjustmentLoss(counts)

    return nn.BCEWithLogitsLoss()


# -------------------
# Model Architectures
# -------------------

# Tabular ResNet
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
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64,
                 num_blocks=3, dropout=0.2, act_cls=nn.ReLU):

        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout, act_cls)
                                    for _ in range(num_blocks)])

        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_bn(x)
        x = self.output_layer(x)
        return x


# Deep & Cross Network (DCN)
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
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64, num_cross=3,
                 num_deep=3, act_cls=nn.ReLU):
        super().__init__()

        self.cross_layers = nn.ModuleList([CrossLayer(input_dim)
                                           for _ in range(num_cross)])

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


# FT-Transformer (simplified)
class FTTransformer(nn.Module):
    def __init__(self, input_dim, cat_dims=[], hidden_dim=64, depth=3,
                 heads=4, dropout=0.2, act_cls=nn.ReLU):
        super().__init__()

        # we project the whole input to d_model for simplicity.
        self.feature_projector = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # B, features -> B, 1, D
        x = self.feature_projector(x).unsqueeze(1)

        # prepend CLS token
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        # take CLS token output
        return self.final(x[:, 0, :])


class TimeScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 dropout=0.1, act_cls=nn.ReLU):
        super().__init__()

        self.proj = nn.Linear(input_dim, input_dim * 32)
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.bn = nn.BatchNorm1d(32)

        self.act = get_activation(act_cls, 32)

        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32 * 3, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x):
        x_emb = self.proj(x).view(x.shape[0],
                                  self.input_dim,
                                  32).permute(0, 2, 1)

        c1 = self.act(self.conv1(x_emb))
        c2 = self.act(self.conv2(x_emb))
        c4 = self.act(self.conv4(x_emb))
        p1 = self.global_pool(c1).squeeze(-1)
        p2 = self.global_pool(c2).squeeze(-1)
        p4 = self.global_pool(c4).squeeze(-1)
        return self.head(self.dropout(torch.cat([p1, p2, p4], dim=1)))


class Experiment:
    def __init__(self, X_train, y_train, X_val, y_val, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test

        # standard Scaling
        self.scaler = StandardScaler()
        self.X_train_s = self.scaler.fit_transform(X_train)
        self.X_val_s = self.scaler.transform(X_val)
        self.X_test_s = self.scaler.transform(X_test)

        # quantile transformation
        self.qt = QuantileTransformer(output_distribution='normal',
                                      random_state=SEED)
        self.X_train_qt = self.qt.fit_transform(X_train)
        self.X_val_qt = self.qt.transform(X_val)
        self.X_test_qt = self.qt.transform(X_test)

        # store predictions for ensemble
        self.preds_val = {}
        self.preds_test = {}
        self.best_lgbm_params = None
        self.optimized_weights = None

    # CNN
    def tune_cnn_optuna(self, n_trials=20):
        print(f"\ntuning time-scale CNN [{n_trials} trials]")

        pos_count = self.y_train.sum()
        neg_count = len(self.y_train) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)]).to(device)

        def objective(trial):
            # Hyperparams
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)

            act_name = trial.suggest_categorical("act", ["ReLU", "LeakyReLU",
                                                 "GELU", "Mish", "Swish"])
            loss_name = trial.suggest_categorical("loss", ["bce", "focal"])

            act_map = {
                "ReLU": nn.ReLU, "LeakyReLU": LeakyReLU, "GELU": GELU,
                "Mish": Mish, "Swish": Swish
            }
            act_cls = act_map[act_name]

            input_dim = self.X_train.shape[1]

            model = TimeScaleCNN(
                input_dim=input_dim,
                dropout=dropout,
                act_cls=act_cls
            ).to(device)

            optimizer = AdamW(model.parameters(), lr=lr)
            criterion = get_loss_function(loss_name,
                                          counts=[neg_count, pos_count],
                                          pos_weight=pos_weight).to(device)

            # Data Loaders (Using Standard Scaled here, but QT is an option)
            train_ds = TensorDataset(torch.FloatTensor(self.X_train_s),
                                     torch.FloatTensor(self.y_train.values))
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

            model.train()
            for epoch in range(50):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb).squeeze()
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(torch.FloatTensor(self.X_val_s)
                                   .to(device)).squeeze()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()

            score = roc_auc_score(self.y_val, val_probs)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("best CNN params:", study.best_params)

        p = study.best_params
        act_map = {"ReLU": nn.ReLU, "LeakyReLU": LeakyReLU, "GELU": GELU,
                   "Mish": Mish, "Swish": Swish}
        best_model = TimeScaleCNN(
            input_dim=self.X_train.shape[1],
            dropout=p['dropout'],
            act_cls=act_map[p['act']]
        ).to(device)

        opt = AdamW(best_model.parameters(), lr=p['lr'])
        crit = get_loss_function(p['loss'],
                                 counts=[neg_count, pos_count],
                                 pos_weight=pos_weight).to(device)


        train_ds = TensorDataset(torch.FloatTensor(self.X_train_s),
                                 torch.FloatTensor(self.y_train.values))
        loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        best_model.train()
        for epoch in range(100):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = best_model(xb).squeeze()
                loss = crit(out, yb)
                loss.backward()
                opt.step()

        best_model.eval()
        with torch.no_grad():
            self.preds_val['CNN'] = \
                torch.sigmoid(best_model(torch.FloatTensor(self.X_val_s)
                                         .to(device)).squeeze()).cpu().numpy()
            self.preds_test['CNN'] = \
                torch.sigmoid(best_model(torch.FloatTensor(self.X_test_s)
                                         .to(device)).squeeze()).cpu().numpy()

        print(f"CNN final AUC: {roc_auc_score(self.y_val,
                                                  self.preds_val['CNN']):.4f}")

    # Catboost <3
    def tune_catboost_optuna(self, n_trials=20):
        print(f"\ntuning CatBoost [{n_trials} trials]")

        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'random_strength': trial.suggest_int('random_strength', 0, 100),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 1, 255),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2, 30),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'verbose': 0,
                'eval_metric': 'AUC',
                'random_state': SEED,
                'allow_writing_files': False
            }

            model = CatBoostClassifier(**param)
            model.fit(self.X_train, self.y_train, verbose=0)
            preds = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("best CatBoost params:", study.best_params)
        best_cb = CatBoostClassifier(**study.best_params, verbose=0, random_state=SEED, allow_writing_files=False)
        best_cb.fit(self.X_train, self.y_train)
        
        self.preds_val['CatBoost'] = best_cb.predict_proba(self.X_val)[:, 1]
        self.preds_test['CatBoost'] = best_cb.predict_proba(self.X_test)[:, 1]
        print(f"CatBoost final AUC: {roc_auc_score(self.y_val, self.preds_val['CatBoost']):.4f}")

    # XGBoost
    def tune_xgboost_optuna(self, n_trials=20):
        print(f"\ntuning XGBoost [{n_trials} trials]")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': 0
            }
            model = xgb.XGBClassifier(**param)
            model.fit(self.X_train, self.y_train)
            preds = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("best XGBoost params:", study.best_params)
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=SEED, n_jobs=-1, verbosity=0)
        best_xgb.fit(self.X_train, self.y_train)

        self.preds_val['XGBoost'] = best_xgb.predict_proba(self.X_val)[:, 1]
        self.preds_test['XGBoost'] = best_xgb.predict_proba(self.X_test)[:, 1]
        print(f"XGBoost final AUC: {roc_auc_score(self.y_val, self.preds_val['XGBoost']):.4f}")

    # LGBM
    def tune_lgbm_optuna(self, n_trials=30):
        print(f"\ntuning LGBM [{n_trials} trials]")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'verbose': -1,
                'random_state': SEED,
                'n_jobs': -1
            }
            model = lgb.LGBMClassifier(**param)
            model.fit(self.X_train, self.y_train)
            preds = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        self.best_lgbm_params = study.best_params

        print("  best LGBM params:", study.best_params)
        best_lgbm = lgb.LGBMClassifier(**study.best_params, verbose=-1,
                                       random_state=SEED, n_jobs=-1)

        best_lgbm.fit(self.X_train, self.y_train)

        self.preds_val['LGBM'] = best_lgbm.predict_proba(self.X_val)[:, 1]
        self.preds_test['LGBM'] = best_lgbm.predict_proba(self.X_test)[:, 1]
        print(f"LGBM final AUC: \
               {roc_auc_score(self.y_val, self.preds_val['LGBM']):.4f}")

    # KNN tuning
    def tune_knn_optuna(self, n_trials=15):
        print(f"\ntuning KNN [{n_trials} trials]")

        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 5, 100)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 2)

            model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                         weights=weights, p=p, n_jobs=-1)
            model.fit(self.X_train_qt, self.y_train)
            preds = model.predict_proba(self.X_val_qt)[:, 1]
            return roc_auc_score(self.y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("best KNN params:", study.best_params)
        best_knn = KNeighborsClassifier(**study.best_params, n_jobs=-1)
        best_knn.fit(self.X_train_qt, self.y_train)

        self.preds_val['KNN'] = best_knn.predict_proba(self.X_val_qt)[:, 1]
        self.preds_test['KNN'] = best_knn.predict_proba(self.X_test_qt)[:, 1]
        print(f"KNN final AUC: {roc_auc_score(self.y_val, self.preds_val['KNN']):.4f}")

    # AdaBoostRegressor
    def tune_adaregressor_optuna(self, n_trials=20):
        print(f"\ntuning AdaBoost [{n_trials} trials]")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                'random_state': SEED
            }
            # AdaBoostRegressor fits on 0/1
            model = AdaBoostRegressor(**param)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_val)
            return roc_auc_score(self.y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("best Ada params:", study.best_params)
        best_ada = AdaBoostRegressor(**study.best_params, random_state=SEED)
        best_ada.fit(self.X_train, self.y_train)

        # clip predictions to [0, 1] for probability
        v_preds = np.clip(best_ada.predict(self.X_val), 0, 1)
        t_preds = np.clip(best_ada.predict(self.X_test), 0, 1)

        self.preds_val['AdaReg'] = v_preds
        self.preds_test['AdaReg'] = t_preds
        print(f"Ada final AUC: \
               {roc_auc_score(self.y_val, self.preds_val['AdaReg']):.4f}")

    # specialist models
    def train_specialists(self):
        print("\ntraining specialists on morphological & physical...")

        # use best params from LGBM tuning if available, else defaults
        if self.best_lgbm_params is not None:
            params = self.best_lgbm_params.copy()
            params.update({'verbose': -1, 'random_state': SEED, 'n_jobs': -1})
        else:
            params = {
                'n_estimators': 1250,
                'learning_rate': 0.01,
                'num_leaves': 31,
                'verbose': -1,
                'random_state': SEED,
                'n_jobs': -1,
            }

        specialists = {
            'Morphological': MORPHOLOGY_FEATURES,
            'Physical': PHYSICS_FEATURES
        }

        for name, feats in specialists.items():
            # filter features that exist in the dataset
            valid_feats = [f for f in feats if f in self.X_train.columns]

            print(f"  Training {name} Specialist on \
                   {len(valid_feats)} features")

            # subset data
            X_tr_sub = self.X_train[valid_feats]
            X_val_sub = self.X_val[valid_feats]
            X_te_sub = self.X_test[valid_feats]

            # train LGBM
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr_sub, self.y_train)

            # predict
            preds_val = model.predict_proba(X_val_sub)[:, 1]
            preds_test = model.predict_proba(X_te_sub)[:, 1]

            self.preds_val[f'Spec_{name}'] = preds_val
            self.preds_test[f'Spec_{name}'] = preds_test

            print(f"{name} specialist AUC: \
                   {roc_auc_score(self.y_val, preds_val):.4f}")

    # apply isotonic regression
    def apply_isotonic_calibration(self):
        print("\napplying isotonic regression calibration...")

        if not self.preds_val:
            print("no predictions found, skipping calibration.")
            return

        best_name = max(self.preds_val, key=lambda k:
                        roc_auc_score(self.y_val, self.preds_val[k]))
        print(f"calibrating best model: {best_name}")

        val_p = self.preds_val[best_name]
        test_p = self.preds_test[best_name]

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_p, self.y_val)

        cal_val = iso.transform(val_p)
        cal_test = iso.transform(test_p)

        self.preds_val[f'{best_name}_Iso'] = cal_val
        self.preds_test[f'{best_name}_Iso'] = cal_test

        print(f"pre-calibration AUC: {roc_auc_score(self.y_val, val_p):.4f}")
        print(f"post-calibration AUC: {roc_auc_score(self.y_val, cal_val):.4f}")

    def optimize_weights_hill_climbing(self, iterations=1000):
        print("\noptimizing ensemble weights")

        models = list(self.preds_val.keys())
        if not models:
            return

        n_models = len(models)
        weights = {m: 1.0 / n_models for m in models}

        # calculate initial best
        def get_f1_score(w_dict):
            # Weighted average
            final_pred = np.zeros_like(next(iter(self.preds_val.values())))
            for m, w in w_dict.items():
                final_pred += self.preds_val[m] * w

            # efficient F1 calculation for all thresholds
            precisions, recalls, thresholds = precision_recall_curve(self.y_val, final_pred)
            # F1 = 2*P*R / (P+R)
            numerator = 2 * precisions * recalls
            denominator = precisions + recalls + 1e-12
            f1_scores = numerator / denominator

            # return max F1 and the best threshold found
            best_idx = np.argmax(f1_scores)

            return f1_scores[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        current_best_score, _ = get_f1_score(weights)
        print(f"initial F1: {current_best_score:.4f}")

        # coordinate descent
        step = 0.05
        patience = 50
        patience_counter = 0

        for i in range(iterations):
            improved = False

            for model in models:
                # try increasing weight
                w_plus = weights.copy()
                w_plus[model] += step

                # normalize
                total = sum(w_plus.values())
                w_plus = {k: v/total for k, v in w_plus.items()}

                score_plus, _ = get_f1_score(w_plus)

                if score_plus > current_best_score:
                    weights = w_plus
                    current_best_score = score_plus
                    improved = True
                    continue  # move to next iteration with new weights

                # try decreasing weight
                w_minus = weights.copy()
                w_minus[model] = max(0.0, w_minus[model] - step)

                # Normalize
                total = sum(w_minus.values())
                if total == 0:
                    total = 1e-9
                w_minus = {k: v/total for k, v in w_minus.items()}

                score_minus, _ = get_f1_score(w_minus)

                if score_minus > current_best_score:
                    weights = w_minus
                    current_best_score = score_minus
                    improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.optimized_weights = weights
        print(f"optimized F1: {current_best_score:.4f}")
        print(f"best Weights: {weights}")

    def ensemble_and_submit(self, submission_ids):
        print("\nbeginning ensembling strategy...")

        if not self.preds_test:
            raise ValueError("no predictions available for ensembling!")

        weights = self.optimized_weights
        final_test = np.zeros(len(self.X_test))
        final_val = np.zeros(len(self.X_val))

        for m, w in weights.items():
            final_test += self.preds_test[m] * w
            final_val += self.preds_val[m] * w

        # optimize threshold finally on the weighted validation set
        precisions, recalls, thresholds = \
            precision_recall_curve(self.y_val, final_val)

        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        auc_score = roc_auc_score(self.y_val, final_val)
        print(f"------------------------------")
        print(f"\nfull ensemble result: AUC= \
               {auc_score:.4f}, best F1={best_f1:.4f} @ t={best_t:.3f}")

        submission = pd.DataFrame({
            'object_id': submission_ids,
            'probability': final_test,
            'predicted_class': (final_test > best_t).astype(int)
        })
        return submission


# experiment pipeline
if __name__ == "__main__":

    X, y = get_prepared_dataset('train')

    X = clean_data(X)
    # just drops redshift err since our current method of
    # processing data does not capture this.

    # stratified split
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # test data
    X_test, test_ids = get_prepared_dataset('test')
    X_test = clean_data(X_test)

    # align columns
    cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[cols]
    X_val = X_val[cols]
    X_test = X_test[cols]

    exp = Experiment(X_train, y_train, X_val, y_val, X_test)

    # catboost, our generalist model
    exp.tune_catboost_optuna(n_trials=50)

    # 2. xgboost
    exp.tune_xgboost_optuna(n_trials=50)

    # 3. lgbm
    exp.tune_lgbm_optuna(n_trials=50)

    # cnn
    exp.tune_cnn_optuna(n_trials=50)

    # kneighbors
    exp.tune_knn_optuna(n_trials=50)

    # adaboostregressor
    exp.tune_adaregressor_optuna(n_trials=50)

    # specialists
    exp.train_specialists()

    # calibration
    exp.apply_isotonic_calibration()

    sub = exp.ensemble_and_submit(test_ids)
    sub.to_csv("../../results/experimental_test_submission.csv", index=False)
    print("Submission test file saved to results folder.")
