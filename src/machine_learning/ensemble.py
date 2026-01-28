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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from catboost import CatBoostClassifier
from config import MODEL_CONFIG
from src.data_loader import get_prepared_dataset
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import clear_output

# Trying some activation functions for the hell of it.
# inspiration: https://arxiv.org/pdf/1905.02473

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
        self.t_l = nn.Parameter(torch.zeros(in_features)) # left threshold
        self.a_l = nn.Parameter(torch.zeros(in_features)) # left slope
        self.t_r = nn.Parameter(torch.zeros(in_features)) # right threshold
        self.a_r = nn.Parameter(torch.ones(in_features))  # right slope (init 1 for identity)

    def forward(self, x):
        # Piecewise logic:
        # x <= t_l: t_l + a_l * (x - t_l)
        # t_l < x < t_r: x
        # x >= t_r: t_r + a_r * (x - t_r)
        
        # We use torch.where for differentiability
        out = torch.where(x <= self.t_l, self.t_l + self.a_l * (x - self.t_l), x)
        out = torch.where(out >= self.t_r, self.t_r + self.a_r * (out - self.t_r), out)
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

# 15. 
# used for finding periodics, this is just a funny addition because of the name i guess.
class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)


# -------------------------------
# Loss Functions : 
# Logit Adjustment Loss
# WeightedBCELoss
# Cross Entropy
# FocalLoss
# Soft F1 Loss
# PolyLoss
# LDAMLoss
# TverskyLoss with high beta
# FocalTverskyLoss
# Cross Balanced Focal Loss
# Matthews Correlation Coefficient

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction=reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss) # prevent nan values
        focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class PolyLoss(nn.Module):
    # polynomial expansion of focal loss
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        # poly1 form: CE + epsilon * (1-pt)
        poly_loss = bce_loss + self.epsilon * (1-pt)

        if self.reduction == 'mean':
            return poly_loss.mean()
        return poly_loss


class LogitAdjustmentLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        # cls_num_list : [neg_count, pos_count] -> [9940, 60]
        priors = torch.tensor(cls_num_list) / sum(cls_num_list)
        self.logit_adj = tau * torch.log(priors + 1e-12)

    def forward(self, logits, targets):
        # (B, 2)
        adjusted_logits 

# label-distribution-aware margin loss
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super.__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits (B, 2)
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1).long(), 1)
        index_float = index.type(torch.cuda.FloatTensor) if logits.is_cuda else index.type(torch.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :].to(logits.device), index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))

        x_m = logits = batch_m
        output = torch.where(index.bool(), x_m, logits)
        return self.criterion(self.s * output, targets.long())



class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        # alpha: false-positive sensitivity, 
        # beta: false negative sensitivity,
        # high beta (0.7-0.8) emphasizes recall

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1-targets)).sum()
        fn = ((1-probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky_index


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(slef, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1-targets)).sum()
        fn = ((1-probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # loss = (1 - tversky)^gamma
        return torch.pow((1 - tversky), self.gamma)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes=2, beta=0.999, gamma=2.0):
        super.__init__()
        self.gamma = gamma
        # 
        # calculate effecive weights
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        self.weights = torch.tensor(weights).float()

    def forward(self, logits, targets):
        # need to manually apply weights to the Focal Loss
        # logits: (B, 1) 
        # calculate standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # calculate Pt
        pt = torch.exp(-bce_loss)
        
        # get class weights for current batch
        # assuming targets are 0/1
        batch_weights = self.weights.to(targets.device)[targets.long()]
        
        # combine
        cb_focal_loss = batch_weights * (1 - pt) ** self.gamma * bce_loss
        return cb_focal_loss.mean()


class MCCloss(nn.Module):
    def __init__():
        super.__init__()
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum()
        tn = ((1 - probs) * (1 - targets)).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        # add epsilon to avoid nan
        mcc = numerator / (denominator + 1e-7)

        # 1 - mcc so that minimizing loss maxes mcc
        return  1 - mcc


class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        # Input Logits (B, 1) -> Sigmoid -> Probabilities
        probs = torch.sigmoid(logits)
        y_true = targets.float()
        
        tp = (probs * y_true).sum(dim=0)
        fp = (probs * (1 - y_true)).sum(dim=0)
        fn = ((1 - probs) * y_true).sum(dim=0)
        
        f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1.mean()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight_value=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight_value])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        if self.pos_weight.device != logits.device:
            self.criterion.pos_weight = self.pos_weight.to(logits.device)
        return self.criterion(logits, targets.float())


# factory function to use for training.
def get_loss_function(name, device, train_y=None):
    if train_y is not None: 
        neg_count = (train_y == 0).sum()
        pos_count = (train_y == 1).sum()
        counts = [neg_count, pos_count]
    else: 
        counts = [100, 100]
    
    if name == 'bce': 
        return nn.BCEWithLogitsLoss()
    elif name == 'weighted_bce': 
        weight = counts[0] / counts[1]
        return WeightedBCELoss(pos_weight_value=weight)
    elif name == 'focal':
        return FocalLoss(gamma=2.0, alpha=0.25)
        
    elif name == 'soft_f1':
        return SoftF1Loss()
        
    elif name == 'tversky':
        # Beta=0.7 favors Recall (catching TDEs)
        return TverskyLoss(alpha=0.3, beta=0.7)
        
    elif name == 'focal_tversky':
        return FocalTverskyLoss(gamma=1.5)
        
    elif name == 'logit_adj':
        return LogitAdjustmentLoss(cls_num_list=counts)
        
    elif name == 'ldam':
        return LDAMLoss(cls_num_list=counts)
        
    elif name == 'poly':
        return PolyLoss(epsilon=1.0)
        
    elif name == 'cb_focal':
        return ClassBalancedFocalLoss(samples_per_cls=counts)
        
    elif name == 'mcc':
        return MCCLoss()

    return None

# TabNet
# FT-Transformer
# CNN
# Tabular Residual Net 
# DCN V2 (Deep & Cross Network)
# SAINT SAINT (Self-Attention and Intersample Attention)


# TabNet : width of 64 or 128, 3-5 layers, dropout (0.[1]1) + batchnorm
# DCN V2 : 32 or 64, 2 Cross layers, Weight Decay 1e-4
# FT-Transformer: 32(d_model) 3 layers, 4 heads, weight decay high
# MPL Baseline : 64/32/16 width, 3 layers (tapering), batchnorm regularization
# ResNet: width of 64 or 128, depth 3-4, input->Linear(128 -> Residual Block x 3 -> Linear(1), 
#        each Residual block:  Linear -> BatchNorm -> SReLU -> Dropout(0.1-0.2) -> Linear -> BatchNorm



# Notes: 
# ensemble with Catboost : 
# Convert predictions to Ranks (0 to 1).
# Average the ranks.
# Find the optimal Threshold on the averaged ranks to maximize F1.

# will use WeightedRandomSampler because without it there is a high probability of 
# sampling no TDEs from the dataset, which was why the F1 would go up and back down dramatically.
# learning rate cannot accomodate for this otherwise. 

# try a variety of activation and loss functions, 
# and tabnet, ft-transformer, cnn's, and whatever else I can think of to try.
# each of these networks will be tested in as many ways as possible with the given strategy above. 


# rewriting training and ensembling functions tomorrow morning with all of this 
# procedurally attempted.



# supposed to show training progress
class Monitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        
    def update(self, t_loss, v_loss, v_f1):
        self.train_losses.append(t_loss)
        self.val_losses.append(v_loss)
        self.val_f1s.append(v_f1)
        
    def plot(self):
        clear_output(wait=True)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.6)
        ax1.plot(self.val_losses, label='Val Loss', color='orange', alpha=0.6)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(self.val_f1s, label='Val F1', color='green')
        ax2.set_ylabel('F1 Score')
        ax2.legend(loc='upper right')
        
        plt.title('Training Progress')
        plt.show()

# want secondary function to display results of each training run from each ensemble run















# Try each neural network, with each activation function * each loss combinatorially.
def search_all_ensemble_strategy():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    X, y = get_prepared_dataset('train')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    # convert data to tensors
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(y_train.values).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # to fix imbalances:
    class_counts = torch.bincount(y_train_tensor) # how many 0's and 1's
    class_weights = 1 / class_counts.float() # weight = 1 / frequency
    sample_weights = class_weights[y_train_tensor] # give every row a weight based on its class
    
    # we create our own sampler.
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                num_samples=len(sample_weights), 
                                replacement=True)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=sampler, 
        pin_memory=True
    )

    monitor = Monitor()


    model_A = TabularResNet(input_dim=16, activation_class=SReLU).to(device)
    criterion = SoftF1Loss()
    optimizer = torch.optim.AdamW(model_A.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)




search_all_ensemble_strategy()