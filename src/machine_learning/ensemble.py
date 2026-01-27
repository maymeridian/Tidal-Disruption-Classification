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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from catboost import CatBoostClassifier
from config import MODEL_CONFIG
from src.data_loader import get_prepared_dataset
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Trying some activation functions for the hell of it.
# inspiration: https://arxiv.org/pdf/1905.02473

# 1. Sigmoid

# 2. Gaussian
class GaussianActivation(nn.Module):
    def __init__(self, init_mu=0.0, init_sigma=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
    
    def forward(self, x):
        return torch.exp(-((x - self.mu) / self.sigma) ** 2)

# 3. ReLU

# 4. Leaky ReLU

# 5. ELU - Exponential Linear Uni

# 6. SELU - Scaled Exponential Linear Unit

# 7. PReLU - Parametric ReLU 

# S-Shaped ReLU

# APLU - Adaptive Piecewise Linear Unit

# 'Mexican ReLU' 
# https://www.researchgate.net/figure/Mexican-hat-type-activation-functions-3_fig1_268386570
class MexicanHatMultiScale(nn.Module):
    def __init__(self, num_scales=3):
        self()


# https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
# 3. Mish: Self Regularized Non-Monotonic Neural Activation Function
# -------------------------
# f(x) = x * tanh(sigma(x))
# where sigma(x) = ln(1 + e^x) is the softplus activation function 
# -------------------------

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# https://arxiv.org/pdf/1710.05941v1
# 4. Swish - Smooth Non-Monotonic Activation Function
# a "self-gated" activation function
# -------------------------
# f(x) = x * sigma(x)
# where sigma is (1 + exp(-x))^-1
# -------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# https://arxiv.org/pdf/1606.08415
# 5. GELU: Gaussian Error Linear Units
# -------------------------
# G(x) = xP(X <= x) = xPhi(x) = x * (1/2) [ 1 + erf( x/sqrt(2) ) ]
# approximated by : 
#  0.5x(1 + tanh[(sqrt(2/pi) * (x + 0.044715x^3))])   or   x*sigma(1.702x)
# -------------------------
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)




# used for periodics, this is just a funny addition 
class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)



# -------------------------------
# Loss Functions



# -------------------------------


# the plan is to try a variety of activation and loss functions, 
# and tabnet, ft-transformer, cnn's, and whatever else I can think of to try.
# each of these networks will be tested in as many ways as possible with the given strategy above. 


# rewriting training and ensembling functions tomorrow morning with all of this 
# procedurally attempted.



