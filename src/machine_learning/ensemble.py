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



# 8. APLU - Adaptive Piecewise Linear Unit

# 9. 'Mexican ReLU' 
# https://www.researchgate.net/figure/Mexican-hat-type-activation-functions-3_fig1_268386570
class MexicanHatMultiScale(nn.Module):
    def __init__(self, num_scales=3):
        self()


# https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
# 10. Mish: Self Regularized Non-Monotonic Neural Activation Function
# -------------------------
# f(x) = x * tanh(sigma(x))
# where sigma(x) = ln(1 + e^x) is the softplus activation function 
# -------------------------

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# https://arxiv.org/pdf/1710.05941v1
# 11. Swish - Smooth Non-Monotonic Activation Function
# a "self-gated" activation function
# -------------------------
# f(x) = x * sigma(x)
# where sigma is (1 + exp(-x))^-1
# -------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# https://arxiv.org/pdf/1606.08415
# 12. GELU: Gaussian Error Linear Units
# -------------------------
# G(x) = xP(X <= x) = xPhi(x) = x * (1/2) [ 1 + erf( x/sqrt(2) ) ]
# approximated by : 
#  0.5x(1 + tanh[(sqrt(2/pi) * (x + 0.044715x^3))])   or   x*sigma(1.702x)
# -------------------------
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# https://arxiv.org/abs/1907.06732
# 13. PADÉ Activation Units

class PADE(nn.Module):
    def __init__(self, m=2, n=2): 
        super().__init__()
        self.m = m
        self.n = n

        self.numerator_coeffs = nn.Parameter(torch.randn(m+1))
        self.denominator_coeffs = nn.Parameter(torch.randn(n))

    def forward(self, x):
        # compute numerator : a0 + a1*x + a2*x^2 + ...
        numerator = torch.zeros_like(x)

        for i, coeff in enumerate(self.numerator_coeffs):
            numerator += coeff * (x**i)
        
        # compute denominator : 1 + b1*x + b2*x^2 + ...
        denominator = torch.ones_like(x)
        for i, coeff in enumerate(self.denominator+coeffs):
            denominator += coeff * (x ** (i+1))

        # added epsilon to prevent division by zero.
        return numerator / (denominator + 1e-8)


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
# AsymetricLoss
# PolyLoss
# LDAMLoss
# TverskyLoss with high beta
# FocalTverskyLoss
# Cross Balanced Focal Loss
# Matthews Correlation Coefficient


# Neural Networks: 
# TabNet
# FT-Transformer
# CNN
# Tabular Residual Net 
# DCN V2 (Deep & Cross Network)
# SAINT SAINT (Self-Attention and Intersample Attention)
# NODE (Neural Oblivious Decision Ensembles)


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


def progress_bar(epoch, batch, total_batches, loss):
    progress = batch / total_batches * 100
    print(f"\rEpoch {epoch} [{batch}/{total_batches}] {progress:.1f}% Loss: {loss:.4f}", 
          end='', flush=True)
    
    if batch == total_batches:
        print()  # Newline at epoch end

# -------------------------------





# essentially, search all possible world lines for the correct model :P
# this is mad science
def search_all_world_lines():
    print("""⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⣿⣿⣿⣿⣿⣿⡿⠟⠋⠉⠉⠉⠙⠛⢿⣿⣿⣿⣿⣿
    ⣿⣿⡟⢩⣶⠂⠄⠄⣠⣶⣿⣯⣉⣷⣦⠈⣻⣿⣿⣿
    ⣿⣿⣿⣄⠁⠄⠄⢸⡿⠟⠛⠉⠉⠉⠛⢧⠘⣿⣿⣿
    ⣿⣿⣿⡿⠄⠄⠄⠄⢀⠄⣠⡄⠄⠄⠄⠄⠄⢹⣿⣿
    ⣿⣿⣿⡇⠄⠄⠄⣸⡘⢴⣻⣧⣤⢀⣂⡀⠄⢸⣿⣿
    ⣿⣿⣿⡇⠄⠘⢢⣿⣷⣼⣿⣿⣿⣮⣴⢃⣤⣿⣿⣿
    ⣿⣿⡿⠄⣠⣄⣀⣙⣿⣿⣿⣿⣿⡿⠋⢸⡇⢹⣿⣿
    ⣿⣿⡇⠰⣻⣿⣿⣿⠿⠮⠙⠿⠓⠛⠄⠄⠈⠄⢻⣿
    ⣿⡟⠄⠄⠈⠙⠋⠄⠄⠄⠄⠁⠄⠄⠄⠄⠄⠄⢾⣿
    ⡏⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢀⠄⠄⠄⠄⠄⠄⠈⣿
    ⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    """
    )

search_all_world_lines()