import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from itertools import product

from ..config import TRAIN_LOG_PATH, TEST_LOG_PATH, FILTER_WAVELENGTHS, PROCESSED_TRAINING_DATA_PATH, PROCESSED_TESTING_DATA_PATH, MODEL_CONFIG
from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device.type}")


# Configuration
BIAS_WEIGHT_NN = 0.0
BIAS_WEIGHT_XGB = 1.0
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DeepTDENet(nn.Module):
    """Deep Neural Network for TDE Classification - 7 hidden layers"""
    
    def __init__(self, input_dim, dropout=0.3):
        super(DeepTDENet, self).__init__()
        
        self.network = nn.Sequential(
            # layer 1: Input -> 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 2: 512 -> 512
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 3: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 4: 256 -> 256
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 5: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 6: 128 -> 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # layer 7: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

