# Neural network experiment using muon optimizer: 
# the goal is to ensemble xgboost with muon...somehow

import pandas as pd
from src.data_loader import load_lightcurves
import os
from config import DATA_DIR, TRAIN_LOG

import torch
import torch.nn as nn
import muon as muon


filters=['z', 'r','y','i','g','u']

class TestNN(nn.Module): 
    def __init__(self, hidden_dim=64, latent_dim=16):
        super().__init__()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        output = self.fc4(x)
        return output


        




log_path = os.path.join(DATA_DIR, TRAIN_LOG)
train_log = pd.read_csv(log_path)

# Load Lightcurves
lc_df = load_lightcurves(train_log, dataset_type='train')

# testing stuff. 



print(f"\nCombined training data: {len(lc_df)} rows, {len(lc_df.columns)} columns")
print(f"Memory usage: {lc_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n  {lc_df.head()} ")
