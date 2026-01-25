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

from config import TRAIN_LOG_PATH, TEST_LOG_PATH, FILTER_WAVELENGTHS, PROCESSED_TRAINING_DATA_PATH, PROCESSED_TESTING_DATA_PATH, MODEL_CONFIG
from src.data_loader import load_lightcurves
from src.features import apply_deextinction, extract_features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device.type}")


# Configuration
BIAS_WEIGHT_NN = 0.68
BIAS_WEIGHT_XGB = 0.32
NUM_EPOCHS = 5000  
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  
TEST_TDE_PERCENTAGE = 6.7 


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


def load_process_training_data():
    train_log = pd.read_csv(TRAIN_LOG_PATH)

    if os.path.exists(PROCESSED_TRAINING_DATA_PATH):
        print(f"Loading cached training data from {PROCESSED_TRAINING_DATA_PATH}...")
        train_features_df = pd.read_csv(PROCESSED_TRAINING_DATA_PATH)
        print(f"Loaded {len(train_features_df)} entries.")
        return train_features_df, train_log
        
    lc_df = load_lightcurves(dataset_type='train')
    print("Applying deextinction...")
    lc_df = apply_deextinction(lc_df, train_log)
    print("Extracting features...")
    train_features_df = extract_features(lc_df, train_log)
    
    print(f"Saving processed training data to {PROCESSED_TRAINING_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_TRAINING_DATA_PATH), exist_ok=True)
    train_features_df.to_csv(PROCESSED_TRAINING_DATA_PATH, index=False)
    
    return train_features_df, train_log


def load_process_test_data(): 
    test_log = pd.read_csv(TEST_LOG_PATH)

    if os.path.exists(PROCESSED_TESTING_DATA_PATH):
        print(f"Loading cached testing data from {PROCESSED_TESTING_DATA_PATH}...")
        test_features_df = pd.read_csv(PROCESSED_TESTING_DATA_PATH)
        print(f"Loaded {len(test_features_df)} entries.")
        return test_features_df, test_log

    lc_df = load_lightcurves(dataset_type='test')
    print("Applying deextinction...")
    lc_df = apply_deextinction(lc_df, test_log)
    print("Extracting features...")
    test_features_df = extract_features(lc_df, test_log)
    
    print(f"Saving processed testing data to {PROCESSED_TESTING_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_TESTING_DATA_PATH), exist_ok=True)
    test_features_df.to_csv(PROCESSED_TESTING_DATA_PATH, index=False)

    return test_features_df, test_log


def train_individual_model(X_train, y_train, X_val, y_val, 
                          dropout=0.3, alpha=0.25, gamma=2.0, 
                          epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train deep neural network with mini-batch SGD for longer, more careful training"""
    
    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # data -> tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # init deep model
    model = DeepTDENet(input_dim=X_train.shape[1], dropout=dropout).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Batch size: {BATCH_SIZE}, Steps per epoch: {len(X_train) // BATCH_SIZE}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-7
    )
    

    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

    best_val_auc = 0.0
    best_val_loss = float('inf')
    patience = 100  # Wait 100 eval cycles before stopping training early, this is to help prevent overfitting.
    patience_counter = 0
    best_model_state = None
    
    epoch_times = []
    
    # mini-batches
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    import time
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + num_batches / len(train_loader))
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        
        if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        eval_frequency = 20 if epoch < 1000 else 50  # evaluate every 20 epochs early, then every 50
        
        if (epoch + 1) % eval_frequency == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = loss_fn(val_logits, y_val_tensor)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                
                temp_threshold = np.percentile(val_probs, 95)
                temp_threshold = max(0.2, min(0.8, temp_threshold))
                
                val_predictions = (val_probs > temp_threshold).astype(float)
                
                acc = (val_predictions == y_val.reshape(-1, 1)).mean()
                
                # calculate metrics
                if y_val.sum() > 0 and val_predictions.sum() > 0:
                    tde_recall = val_predictions[y_val == 1].mean()
                    precision = (val_predictions[y_val.reshape(-1, 1) == 1]).sum() / max(val_predictions.sum(), 1)
                    f1 = 2 * (precision * tde_recall) / max(precision + tde_recall, 1e-8)
                    
                    if len(np.unique(y_val)) > 1:
                        val_auc = roc_auc_score(y_val, val_probs)
                    else:
                        val_auc = 0.0
                else:
                    tde_recall = precision = f1 = val_auc = 0.0

                num_pred_tdes = val_predictions.sum()
                num_actual_tdes = y_val.sum()

                print(f"Epoch {epoch+1:5d}/{epochs} | "
                      f"Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss.item():.4f} | "
                      f"AUC: {val_auc:.4f} | F1: {f1:.4f} | "
                      f"Recall: {tde_recall:.3f} | Prec: {precision:.3f} | "
                      f"Pred: {num_pred_tdes:.0f}/{num_actual_tdes:.0f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | ")

                # early stopping check
                improved = False
                if val_auc > best_val_auc:
                    if num_pred_tdes >= 1:  # predicting some TDEs
                        improvement = val_auc - best_val_auc
                        best_val_auc = val_auc
                        best_val_loss = val_loss.item()
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                        improved = True
                
                if not improved:
                    patience_counter += 1
                
                # early stopping
                if patience_counter >= patience:
                    print(f"\n✗ Early stopping at epoch {epoch+1}")
                    print(f"  Best validation AUC: {best_val_auc:.4f}")
                    print(f"  No improvement for {patience * eval_frequency} epochs")
                    break
                

    
    total_time = (time.time() - start_time) / 3600
    print(f"\nTraining completed in {total_time:.2f} hours")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✓ Loaded best model (AUC: {best_val_auc:.4f})")


    # final predictions
    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(X_train_tensor)).cpu().numpy()
        val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
    

    return model, train_probs, val_probs, best_val_loss, scaler


def find_optimal_f1_threshold(y_true, y_probs, min_threshold=0.05, max_threshold=0.95):
    """Find threshold that maximizes F1 score with extensive search"""
    thresholds = np.linspace(min_threshold, max_threshold, 500)  # Very granular search
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int).flatten()
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'num_pred': y_pred.sum(),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    # print top 10 thresholds for fun
    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    print("\n" + "="*80)
    print("Top 10 F1 thresholds:")
    print("="*80)
    print(results_df.head(10).to_string(index=False))
    print("="*80)
    
    return best_threshold, best_f1, best_precision, best_recall


def ensemble_nn_xgb(X_train, y_train, X_val, y_val, nn_train_probs, nn_val_probs):
    """Ensemble neural network with XGBoost"""
    
    print("\n" + "="*80)
    print("=== Training XGBoost ===")
    print("="*80)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    print(f"Class imbalance ratio: {scale_pos_weight:.2f} (using as scale_pos_weight)")
    print(f"Training set: {(y_train == 1).sum()} TDEs out of {len(y_train)} ({100*(y_train == 1).sum()/len(y_train):.2f}%)")

    seed = MODEL_CONFIG['random_seed'] if 'MODEL_CONFIG' in globals() else 42
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,  
        max_depth=4,  
        learning_rate=0.03,  
        eval_metric='logloss', 
        scale_pos_weight=scale_pos_weight, 
        min_child_weight=1, 
        random_state=seed,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,      # regularization
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0  # L2 regularization
    )
    
    # train xgboost
    xgb_model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=True 
    )
    
    xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]


    # ensemble weighted avg search
    print("\n" + "="*80)
    print("Testing ensemble weights:")
    print("="*80)
    best_ensemble_auc = 0
    best_weights = (BIAS_WEIGHT_NN, BIAS_WEIGHT_XGB)
    

    # want to test a range of weights for deep neural network and xgboost to find the best ones
    # for weighted avg
    weight_results = []
    for nn_weight in np.linspace(0.3, 0.9, 13):  
        xgb_weight = 1 - nn_weight
        test_probs = nn_val_probs.flatten() * nn_weight + xgb_val_probs * xgb_weight
        test_auc = roc_auc_score(y_val, test_probs) if len(np.unique(y_val)) > 1 else 0
        
        weight_results.append({
            'nn_weight': nn_weight,
            'xgb_weight': xgb_weight,
            'auc': test_auc
        })
        
        print(f"  NN={nn_weight:.2f}, XGB={xgb_weight:.2f}: AUC={test_auc:.4f}")
        
        if test_auc > best_ensemble_auc:
            best_ensemble_auc = test_auc
            best_weights = (nn_weight, xgb_weight)
    
    print(f"\n✓ Best weights: NN={best_weights[0]:.2f}, XGB={best_weights[1]:.2f} (AUC={best_ensemble_auc:.4f})")

    # using best weights
    ensemble_train_probs = (nn_train_probs.flatten() * best_weights[0] + 
                           xgb_train_probs * best_weights[1])
    ensemble_val_probs = (nn_val_probs.flatten() * best_weights[0] + 
                         xgb_val_probs * best_weights[1])

    # individual models area-under-curve
    nn_auc = roc_auc_score(y_val, nn_val_probs) if len(np.unique(y_val)) > 1 else 0
    xgb_auc = roc_auc_score(y_val, xgb_val_probs) if len(np.unique(y_val)) > 1 else 0
    ensemble_auc = roc_auc_score(y_val, ensemble_val_probs) if len(np.unique(y_val)) > 1 else 0
    
    # find optimal threshold with extensive search
    print("\nSearching for optimal F1 threshold...")
    optimal_threshold, best_f1, precision, recall = find_optimal_f1_threshold(
        y_val, ensemble_val_probs, 
        min_threshold=0.05, 
        max_threshold=0.95
    )
    
    print("\n" + "="*80)
    print("FINAL VALIDATION SCORES")
    print("="*80)
    print(f"  Neural Network AUC:  {nn_auc:.4f}")
    print(f"  XGBoost AUC:         {xgb_auc:.4f}")
    print(f"  Ensemble AUC:        {ensemble_auc:.4f}")
    print(f"\n  Optimal F1 Score:    {best_f1:.4f}")
    print(f"    Threshold:         {optimal_threshold:.4f}")
    print(f"    Precision:         {precision:.4f}")
    print(f"    Recall:            {recall:.4f}")
    print("="*80 + "\n")
    
    return ensemble_train_probs, ensemble_val_probs, xgb_model, optimal_threshold


def train_selected_model(df, train_log): 
    """Train the deep model with selected hyperparameters for LONG, CAREFUL training"""
    X = df.drop(columns='object_id').values
    y = df['object_id'].map(train_log.set_index('object_id')['target']).values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Maximum epochs:      {NUM_EPOCHS}")
    print(f"  Initial learning rate: {LEARNING_RATE}")
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Training samples:    {len(X_train)} ({(y_train==1).sum()} TDEs, {100*(y_train==1).sum()/len(y_train):.2f}%)")
    print(f"  Validation samples:  {len(X_val)} ({(y_val==1).sum()} TDEs, {100*(y_val==1).sum()/len(y_val):.2f}%)")
    print(f"  Feature dimensions:  {X.shape[1]}")
    print("="*80 + "\n")
    
    print("=== Training Deep Neural Network ===")
    model, train_probs, val_probs, val_loss, scaler = train_individual_model(
        X_train, y_train, X_val, y_val, 
        dropout=0.4,
        alpha=0.25, 
        gamma=2.0,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )

    # ensemble with xgboost
    ensemble_train_probs, ensemble_val_probs, xgb_model, optimal_threshold = ensemble_nn_xgb(
        X_train, y_train, X_val, y_val,
        train_probs,
        val_probs
    )
    
    return model, xgb_model, ensemble_train_probs, ensemble_val_probs, scaler, optimal_threshold

def predict_test_set(neural_network_model, xgb_model, scaler, test_features_df, test_log, train_log, 
                     optimal_threshold=0.5, output_path='submission.csv'):
    """Generate predictions for test set using optimal F1 threshold"""
    X_test = test_features_df.drop(columns='object_id').values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = scaler.transform(X_test)
    
    neural_network_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        nn_logits = neural_network_model(X_test_tensor)
        neural_network_probs = torch.sigmoid(nn_logits).cpu().numpy().flatten()

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    ensemble_probs = (neural_network_probs * BIAS_WEIGHT_NN + xgb_probs * BIAS_WEIGHT_XGB)
    
    predictions = (ensemble_probs >= optimal_threshold).astype(int)
    predicted_tdes = predictions.sum()
    predicted_percentage = (predicted_tdes / len(test_log)) * 100
    
    print(f"\nTest Predictions (threshold={optimal_threshold:.4f}):")
    print(f"  {predicted_tdes} TDEs out of {len(test_log)} objects ({predicted_percentage:.2f}%)")
    

    submission = pd.DataFrame({
        'object_id': test_features_df['object_id'],
        'target': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return predictions, ensemble_probs



if __name__ == "__main__":
    train_features_df, train_log = load_process_training_data()
    test_features_df, test_log = load_process_test_data()

    nn_model, xgb_model, ensemble_train_probs, ensemble_val_probs, scaler, optimal_threshold = train_selected_model(
        train_features_df, train_log
    )

    predictions, probs = predict_test_set(
        neural_network_model=nn_model,
        xgb_model=xgb_model, 
        scaler=scaler,  
        test_features_df=test_features_df,
        test_log=test_log,
        train_log=train_log,
        optimal_threshold=optimal_threshold, 
        output_path='submission.csv'
    )
    