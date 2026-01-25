import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
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

# for weighted average in ensemble stage
BIAS_WEIGHT_NN = 0.4
BIAS_WEIGHT_XGB = 0.6
NUM_EPOCHS = 15000
LEARNING_RATE = 5e-5

THRESHOLD = 0.75

# load or process training data
def load_process_training_data():
    train_log = pd.read_csv(TRAIN_LOG_PATH)

    if os.path.exists(PROCESSED_TRAINING_DATA_PATH):
        print(f"Loading cached training data from {PROCESSED_TRAINING_DATA_PATH}...")
        train_features_df = pd.read_csv(PROCESSED_TRAINING_DATA_PATH)
        print(f"Loaded {len(train_features_df)} entries.")
        return train_features_df, train_log
        
    lc_df = load_lightcurves(dataset_type='train')
    
    # Preprocess
    print("Applying deextinction...")
    lc_df = apply_deextinction(lc_df, train_log)
    
    # Feature Engineering
    print("Extracting features...")
    train_features_df = extract_features(lc_df, train_log)
    
    # save result
    print(f"Saving processed training data to {PROCESSED_TRAINING_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_TRAINING_DATA_PATH), exist_ok=True)
    train_features_df.to_csv(PROCESSED_TRAINING_DATA_PATH, index=False)
    
    return train_features_df

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
    
    # save result
    print(f"Saving processed testing data to {PROCESSED_TESTING_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_TESTING_DATA_PATH), exist_ok=True)
    test_features_df.to_csv(PROCESSED_TESTING_DATA_PATH, index=False)

    return test_features_df, test_log



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Binary Cross-Entropy loss calculation
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)  # convert BCE loss to probability
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss  # apply focal adjustment
        return focal_loss.mean()


class TestNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(0.5)  # Increased from 0.3

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.hidden_activation = nn.ReLU()
        self.hidden_dropout = nn.Dropout(0.5)  # Increased from 0.3

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_batch_norm(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        x = self.hidden_layer(x)
        x = self.hidden_batch_norm(x)
        x = self.hidden_activation(x)
        x = self.hidden_dropout(x)

        x = self.output_layer(x)
        return x



def train_individual_model(X_train, y_train, X_val, y_val, 
                          hidden_dim=64, alpha=0.25, gamma=2.0, epochs=100, lr=LEARNING_RATE):

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Move to device
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    model = TestNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    
    # weight decay to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience = 15  # early stopping if accuracy 'too high'
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)

        loss = loss_fn(logits, y_train_tensor)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = loss_fn(val_logits, y_val_tensor)
                val_probs_temp = torch.sigmoid(val_logits).cpu().numpy()
                val_predictions = (val_probs_temp > THRESHOLD).astype(float)
                
                acc = (val_predictions == y_val.reshape(-1, 1)).mean()
                
                if y_val.sum() > 0:
                    tde_recall = val_predictions[y_val == 1].mean()
                    precision = val_predictions[val_predictions == 1].sum() / max(val_predictions.sum(), 1)
                    f1 = 2 * (precision * tde_recall) / max(precision + tde_recall, 1e-8)
                    
                    # Calculate AUC
                    if len(np.unique(y_val)) > 1:
                        val_auc = roc_auc_score(y_val, val_probs_temp)
                    else:
                        val_auc = 0.0
                else:
                    tde_recall = precision = f1 = val_auc = 0.0
                
                extra_space = ''
                if epoch+1 < 1000 == 0: 
                    extra_space = ' '

                print(f"  Epoch {epoch+1} {extra_space}| Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Acc: {acc:.4f} | Recall: {tde_recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | AUC: {val_auc:.4f}")

                # early stopping based on AUC
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_loss = val_loss.item()
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                

    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(X_train_tensor)).cpu().numpy()
        val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
    
    if np.isnan(train_probs).any():
        print("NaN in train predictions. This is _not_ good.")
    
    return model, train_probs, val_probs, best_val_loss, scaler



def hyperparameter_calibration(X_train, y_train, X_val, y_val):
    """Test different hyperparameter combinations"""
    alpha_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35] 
    gamma_values = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5]
    hidden_dims = [32, 64, 128, 256, 512, 1024, 2048]
    
    results = []
    print("\n=== Calibrating Hyperparameters ===\n")
    
    for hidden_dim, alpha, gamma in product(hidden_dims, alpha_values, gamma_values):
        print(f"\nTesting: hidden_dim={hidden_dim}, alpha={alpha}, gamma={gamma}")

        model, train_probs, val_probs, val_loss, scaler = train_individual_model(
            X_train, y_train, X_val, y_val, 
            hidden_dim=hidden_dim, 
            alpha=alpha,
            gamma=gamma,
            epochs=NUM_EPOCHS,
            lr=LEARNING_RATE
        )
        results.append({
            'hidden_dim': hidden_dim,
            'alpha': alpha, 
            'gamma': gamma,
            'val_loss': val_loss, 
            'model': model,
            'train_probs': train_probs, 
            'val_probs': val_probs,
            'scaler': scaler
        })
    
    results.sort(key=lambda x: x['val_loss'])
    
    print("\n=== Best Model ===")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"{i+1}. hidden_dim={r['hidden_dim']}, alpha={r['alpha']}, gamma={r['gamma']}, val_loss={r['val_loss']:.4f}")

    return results



def ensemble_nn_xgb(X_train, y_train, X_val, y_val, nn_train_probs, nn_val_probs):
    """Ensemble neural network with XGBoost"""
    
    print("\n=== Training XGBoost ===")

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    seed = MODEL_CONFIG['random_seed']
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                 eval_metric='logloss', scale_pos_weight=scale_pos_weight, min_child_weight=2, random_state=seed)
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

    # weighted average ensemble
    ensemble_train_probs = (nn_train_probs.flatten() * BIAS_WEIGHT_NN + xgb_train_probs * BIAS_WEIGHT_XGB)
    ensemble_val_probs = (nn_val_probs.flatten() * BIAS_WEIGHT_NN + xgb_val_probs * BIAS_WEIGHT_XGB)

    # eval model accuracy
    nn_auc = roc_auc_score(y_val, nn_val_probs) if len(np.unique(y_val)) > 1 else 0
    xgb_auc = roc_auc_score(y_val, xgb_val_probs) if len(np.unique(y_val)) > 1 else 0
    ensemble_auc = roc_auc_score(y_val, ensemble_val_probs) if len(np.unique(y_val)) > 1 else 0
    
    print(f"\n Area-Under-Curve Scores:")
    print(f"  Neural Network: {nn_auc:.4f}")
    print(f"  XGBoost:        {xgb_auc:.4f}")
    print(f"  ensembled:       {ensemble_auc:.4f}")
    
    return ensemble_train_probs, ensemble_val_probs, xgb_model



def train_model(df):
    X = df.drop(columns='object_id').values
    y = df['object_id'].map(train_log.set_index('object_id')['target']).values
    
    # drop nan's
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Hyperparameter search
    results = hyperparameter_calibration(X_train, y_train, X_val, y_val)

    best_model = results[0]
    print(f"\n=== Using Best Configuration ===")
    print(f"hidden_dim={best_model['hidden_dim']}, alpha={best_model['alpha']}, gamma={best_model['gamma']}")
    
    # ensemble with XGBoost
    ensemble_train_probs, ensemble_val_probs, xgb_model = ensemble_nn_xgb(
        X_train, y_train, X_val, y_val,
        best_model['train_probs'],
        best_model['val_probs']
    )
    
    return best_model['model'], xgb_model, ensemble_train_probs, ensemble_val_probs, best_model['scaler'] 


# use predetermined model for training
def train_selected_model(df): 
    X = df.drop(columns='object_id').values
    y = df['object_id'].map(train_log.set_index('object_id')['target']).values
    
    # drop nan's
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model, train_probs, val_probs, val_loss, scaler = train_individual_model(
        X_train, y_train, X_val, y_val, 
        hidden_dim=256, 
        alpha=0.1,
        gamma=2.5,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )

    # ensemble with XGBoost
    ensemble_train_probs, ensemble_val_probs, xgb_model = ensemble_nn_xgb(
        X_train, y_train, X_val, y_val,
        train_probs,
        val_probs
    )
    
    return model, xgb_model, ensemble_train_probs, ensemble_val_probs, scaler

        


def predict_test_set(neural_network_model, xgb_model, scaler, test_features_df, test_log, output_path='submission.csv'):
    X_test = test_features_df.drop(columns='object_id').values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = scaler.transform(X_test)
    neural_network_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        nn_logits = neural_network_model(X_test_tensor)
        neural_network_probs = torch.sigmoid(nn_logits).cpu().numpy().flatten()  

    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    ensemble_probs = (neural_network_probs * BIAS_WEIGHT_NN + xgb_probs * BIAS_WEIGHT_XGB)    
    predictions = (ensemble_probs >= THRESHOLD).astype(int) 
    submission = pd.DataFrame({
        'object_id': test_features_df['object_id'],
        'prediction': predictions
    })

    final_submission = test_log[['object_id']].merge(
        submission, on='object_id', how='left'
    )
    final_submission['prediction'] = final_submission['prediction'].fillna(0).astype(int)
    final_submission[['object_id', 'prediction']].to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total objects: {len(final_submission)}")
    print(f"Predicted TDEs: {final_submission['prediction'].sum()}")
    print(f"Predicted not-TDEs: {(final_submission['prediction'] == 0).sum()}")



# just testing this here, I'll move things around tomorrow
train_features_df, train_log = load_process_training_data()
test_features_df, test_log = load_process_test_data()

nn_model, xgb_model, ensemble_train_probs, ensemble_val_probs, scaler = train_selected_model(train_features_df)

predict_test_set(
    neural_network_model=nn_model,
    xgb_model=xgb_model, 
    scaler=scaler,  
    test_features_df=test_features_df,
    test_log=test_log,
    output_path='submission.csv'
)