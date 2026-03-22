import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
from src.models import AlertingLSTM, ForecastingLSTM

class ClassificationDataset(Dataset):
    """
    I am wrapping the 3D telemetry tensors and binary targets for the supervised alerting task.
    """
    def __init__(self, x_windows, y_targets):
        self.x_data = torch.tensor(x_windows, dtype=torch.float32)
        self.y_data = torch.tensor(y_targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class ForecastingDataset(Dataset):
    """
    I am creating a dedicated dataset for continuous multivariate forecasting.
    """
    def __init__(self, x_windows, y_targets):
        self.x_data = torch.tensor(x_windows, dtype=torch.float32)
        self.y_data = torch.tensor(y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def get_balanced_loader(X_train, y_train, batch_size=64):
    dataset = ClassificationDataset(X_train, y_train)
    class_counts = np.bincount(y_train)
    num_samples = len(y_train)
    class_weights = num_samples / (len(class_counts) * class_counts)
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

def get_sequential_loader(X_data, y_data, batch_size=64, shuffle=True):
    dataset = ForecastingDataset(X_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_lstm(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
    return time.time() - start_time

def optimize_lstm(X_train, y_train, X_val, y_val, input_dim):
    """
    I am executing a grid search to identify the optimal Alerting LSTM architecture.
    The champion model is selected based on the lowest Binary Cross Entropy loss on the validation split.
    """
    param_grid = {'hidden_dim': [32, 64, 128], 'learning_rate': [0.0005, 0.001, 0.005, 0.01]}
    best_loss, best_model, best_params = float('inf'), None, None
    history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = get_balanced_loader(X_train, y_train)
    val_loader = DataLoader(ClassificationDataset(X_val, y_val), batch_size=256)

    print(f"Initiating Alerting LSTM Grid Search (12 Configs)...")
    for hd in param_grid['hidden_dim']:
        for lr in param_grid['learning_rate']:
            model = AlertingLSTM(input_dim=input_dim, hidden_dim=hd, num_layers=2)
            duration = train_lstm(model, train_loader, learning_rate=lr)
            
            model.eval()
            val_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    val_loss += criterion(logits, by).item()
                    all_preds.extend((torch.sigmoid(logits) >= 0.5).int().cpu().numpy())
                    all_trues.extend(by.int().cpu().numpy())
            
            avg_loss = val_loss / len(val_loader)
            p, r, f, _ = precision_recall_fscore_support(all_trues, all_preds, average='macro', zero_division=0)
            print(f"Config [HD: {hd}, LR: {lr}] | Time: {duration:.2f}s | Val BCE: {avg_loss:.4f} | F1: {f:.4f}")
            
            history.append({
                'params': {'hidden_dim': hd, 'learning_rate': lr}, 
                'metric': avg_loss,
                'metrics': {
                    'Val BCE Loss': avg_loss,
                    'Val F1': f,
                    'Val Precision': p,
                    'Val Recall': r
                }
            })
            
            if avg_loss < best_loss:
                best_loss, best_model, best_params = avg_loss, model, {'hidden_dim': hd, 'learning_rate': lr}
                
    top_5 = sorted(history, key=lambda x: x['metric'])[:5]
    return best_model, best_params, top_5

def train_forecasting_lstm(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
    return time.time() - start_time

def optimize_forecasting_lstm(X_train, y_train, X_val, y_val, input_dim):
    """
    I am optimizing the Predictive LSTM by searching for the configuration that minimizes 
    Mean Squared Error on the healthy validation telemetry.
    """
    param_grid = {'hidden_dim': [32, 64, 128], 'learning_rate': [0.0005, 0.001, 0.005, 0.01]}
    best_loss, best_model, best_params = float('inf'), None, None
    history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    
    train_loader = get_sequential_loader(X_train, y_train)
    val_loader = DataLoader(ForecastingDataset(X_val, y_val), batch_size=256)

    print(f"Initiating Predictive LSTM Grid Search (12 Configs)...")
    for hd in param_grid['hidden_dim']:
        for lr in param_grid['learning_rate']:
            model = ForecastingLSTM(input_dim=input_dim, hidden_dim=hd, num_layers=2)
            duration = train_forecasting_lstm(model, train_loader, learning_rate=lr)
            
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = model(bx)
                    val_loss += criterion(preds, by).item()
                    val_mae += nn.L1Loss()(preds, by).item()
            
            avg_mse = val_loss / len(val_loader)
            avg_mae = val_mae / len(val_loader)
            print(f"Config [HD: {hd}, LR: {lr}] | Time: {duration:.2f}s | Val MSE: {avg_mse:.6f}")
            
            history.append({
                'params': {'hidden_dim': hd, 'learning_rate': lr}, 
                'metric': avg_mse,
                'metrics': {
                    'Val MSE Loss': avg_mse,
                    'Val RMSE Loss': math.sqrt(avg_mse),
                    'Val MAE Loss': avg_mae
                }
            })
            
            if avg_mse < best_loss:
                best_loss, best_model, best_params = avg_mse, model, {'hidden_dim': hd, 'learning_rate': lr}

    top_5 = sorted(history, key=lambda x: x['metric'])[:5]
    return best_model, best_params, top_5

def get_forecasting_residuals(model, X_data, y_true):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dataset = ForecastingDataset(X_data, y_true)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_residuals = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            error = torch.mean(torch.abs(predictions - batch_y), dim=1)
            all_residuals.extend(error.cpu().numpy())
    return np.array(all_residuals)

def optimize_tree_ensemble(tree_model, X_train, y_train):
    num_samples = X_train.shape[0]
    flattened_features = X_train.shape[1] * X_train.shape[2]
    X_train_flat = X_train.reshape(num_samples, flattened_features)
    y_train_flat = y_train.ravel()
    
    param_distributions = {'max_iter': randint(100, 300), 'learning_rate': uniform(0.01, 0.2), 'max_depth': [3, 5, 7, None]}
    search = RandomizedSearchCV(
        estimator=tree_model, 
        param_distributions=param_distributions, 
        n_iter=50, 
        cv=TimeSeriesSplit(n_splits=3), 
        scoring={'f1_macro': 'f1_macro', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'},
        refit='f1_macro',
        n_jobs=-1,
        verbose=3
    )
    
    start_time = time.time()
    search.fit(X_train_flat, y_train_flat)
    print(f"Tree search completed in {time.time() - start_time:.2f}s.")
    
    cv_res = search.cv_results_
    sorted_indices = np.argsort(cv_res['rank_test_f1_macro'])[:5]
    top_5 = []
    for idx in sorted_indices:
        top_5.append({
            'params': cv_res['params'][idx],
            'metrics': {
                'Mean CV F1': cv_res['mean_test_f1_macro'][idx],
                'Mean CV Precision': cv_res['mean_test_precision_macro'][idx],
                'Mean CV Recall': cv_res['mean_test_recall_macro'][idx]
            }
        })
        
    return search.best_estimator_, search.best_params_, top_5

def get_tree_probabilities(tree_model, X_data):
    num_samples, lookback, features = X_data.shape
    return tree_model.predict_proba(X_data.reshape(num_samples, lookback * features))[:, 1]

def get_lstm_probabilities(model, X_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(ClassificationDataset(X_data, np.zeros(len(X_data))), batch_size=256)
    probs = []
    with torch.no_grad():
        for bx, _ in loader:
            probs.extend(torch.sigmoid(model(bx.to(device))).squeeze().cpu().numpy())
    return np.array(probs)