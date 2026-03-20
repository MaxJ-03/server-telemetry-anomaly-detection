import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class ServerMetricsDataset(Dataset):
    """
    PyTorch Dataset to wrap the 3D telemetry tensors and binary targets.
    Converts numpy arrays to PyTorch tensors for neural network ingestion.
    """
    def __init__(self, x_windows, y_targets):
        self.x_data = torch.tensor(x_windows, dtype=torch.float32)
        self.y_data = torch.tensor(y_targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def get_balanced_loader(X_train, y_train, batch_size=64):
    """
    Constructs a DataLoader utilizing a WeightedRandomSampler.
    Assigns higher selection probability to the minority anomaly class 
    to prevent the network from predicting a perpetual safe state.
    """
    dataset = ServerMetricsDataset(X_train, y_train)
    
    class_counts = np.bincount(y_train)
    num_samples = len(y_train)
    
    class_weights = num_samples / (len(class_counts) * class_counts)
    sample_weights = [class_weights[int(label)] for label in y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=num_samples, 
        replacement=True
    )
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

def train_lstm(model, train_loader, epochs=10, learning_rate=0.001):
    """
    Executes the training loop for the LSTM architecture.
    Utilizes Binary Cross Entropy with Logits to maintain numerical stability.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    return model

def get_lstm_probabilities(model, X_data):
    """
    Extracts the raw probability scores from the trained LSTM.
    Applies a sigmoid activation to the raw logits for the calibration phase.
    """
    model.eval()
    dataset = ServerMetricsDataset(X_data, np.zeros(len(X_data))) 
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    probabilities = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x)
            probs = torch.sigmoid(logits).squeeze().numpy()
            probabilities.extend(probs)
            
    return np.array(probabilities)

def train_tree_ensemble(tree_model, X_train, y_train):
    """
    Fits the Histogram Gradient Boosting ensemble.
    Requires flattening the 3D sequence tensors into 2D tabular arrays.
    """
    num_samples = X_train.shape[0]
    flattened_features = X_train.shape[1] * X_train.shape[2]
    
    X_train_flat = X_train.reshape(num_samples, flattened_features)
    y_train_flat = y_train.ravel()
    
    tree_model.fit(X_train_flat, y_train_flat)
    return tree_model

def get_tree_probabilities(tree_model, X_data):
    """
    Extracts the positive class probabilities from the fitted tree ensemble.
    """
    num_samples = X_data.shape[0]
    flattened_features = X_data.shape[1] * X_data.shape[2]
    X_data_flat = X_data.reshape(num_samples, flattened_features)
    
    return tree_model.predict_proba(X_data_flat)[:, 1]