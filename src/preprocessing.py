import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_psm_data(test_path, label_path):
    """
    Loads the raw multivariate server metrics and their corresponding incident labels.
    The timestamp column is removed to focus strictly on the metric dimensions.
    """
    test_df = pd.read_csv(test_path)
    if 'timestamp_' in test_df.columns:
        test_df = test_df.drop(columns=['timestamp_'])
    
    labels_df = pd.read_csv(label_path)
    test_labels = labels_df.values.flatten()
    test_array = test_df.values
    
    return test_array, test_labels

def prepare_supervised_data(data_array, labels, window_size=30, horizon=10):
    """
    Executes a three-way chronological split and transforms the data into sliding windows.
    Returns scaled 3D tensors for the training, calibration, and evaluation phases.
    """
    # Establishing the chronological split points to preserve the timeline
    train_ratio = 0.50
    cal_ratio = 0.20
    
    train_idx = int(len(data_array) * train_ratio)
    cal_idx = train_idx + int(len(data_array) * cal_ratio)
    
    # Slicing the arrays into the three independent phases
    train_raw = data_array[:train_idx]
    cal_raw = data_array[train_idx:cal_idx]
    eval_raw = data_array[cal_idx:]
    
    train_labels = labels[:train_idx]
    cal_labels = labels[train_idx:cal_idx]
    eval_labels = labels[cal_idx:]
    
    # Normalizing the dimensions based strictly on the training phase distribution
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    cal_scaled = scaler.transform(cal_raw)
    eval_scaled = scaler.transform(eval_raw)
    
    # Transforming the continuous sequences into 3D historical context windows
    x_train, y_train = _create_windows(train_scaled, train_labels, window_size, horizon)
    x_cal, y_cal = _create_windows(cal_scaled, cal_labels, window_size, horizon)
    x_eval, y_eval = _create_windows(eval_scaled, eval_labels, window_size, horizon)
    
    return (x_train, y_train), (x_cal, y_cal), (x_eval, y_eval)

def _create_windows(features, labels, lookback, horizon):
    """
    Internal helper to generate the 3D context windows and binary horizon targets.
    """
    x, y = [], []
    for i in range(len(features) - lookback - horizon + 1):
        x.append(features[i : i + lookback])
        
        # Checking for any incident within the prediction horizon
        future_labels = labels[i + lookback : i + lookback + horizon]
        y.append(int(np.any(future_labels == 1)))
        
    return np.array(x), np.array(y)

def apply_rolling_standardization(data_array, window_size=288):
    """
    Applies adaptive rolling standardization to mitigate Data Drift.
    Normalizes each time step based on the local mean and standard deviation
    of the preceding trailing window rather than a static historical baseline.
    """
    df = pd.DataFrame(data_array)
    rolling_mean = df.rolling(window=window_size, min_periods=1).mean()
    rolling_std = df.rolling(window=window_size, min_periods=1).std()
    
    # Handling zero standard deviation to prevent division by zero errors
    rolling_std = rolling_std.replace(0, 1e-6)
    
    standardized_df = (df - rolling_mean) / rolling_std
    
    # Filling initial null values resulting from the first few periods
    standardized_df = standardized_df.fillna(0)
    
    return standardized_df.values