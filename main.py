import os
import numpy as np
from sklearn.metrics import classification_report

from src.preprocessing import (
    load_psm_data, 
    apply_rolling_standardization, 
    _create_windows
)
from src.models import get_tree_ensemble, AlertingLSTM
from src.trainer import (
    get_balanced_loader, 
    train_lstm, 
    get_lstm_probabilities, 
    train_tree_ensemble, 
    get_tree_probabilities
)
from src.calibration import (
    fit_tree_calibrator, 
    apply_calibration, 
    find_empirical_threshold
)
from src.automation import inject_metrics_to_readme

def execute_pipeline():
    """
    Orchestrates the end-to-end machine learning pipeline for the predictive alerting system.
    Handles data ingestion, adaptive scaling, model training, evaluation, and documentation updates.
    """
    
    print("Initiating predictive alerting pipeline execution...")

    test_path = 'data/test.csv'
    label_path = 'data/test_label.csv'
    full_data, full_labels = load_psm_data(test_path, label_path)

    # Applying adaptive rolling standardization to mitigate Data Drift before splitting
    print("Applying adaptive rolling standardization...")
    rolling_window = 288 
    scaled_full_data = apply_rolling_standardization(full_data, window_size=rolling_window)

    # Establishing the chronological split points to preserve the timeline
    train_ratio = 0.50
    cal_ratio = 0.20
    train_idx = int(len(scaled_full_data) * train_ratio)
    cal_idx = train_idx + int(len(scaled_full_data) * cal_ratio)

    train_raw = scaled_full_data[:train_idx]
    cal_raw = scaled_full_data[train_idx:cal_idx]
    eval_raw = scaled_full_data[cal_idx:]

    train_labels = full_labels[:train_idx]
    cal_labels = full_labels[train_idx:cal_idx]
    eval_labels = full_labels[cal_idx:]

    # Generating 3D context windows and binary horizon targets
    window_size = 30
    horizon = 10
    X_train, y_train = _create_windows(train_raw, train_labels, window_size, horizon)
    X_cal, y_cal = _create_windows(cal_raw, cal_labels, window_size, horizon)
    X_eval, y_eval = _create_windows(eval_raw, eval_labels, window_size, horizon)

    y_cal_flat = y_cal.ravel()
    y_eval_flat = y_eval.ravel()

    # Training the Tree Ensemble
    print("\nTraining Histogram Gradient Boosting Ensemble...")
    tree_model = get_tree_ensemble()
    tree_model = train_tree_ensemble(tree_model, X_train, y_train)

    tree_cal_probs = get_tree_probabilities(tree_model, X_cal)
    tree_eval_probs = get_tree_probabilities(tree_model, X_eval)

    # Training the Long Short-Term Memory Network
    print("\nTraining LSTM Network...")
    lstm_model = AlertingLSTM(input_dim=X_train.shape[2], hidden_dim=64, num_layers=2)
    train_loader = get_balanced_loader(X_train, y_train, batch_size=64)
    lstm_model = train_lstm(lstm_model, train_loader, epochs=10, learning_rate=0.001)

    lstm_cal_probs = get_lstm_probabilities(lstm_model, X_cal)
    lstm_eval_probs = get_lstm_probabilities(lstm_model, X_eval)

    # Calibrating tree probabilities and determining the optimal decision boundary
    print("\nCalibrating models and calculating empirical thresholds...")
    tree_calibrator = fit_tree_calibrator(tree_cal_probs, y_cal_flat)
    tree_cal_calibrated = apply_calibration(tree_calibrator, tree_cal_probs)
    tree_eval_calibrated = apply_calibration(tree_calibrator, tree_eval_probs)

    tree_threshold, _, _ = find_empirical_threshold(tree_cal_calibrated, y_cal_flat, target_recall=0.80)
    tree_final_predictions = (tree_eval_calibrated >= tree_threshold).astype(int)

    # Bypassing formal calibration for the neural network to preserve rank ordering
    lstm_threshold, _, _ = find_empirical_threshold(lstm_cal_probs, y_cal_flat, target_recall=0.80)
    lstm_final_predictions = (lstm_eval_probs >= lstm_threshold).astype(int)

    # Generating evaluation metrics for documentation injection
    print("\nGenerating final evaluation reports...")
    tree_report = classification_report(y_eval_flat, tree_final_predictions, output_dict=True)
    lstm_report = classification_report(y_eval_flat, lstm_final_predictions, output_dict=True)

    print("\nUpdating README.md with latest metrics...")
    inject_metrics_to_readme(tree_report, "TREE", readme_path="README.md")
    inject_metrics_to_readme(lstm_report, "LSTM", readme_path="README.md")
    
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    execute_pipeline()