import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from src.preprocessing import load_psm_data, apply_rolling_standardization, _create_windows
from src.models import get_tree_ensemble, AlertingLSTM, ForecastingLSTM
from src.trainer import (
    optimize_tree_ensemble, get_tree_probabilities,
    optimize_lstm, get_lstm_probabilities,
    optimize_forecasting_lstm, get_forecasting_residuals,
    train_lstm, train_forecasting_lstm,
    get_balanced_loader, get_sequential_loader
)
from src.calibration import fit_tree_calibrator, apply_calibration, find_empirical_threshold

def parse_args():
    parser = argparse.ArgumentParser(description="Antigravity AIOps Pipeline Execution")
    parser.add_argument('--skip-optimization', action='store_true', help="Bypass Grid Search and use fixed configurations")
    
    # Tree parameters
    parser.add_argument('--tree-max-depth', type=int, default=5, help="Tree max depth (used if skipping optimization)")
    parser.add_argument('--tree-lr', type=float, default=0.1, help="Tree learning rate (used if skipping optimization)")
    parser.add_argument('--tree-max-iter', type=int, default=200, help="Tree max iterations (used if skipping optimization)")
    
    # Alerting LSTM parameters
    parser.add_argument('--lstm-hidden-dim', type=int, default=64, help="Alerting LSTM hidden dimension")
    parser.add_argument('--lstm-lr', type=float, default=0.001, help="Alerting LSTM learning rate")
    parser.add_argument('--lstm-layers', type=int, default=2, help="Alerting LSTM number of layers")
    parser.add_argument('--lstm-epochs', type=int, default=10, help="Alerting LSTM training epochs")
    
    # Predictive LSTM parameters
    parser.add_argument('--pred-hidden-dim', type=int, default=64, help="Predictive LSTM hidden dimension")
    parser.add_argument('--pred-lr', type=float, default=0.001, help="Predictive LSTM learning rate")
    parser.add_argument('--pred-layers', type=int, default=2, help="Predictive LSTM number of layers")
    parser.add_argument('--pred-epochs', type=int, default=10, help="Predictive LSTM training epochs")
    
    return parser.parse_args()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_best_configurations(config_data, filepath="results/best_configurations.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=4, cls=NpEncoder)
    print(f"Architectural log persisted to {filepath}")

def execute_pipeline(args):
    results_log = {}
    pipeline_start = time.time()
    
    print("\n--- Starting Data Preparation & Chronological Split ---")
    data_start = time.time()
    # Data Preparation
    full_data, full_labels = load_psm_data('data/test.csv', 'data/test_label.csv')
    scaled_data = apply_rolling_standardization(full_data)

    # 4-Way Chronological Split
    ratios = [0.4, 0.2, 0.2, 0.2]
    idxs = np.cumsum([int(len(scaled_data) * r) for r in ratios])
    train_raw, val_raw, cal_raw, eval_raw = np.split(scaled_data, idxs[:-1])
    train_lab, val_lab, cal_lab, eval_lab = np.split(full_labels, idxs[:-1])

    X_train, y_train = _create_windows(train_raw, train_lab, 30, 10)
    X_val, y_val = _create_windows(val_raw, val_lab, 30, 10)
    X_cal, y_cal = _create_windows(cal_raw, cal_lab, 30, 10)
    X_eval, y_eval = _create_windows(eval_raw, eval_lab, 30, 10)
    print(f"Data Preparation completed in {time.time() - data_start:.2f}s.")

    # Tree Optimization
    if args.skip_optimization:
        print("\n--- Phase 1A: Training Tree Ensemble (Fixed Config) ---")
        tree_champion = get_tree_ensemble()
        tree_champion.set_params(max_depth=args.tree_max_depth, learning_rate=args.tree_lr, max_iter=args.tree_max_iter)
        
        num_samples = X_train.shape[0]
        flattened_features = X_train.shape[1] * X_train.shape[2]
        X_train_flat = X_train.reshape(num_samples, flattened_features)
        
        start_time = time.time()
        tree_champion.fit(X_train_flat, y_train.ravel())
        print(f"Tree training completed in {time.time() - start_time:.2f}s.")
        tree_params = tree_champion.get_params()
        tree_top5 = []
    else:
        print("\n--- Phase 1A: Tree Ensemble Optimization ---")
        phase1a_start = time.time()
        tree_champion, tree_params, tree_top5 = optimize_tree_ensemble(get_tree_ensemble(), X_train, y_train)
        print(f"Tree Ensemble Optimization completed in {time.time() - phase1a_start:.2f}s.")
        
    tree_cal_probs = get_tree_probabilities(tree_champion, X_cal)
    tree_calibrator = fit_tree_calibrator(tree_cal_probs, y_cal.ravel())
    tree_threshold, _, _ = find_empirical_threshold(apply_calibration(tree_calibrator, tree_cal_probs), y_cal.ravel(), 0.80)
    tree_preds = (apply_calibration(tree_calibrator, get_tree_probabilities(tree_champion, X_eval)) >= tree_threshold).astype(int)
    tree_rep = classification_report(y_eval.ravel(), tree_preds, output_dict=True, zero_division=0)
    results_log["TREE_ENSEMBLE"] = {"params": tree_params, "metrics": tree_rep, "top_5_configs": tree_top5}

    # Alerting LSTM Optimization
    if args.skip_optimization:
        print("\n--- Phase 1B: Training Alerting LSTM (Fixed Config) ---")
        lstm_champion = AlertingLSTM(input_dim=X_train.shape[2], hidden_dim=args.lstm_hidden_dim, num_layers=args.lstm_layers)
        train_loader = get_balanced_loader(X_train, y_train)
        duration = train_lstm(lstm_champion, train_loader, epochs=args.lstm_epochs, learning_rate=args.lstm_lr)
        print(f"Config [HD: {args.lstm_hidden_dim}, LR: {args.lstm_lr}] | Time: {duration:.2f}s")
        lstm_params = {'hidden_dim': args.lstm_hidden_dim, 'learning_rate': args.lstm_lr, 'num_layers': args.lstm_layers}
        lstm_top5 = []
    else:
        print("\n--- Phase 1B: Alerting LSTM Optimization ---")
        phase1b_start = time.time()
        lstm_champion, lstm_params, lstm_top5 = optimize_lstm(X_train, y_train, X_val, y_val, X_train.shape[2])
        print(f"Alerting LSTM Optimization completed in {time.time() - phase1b_start:.2f}s.")
        
    lstm_cal_probs = get_lstm_probabilities(lstm_champion, X_cal)
    lstm_threshold, _, _ = find_empirical_threshold(lstm_cal_probs, y_cal.ravel(), 0.80)
    lstm_preds = (get_lstm_probabilities(lstm_champion, X_eval) >= lstm_threshold).astype(int)
    lstm_rep = classification_report(y_eval.ravel(), lstm_preds, output_dict=True, zero_division=0)
    results_log["ALERTING_LSTM"] = {"params": lstm_params, "metrics": lstm_rep, "top_5_configs": lstm_top5}

    # Predictive LSTM Optimization
    print("\n--- Preparing Healthy Data for Phase 2 ---")
    data_p2_start = time.time()
    healthy_data = pd.read_csv('data/train.csv').drop(columns=['timestamp_(min)', 'timestamp_'], errors='ignore').values
    healthy_scaled = apply_rolling_standardization(healthy_data)
    
    h_idx = int(len(healthy_scaled) * 0.7)
    X_h_train, _ = _create_windows(healthy_scaled[:h_idx], np.zeros(h_idx), 30, 1)
    y_h_train = healthy_scaled[30:h_idx]
    X_h_val, _ = _create_windows(healthy_scaled[h_idx:], np.zeros(len(healthy_scaled)-h_idx), 30, 1)
    y_h_val = healthy_scaled[h_idx+30:]
    print(f"Healthy Data Preparation completed in {time.time() - data_p2_start:.2f}s.")

    if args.skip_optimization:
        print("\n--- Phase 2: Training Predictive LSTM (Fixed Config) ---")
        pred_champion = ForecastingLSTM(input_dim=X_train.shape[2], hidden_dim=args.pred_hidden_dim, num_layers=args.pred_layers)
        forecast_loader = get_sequential_loader(X_h_train, y_h_train[:len(X_h_train)], batch_size=128)
        duration = train_forecasting_lstm(pred_champion, forecast_loader, epochs=args.pred_epochs, learning_rate=args.pred_lr)
        print(f"Config [HD: {args.pred_hidden_dim}, LR: {args.pred_lr}] | Time: {duration:.2f}s")
        pred_params = {'hidden_dim': args.pred_hidden_dim, 'learning_rate': args.pred_lr, 'num_layers': args.pred_layers}
        pred_top5 = []
    else:
        print("\n--- Phase 2: Predictive LSTM Optimization ---")
        phase2_start = time.time()
        pred_champion, pred_params, pred_top5 = optimize_forecasting_lstm(X_h_train, y_h_train[:len(X_h_train)], X_h_val, y_h_val[:len(X_h_val)], X_train.shape[2])
        print(f"Predictive LSTM Optimization completed in {time.time() - phase2_start:.2f}s.")
    
    y_cal_cont = cal_raw[30:30+len(X_cal)]
    y_eval_cont = eval_raw[30:30+len(X_eval)]
    cal_res = get_forecasting_residuals(pred_champion, X_cal[:len(y_cal_cont)], y_cal_cont)
    eval_res = get_forecasting_residuals(pred_champion, X_eval[:len(y_eval_cont)], y_eval_cont)
    unsup_thresh, _, _ = find_empirical_threshold(cal_res, y_cal.ravel()[:len(cal_res)], 0.80)
    print(f"Derived Unsupervised Empirical Threshold (Target 80% Recall): {unsup_thresh:.4f}")
    unsup_preds = (eval_res >= unsup_thresh).astype(int)
    unsup_rep = classification_report(y_eval.ravel()[:len(eval_res)], unsup_preds, output_dict=True, zero_division=0)
    results_log["PREDICTIVE_LSTM"] = {"params": pred_params, "threshold": float(unsup_thresh), "metrics": unsup_rep, "top_5_configs": pred_top5}

    print(f"\n--- Entire Pipeline Completed in {time.time() - pipeline_start:.2f}s. ---")

    save_best_configurations(results_log)
    
    # Decoupled README injection
    from src.automation import update_readme_from_json
    update_readme_from_json()

if __name__ == "__main__":
    args = parse_args()
    execute_pipeline(args)
