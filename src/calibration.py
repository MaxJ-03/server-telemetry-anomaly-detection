import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

def fit_tree_calibrator(raw_probabilities, true_labels):
    """
    Fits an Isotonic Regression model to map raw tree ensemble scores to empirical probabilities.
    Utilizes a non-parametric approach to correct probability distortions without assuming a fixed curve shape.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probabilities, true_labels)
    return calibrator

def apply_calibration(calibrator, raw_probabilities):
    """
    Transforms uncalibrated tree model outputs into trustworthy probability metrics.
    """
    return calibrator.predict(raw_probabilities)

def find_empirical_threshold(model_scores, true_labels, target_recall=0.80):
    """
    Identifies the optimal decision boundary to achieve a specific operational recall requirement.
    Prioritizes catching incidents while maximizing precision at the defined recall target.
    """
    precisions, recalls, thresholds = precision_recall_curve(true_labels, model_scores)
    
    valid_indices = np.where(recalls >= target_recall)[0]
    
    if len(valid_indices) == 0:
        return 0.0, 0.0, 0.0
        
    best_index = valid_indices[-1] 
    optimal_threshold = thresholds[best_index] if best_index < len(thresholds) else thresholds[-1]
    
    expected_precision = precisions[best_index]
    achieved_recall = recalls[best_index]
    
    return optimal_threshold, expected_precision, achieved_recall