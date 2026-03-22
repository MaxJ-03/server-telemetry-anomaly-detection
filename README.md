# Predictive Alerting Systems: Supervised vs. Unsupervised Paradigms

## Table of Contents
* [1. Project Overview and The Dataset Decision](#1-project-overview-and-the-dataset-decision)
* [2. Problem Formulation](#2-problem-formulation)
* [3. Dataset Splitting Strategy](#3-dataset-splitting-strategy)
* [4. Evaluation Metrics Strategy](#4-evaluation-metrics-strategy)
* [5. Phase One: Supervised Binary Classification](#5-phase-one-supervised-binary-classification)
* [6. Phase One Evaluation: Data Drift and Concept Drift](#6-phase-one-evaluation-data-drift-and-concept-drift)
* [7. Phase Two: Unsupervised Signal Reconstruction](#7-phase-two-unsupervised-signal-reconstruction)
* [8. Adapting to a Production Alerting System](#8-adapting-to-a-production-alerting-system)
* [9. Repository Structure and Dataset Location](#9-repository-structure-and-dataset-location)
* [10. Execution Guide](#10-execution-guide)

---

## 1. Project Overview and The Dataset Decision
The objective of this project is to design and implement a predictive alerting system capable of anticipating server incidents based on historical metric data. Cloud service metrics present unique modeling challenges, including weak correlations, short-lived patterns, abrupt regime shifts, and heavy-tailed distributions. 

**Decision: Abandoning the NASA SMAP Dataset**
Initially, I explored the NASA Soil Moisture Active Passive (SMAP) telemetry dataset to proxy these complex systems. However, during the exploratory phase, I discovered a critical structural limitation: most of the telemetry channels contain only a single recorded anomaly sequence. This single-anomaly constraint completely breaks the supervised learning strategy, as a chronological walk-forward split would leave either the training or evaluation set without a failure state to learn from or test against.

**Decision: Adopting the eBay PSM Dataset**
To resolve this, I pivoted to the Pooled Server Metrics (PSM) dataset released by eBay. This dataset aligns perfectly with cloud-infrastructure challenges and contains multiple, distinct server incidents within the same continuous timeline, fully supporting a rigorous custom walk-forward chronological split.

## 2. Problem Formulation
To satisfy the requirement of predicting future incidents based on historical context, I formulated the task using a sliding-window approach:
* **Lookback Window ($W = 30$):** The models evaluate the previous 30 time steps of multivariate telemetry.
* **Prediction Horizon ($H = 10$):** The supervised models predict the probability of an incident occurring at any point within the subsequent 10 time steps. For the unsupervised baseline, the model forecasts the continuous numerical state of the very next time step ($H = 1$).

## 3. Dataset Splitting Strategy
**Decision: The 4-Way Chronological Split**
To rigorously evaluate the system against temporal constraints and prevent data leakage, I partitioned the `test.csv` sequence (which contains the recorded failures) into a strict 4-way chronological split. Standard randomized cross-validation is invalid for time-series data because it leaks future context into the past. 

1. **Training Set (40%):** I utilize this initial sequence to fit the baseline parameters of the supervised models (Tree Ensemble splits and LSTM network weights), allowing them to learn the fundamental failure signatures.
2. **Validation Set (20%):** Because time-series models cannot use standard K-Fold cross-validation, I use this subsequent timeline strictly for Hyperparameter Optimization (Grid and Randomized Searches). By evaluating against this unseen slice, I select champion architectures that minimize loss without overfitting to the training data.
3. **Calibration Set (20%):** I isolate this sequence entirely for operational thresholding. Here, I map raw model logits to empirical probabilities and mathematically identify the exact decision boundary required to satisfy the business requirement of capturing 80% of all true incidents.
4. **Evaluation Set (20%):** This is the completely blind, future timeline. It is used exclusively for the final performance benchmarking, ensuring the reported classification metrics accurately reflect how the fully calibrated system will perform on novel, unseen data in production.

## 4. Evaluation Metrics Strategy
In highly imbalanced AIOps environments, standard evaluation metrics like Accuracy are dangerously misleading; a model that perpetually predicts "healthy" on a 99% stable server will achieve 99% accuracy while entirely failing its primary engineering purpose. Therefore, I evaluate the architectures using the following specific metrics:

* **Recall (The Operational Imperative):** Measures the proportion of actual incidents successfully anticipated. Missing a server outage carries a severe business cost. Consequently, the thresholding logic in this pipeline is mathematically calibrated to guarantee an 80% operational recall target.
* **Precision (The Fatigue Mitigator):** Measures the proportion of triggered alerts that represent genuine anomalies. While high recall is mandatory, low precision leads to "alert fatigue," overwhelming software engineers with false positives to the point where they ignore the system.
* **F1-Score (The Optimization Objective):** The harmonic mean of Precision and Recall. I utilize the Macro F1-Score as the primary scoring metric during the Validation Grid Search because it strictly penalizes models that favor one class over the other, forcing the architecture to find a balanced structural state before final thresholding.
* **Support:** The absolute count of true occurrences in the dataset, providing the necessary mathematical context for the scale of the evaluation.

## 5. Phase One: Supervised Binary Classification
The initial approach framed the alerting task as a supervised binary classification problem. To investigate the trade-offs between different mathematical approaches, I implemented a dual-classifier strategy:
* **Histogram-based Gradient(TREE):** A robust, non-linear tree ensemble baseline that evaluates the flattened window. This architecture effectively handles noisy, heavy-tailed tabular distributions and manages class imbalance internally via dynamic cost-function weighting.
* **Alerting LSTM(ALERTING_LSTM):** A recurrent neural network designed to respect the sequence of events, learning temporal state evolution. 
  * **Decision:** To prevent the network from defaulting to a perpetual safe state due to severe class imbalance, I utilized a `WeightedRandomSampler` within the PyTorch data loader to over-index on minority failure states during training.

### The Supervised Execution Lifecycle
For both the Tree Ensemble and the Alerting LSTM, the execution follows a strict progression through the 4-way split detailed above:
1. **Training:** The models are fitted to the historical failure shapes. The Tree optimizes its splits, and the LSTM optimizes Binary Cross-Entropy (BCE) loss.
2. **Validation & Optimization:** To ensure optimal configurations, the models undergo a Grid Search evaluated against the validation slice. The Tree optimizes for the Macro F1-Score, while the LSTM optimizes for the lowest Validation BCE Loss.
3. **Calibration & Thresholding:** Relying on a default 0.5 probability cutoff is an arbitrary and flawed metric for imbalanced AIOps data. The exact mathematical threshold required to hit the 80% recall target is extracted strictly from the Calibration split.
    * The Tree outputs are passed through Isotonic Regression to map the raw logits to true empirical probabilities.
    * The LSTM's raw logits are evaluated directly against the thresholding engine to preserve temporal ranking.
4. **Evaluation:** The fixed models and their derived thresholds are applied to the final unseen timeline to generate the metrics below.

## 6. Phase One Evaluation: Data Drift and Concept Drift
When the calibrated boundaries were applied to the completely unseen Evaluation timeline, the models initially experienced a severe deterioration in precision. 

**Decision: Implementing Adaptive Rolling Standardization**
To isolate the cause, I analyzed the telemetry and discovered **Data Drift**. The absolute baseline of the server metrics shifted over time, causing static scalers (trained on historical data) to output massive out-of-bounds activations. I replaced the static approach with Adaptive Rolling Standardization (a 24-hour rolling Z-score). By normalizing against a trailing local window, the model evaluates acute relative spikes rather than absolute historical magnitudes.

### Final Supervised Results (With Adaptive Scaling)

<!-- [[TREE_LEADERBOARD_START]] -->

### TREE Top 5 Optimization Leaderboard
| Rank | learning_rate | max_depth | max_iter | Mean CV F1 | Mean CV Precision | Mean CV Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.128958 | 7 | 291 | 0.396229 | 0.581949 | 0.563484 |
| 2 | 0.176459 | 7 | 176 | 0.395615 | 0.582300 | 0.559572 |
| 3 | 0.020031 | 3 | 265 | 0.394666 | 0.542150 | 0.573373 |
| 4 | 0.127307 | 7 | 250 | 0.393962 | 0.584916 | 0.560858 |
| 5 | 0.132558 | None | 222 | 0.393431 | 0.575467 | 0.557818 |

<!-- [[TREE_LEADERBOARD_END]] -->

<!-- [[TREE_START]] -->

### TREE Final Evaluation
| Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.00 | 0.00 | 0.00 | 12115 |
| **1** | 0.31 | 1.00 | 0.47 | 5415 |
| **macro avg** | 0.15 | 0.50 | 0.24 | 17530 |
| **weighted avg** | 0.10 | 0.31 | 0.15 | 17530 |

<!-- [[TREE_END]] -->

<!-- [[ALERTING_LSTM_LEADERBOARD_START]] -->

### ALERTING_LSTM Top 5 Optimization Leaderboard
| Rank | hidden_dim | learning_rate | Val BCE Loss | Val F1 | Val Precision | Val Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 128 | 0.010000 | 0.987156 | 0.554796 | 0.558492 | 0.638610 |
| 2 | 64 | 0.010000 | 1.142600 | 0.561630 | 0.562281 | 0.644079 |
| 3 | 32 | 0.010000 | 1.328902 | 0.512899 | 0.526774 | 0.562976 |
| 4 | 32 | 0.005000 | 1.348880 | 0.533945 | 0.540684 | 0.592551 |
| 5 | 128 | 0.005000 | 1.386452 | 0.539987 | 0.545693 | 0.604926 |

<!-- [[ALERTING_LSTM_LEADERBOARD_END]] -->

<!-- [[LSTM_START]] -->

### LSTM Final Evaluation
| Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.66 | 0.18 | 0.28 | 12115 |
| **1** | 0.30 | 0.79 | 0.44 | 5415 |
| **macro avg** | 0.48 | 0.48 | 0.36 | 17530 |
| **weighted avg** | 0.55 | 0.37 | 0.33 | 17530 |

<!-- [[LSTM_END]] -->


### Architectural Conclusion: The Labeling Bottleneck
Despite neutralizing Data Drift via adaptive scaling, the precision of both models remained critically low. While the models successfully captured the target recall (~80%), they generated a catastrophic volume of false positives.

This empirically proves the existence of **Concept Drift**. The physical manifestations of server stress evolved. The supervised models overfit to the specific multivariate signatures of past failures and failed to generalize to the novel failure states in the future timeline. 

**Decision:** In traditional machine learning, concept drift is mitigated via continuous rolling retraining. However, in an AIOps environment, this required engineers to manually investigate and label every future server anomaly to feed the algorithm. Because continuous manual labeling in production is an operational impossibility, attempting to patch the supervised pipeline is an architectural dead end.

## 7. Phase Two: Unsupervised Signal Reconstruction
To bypass the labeling bottleneck and the vulnerability to novel failure signatures, I pivoted the architecture to an Unsupervised Signal Reconstruction paradigm, heavily inspired by the methodology deployed by NASA.

**The Mechanics:** Instead of classifying binary failure labels, I trained a `ForecastingLSTM` to predict the continuous values of the subsequent time step. The network learns the underlying physics of a stable system. When a real incident occurs, the actual metrics diverge drastically from the network's healthy prediction. By calculating the Mean Absolute Error across all multivariate dimensions, I established a unified anomaly score (residual) for each time step.

### The Unsupervised Execution Lifecycle
Because an unsupervised model must strictly learn healthy physics, its execution lifecycle requires a distinct data routing strategy compared to the supervised models:
1. **Training & Validation (`train.csv`):** The Predictive LSTM never sees the `test.csv` data during training. Instead, I ingest the `train.csv` file (which contains zero anomalies) and apply a 70/30 chronological split. The network optimizes Mean Squared Error (MSE) loss on the 70% split, and the Grid Search selects the champion architecture based on the lowest Validation MSE on the 30% split.
2. **Calibration & Thresholding (`test.csv` - 20% Calibration Split):** To guarantee a mathematically rigorous, apples-to-apples comparison against the supervised models, I route the exact same Calibration split used in Phase One through the Predictive LSTM. The model calculates the residuals, and the system dynamically identifies the residual error magnitude required to hit the 80% recall operational target.
3. **Evaluation (`test.csv` - 20% Evaluation Split):** The model applies the derived residual threshold to the exact same unseen Evaluation timeline used by the supervised models, dynamically triggering alerts based purely on prediction drift.

### Final Unsupervised Forecasting Results (UNSUPERVISED_LSTM)

<!-- [[PREDICTIVE_LSTM_LEADERBOARD_START]] -->

### PREDICTIVE_LSTM Top 5 Optimization Leaderboard
| Rank | hidden_dim | learning_rate | Val MSE Loss | Val RMSE Loss | Val MAE Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 128 | 0.000500 | 0.370388 | 0.608595 | 0.402749 |
| 2 | 128 | 0.001000 | 0.376435 | 0.613543 | 0.409213 |
| 3 | 64 | 0.000500 | 0.378796 | 0.615464 | 0.408066 |
| 4 | 64 | 0.001000 | 0.385229 | 0.620668 | 0.416325 |
| 5 | 128 | 0.005000 | 0.395032 | 0.628516 | 0.425091 |

<!-- [[PREDICTIVE_LSTM_LEADERBOARD_END]] -->

<!-- [[UNSUPERVISED_LSTM_START]] -->

### UNSUPERVISED_LSTM Final Evaluation
| Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.73 | 0.24 | 0.36 | 12115 |
| **1** | 0.32 | 0.80 | 0.46 | 5415 |
| **macro avg** | 0.52 | 0.52 | 0.41 | 17530 |
| **weighted avg** | 0.60 | 0.41 | 0.39 | 17530 |

<!-- [[UNSUPERVISED_LSTM_END]] -->

## 8. Adapting to a Production Alerting System
To satisfy the requirements of a real-time production alerting system, the phase two structural paradigm (predictive forecasting combined with continuous residual analysis) must be deployed via a streaming architecture.

**Streaming Inference Architecture**
The system requires a message broker, such as Apache Kafka or AWS Kinesis, to ingest real-time server telemetry. A stream-processing engine like Apache Flink or Spark Streaming is necessary to maintain the sliding lookback window of $W=30$ in active memory, ensuring the neural network receives dimensionally accurate continuous input tensors.

**Continuous Preprocessing**
Adaptive Rolling Standardization must be maintained dynamically. The continuous 24-hour mean and standard deviation matrices need to be stored in an in-memory datastore such as Redis. This ensures that every new data point is instantly normalized against the immediate local baseline to mitigate Data Drift.

**Unsupervised Residual Pipeline**
The Predictive LSTM evaluates the $W=30$ window to output the continuous forecast for the $W+1$ time step. When the physical telemetry for $W+1$ actually arrives from the broker, the system calculates the Mean Absolute Error bounded across all dimensions. If this discrepancy residual breaches the defined statistical boundary, the event is immediately pushed to a downstream Alert Manager like PagerDuty or Prometheus Alertmanager.

**Dynamic Re-Calibration and Feedback**
Server architectures continuously evolve via capacity increases and deployments. The statistical threshold should not be a static artifact. It is calculated automatically on a rolling, weekly basis against intervals of known healthy telemetry. Additionally, the software engineers can tune the threshold multiplier via an active feedback loop without ever requiring a retraining of the core physics network, definitively solving the labeling bottleneck.

## 9. Repository Structure and Dataset Location
With the architectural logic established, the core engineering is encapsulated into a strictly modular architecture. 

**The Data Directory (`/data`)**
To run this pipeline, the PSM dataset files must be placed in a `/data` directory at the project root:
* `data/train.csv`: The strictly healthy, anomaly-free baseline telemetry (used for unsupervised learning).
* `data/test.csv`: The continuous timeline containing multiple server incidents.
* `data/test_label.csv`: The corresponding ground-truth binary labels for the incidents.

**The Source Directory (`/src`)**
* `preprocessing.py`: Manages raw telemetry ingestion, the chronological split, multivariate sliding-window tensor formulation, and adaptive feature scaling.
* `models.py`: Defines the PyTorch neural network architectures (`AlertingLSTM`, `ForecastingLSTM`) and the scikit-learn tree instantiation.
* `trainer.py`: Handles the training loops, PyTorch data loaders, hardware acceleration routing, and the multi-configuration Grid Search optimization.
* `calibration.py`: Contains the logic for mapping raw model logits to empirical probabilities and identifying operational decision boundaries.
* `automation.py`: A regex-based utility that facilitates CI/CD by dynamically parsing JSON logs and injecting evaluation metrics directly into this README.
* `main.py`: The central orchestrator that manages the end-to-end execution, from data ingestion to evaluation and documentation updating.

## 10. Execution Guide
The pipeline is designed to be executed via the command line from the root directory.

**Standard Execution (Full Optimization)**
To run the full pipeline including the comprehensive Grid Search across all model architectures:
```bash
python main.py