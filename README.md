# Predictive Alerting Systems: Supervised vs. Unsupervised Paradigms

## Project Goal and Dataset Evolution
The objective of this project is to design and implement a predictive alerting system capable of anticipating incidents based on historical metric data. Cloud service metrics present unique modeling challenges, including weak correlations, short-lived patterns, abrupt regime shifts, and heavy-tailed distributions. 

Initially, I explored the NASA Soil Moisture Active Passive (SMAP) telemetry dataset to proxy these complex systems. However, during the exploratory phase, I discovered a critical structural limitation: most of the telemetry channels contain only a single recorded anomaly sequence. This single-anomaly constraint completely breaks the supervised learning strategy. To train a predictive classifier, the dataset must be chronologically split so the model can learn precursor patterns during training while reserving a completely unseen incident for evaluation. A single incident cannot be split across both phases.

Therefore, I have focused this project on the Pooled Server Metrics (PSM) dataset released by eBay. This dataset aligns with cloud-infrastructure challenges and contains multiple, distinct server incidents within the same timeline, fully supporting a custom chronological split.

## Methodology and Data Strategy
To thoroughly investigate the optimal architecture, I am implementing two fundamentally different machine learning paradigms. Crucially, each approach requires a different strategy for utilizing the dataset files.

1. **The Supervised Binary Classifier:** The primary task is framed as a supervised, sliding-window binary classification problem: using the previous $W$ time steps of multivariate telemetry to predict the probability of an incident occurring within the subsequent $H$ time steps. For this approach, I am exclusively using the `test.csv` file. The provided `train.csv` is entirely anomaly-free; if I trained a classifier on it, the model would simply learn to predict a stable state forever. By performing a custom split directly on `test.csv`, I guarantee the classifier actually has incident data to learn from during its training phase.
2. **The Unsupervised Signal Reconstruction (The NASA Baseline):** As a comparative baseline, I am replicating NASA's original anomaly detection methodology. This approach does not predict a binary label; instead, it learns to reconstruct what a "normal" signal looks like and flags an anomaly when the actual data deviates too far from the prediction. For this approach, I will use the `train.csv` file to teach the model the baseline of healthy server operations, and then evaluate its detection capabilities on the `test.csv` file.

## Structure of the Data 
The PSM dataset consists of continuous, expert-labeled telemetry from internal application servers. The primary components are:
* **Ground Truth Labels**: The `test_label.csv` file contains a continuous binary sequence mapping the exact time steps of true server incidents (e.g., memory leaks, resource contention).
* **Telemetry Arrays**: The raw multivariate time-series data is stored in `test.csv` (and `train.csv` for the unsupervised baseline). The dataset tracks 26 concurrent dimensions representing various internal server metrics, anonymized to evaluate algorithmic generalization rather than domain-specific heuristics.

## Project Architecture and Modular Design
To ensure the code is maintainable and aligned with professional software standards, I have organized the project into a modular directory structure. The core engineering logic is encapsulated in the `/src` directory, allowing this notebook to serve as a high-level research journal focused on visualization and performance analysis.

The system is powered by two primary modules:
* **`preprocessing.py`**: Manages the ingestion of the raw telemetry, the three-way chronological split, and the multivariate sliding-window formulation.
* **`models.py`**: Contains the architectures for the supervised classifiers, allowing for a streamlined training pipeline.

## Phase One Strategy: Supervised Binary Classification
The supervised alerting task requires the model to identify patterns in a 30-step historical window that indicate a failure is likely within the next 10 steps. To investigate the trade-offs between different mathematical approaches, I am implementing a dual-classifier strategy:

1. **Long Short-Term Memory (LSTM):** This recurrent neural network is designed to respect the sequence of events, learning how the server's state evolves over the window.
2. **Histogram-based Gradient Boosting:** This acts as a robust, non-neural network baseline. It evaluates the metrics across the window as a flattened tabular structure, which is often more effective at handling the noisy, "heavy-tailed" distributions found in real-world server logs.

Both models undergo a probability calibration phase using a dedicated slice of the timeline. This ensures that the danger scores they produce are empirically accurate, helping to avoid alert fatigue in a production setting.