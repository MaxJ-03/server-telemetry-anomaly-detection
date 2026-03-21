# Predictive Alerting Systems: Supervised vs. Unsupervised Paradigms in AIOps

## Project Overview and Dataset Strategy
The objective of this project is to design and implement a predictive alerting system capable of anticipating server incidents based on historical metric data. Cloud service metrics present unique modeling challenges, including weak correlations, short-lived patterns, abrupt regime shifts, and heavy-tailed distributions. 

Initially, I explored the NASA Soil Moisture Active Passive (SMAP) telemetry dataset to proxy these complex systems. However, during the exploratory phase, I discovered a critical structural limitation: most of the telemetry channels contain only a single recorded anomaly sequence. This single-anomaly constraint breaks the supervised learning strategy, as a chronological split would leave either the training or evaluation set without an incident to learn from.

Therefore, I focused this project on the Pooled Server Metrics (PSM) dataset released by eBay. This dataset aligns with cloud-infrastructure challenges and contains multiple, distinct server incidents within the same timeline, fully supporting a custom walk-forward chronological split.

## System Architecture
To handle the complexity of the PSM data and ensure maintainability, the project relies on a strictly modular architecture. The core engineering logic is encapsulated in the `/src` directory:
* `preprocessing.py`: Manages raw telemetry ingestion, the chronological split, multivariate sliding-window formulation, and adaptive feature scaling.
* `models.py`: Defines the neural network and tree-based architectures.
* `trainer.py`: Handles the training loops, PyTorch data loaders, and hardware acceleration routing.
* `calibration.py`: Contains the logic for mapping raw model logits to empirical probabilities and identifying operational decision boundaries.

## Phase One: Supervised Binary Classification
The initial approach framed the alerting task as a supervised, sliding-window binary classification problem. The model evaluates the previous 30 time steps of multivariate telemetry to predict the probability of an incident occurring within the subsequent 10 time steps. 

To investigate the trade-offs between different mathematical approaches, I implemented a dual-classifier strategy:
1. **Long Short-Term Memory (LSTM):** A recurrent neural network designed to respect the sequence of events, learning temporal state evolution. To prevent the network from defaulting to a perpetual safe state due to severe class imbalance, I utilized a weighted random sampling strategy within the PyTorch data loader.
2. **Histogram-based Gradient Boosting:** A robust, non-linear tree ensemble baseline that evaluates the flattened window. This architecture effectively handles noisy, heavy-tailed tabular distributions and manages class imbalance internally via dynamic cost-function weighting.

### Probability Calibration and Empirical Thresholding
Deep learning models and tree ensembles generate fundamentally different distributions of raw output scores. The business logic dictates a strict operational requirement: capturing at least 80% of all true incidents. 

To achieve this, the models underwent a calibration and thresholding phase on an isolated data split:
* **Tree Ensemble Calibration:** I applied Isotonic Regression to map the tree's raw outputs to true empirical probabilities. This non-parametric technique corrects the inherent probability distortions of the leaf nodes.
* **LSTM Empirical Thresholding:** Neural networks optimized with Cross-Entropy Loss are notoriously overconfident. Instead of formal calibration, I utilized empirical threshold identification directly on the raw logits. This preserves the exact temporal ranking the neural network learned while satisfying the operational recall requirement.

## Phase One Evaluation: Data Drift and Concept Drift
When the calibrated boundaries were applied to a completely unseen future timeline, the models experienced a severe deterioration in precision. 

To isolate the cause of this degradation, I implemented two major structural adjustments:
1. **Hardware Acceleration:** I refactored the PyTorch pipeline to offload tensor calculations to an NVIDIA GeForce GTX 1050 Ti, allowing for rapid iteration of data engineering fixes.
2. **Mitigating Data Drift:** The absolute baseline of the server metrics shifted over time, causing static scalers to output out-of-bounds activations. I replaced the static approach with Adaptive Rolling Standardization (a 24-hour rolling Z-score). By normalizing against a trailing local window, the model evaluates acute relative spikes rather than absolute historical magnitudes.

### Final Supervised Results (With Adaptive Scaling)

*(Automated metrics will be injected here by the CI/CD pipeline)*
*(Automated metrics will be injected here by the CI/CD pipeline)*
### Architectural Conclusion: The Labeling Bottleneck
Despite neutralizing Data Drift via adaptive scaling, the precision of both models remained at approximately 2%. While the models successfully captured the target recall (~77%), they generated a catastrophic volume of false positives.

This confirms that the primary obstacle is **Concept Drift**. The physical manifestations of server stress evolved. The supervised models overfit to the specific multivariate "signatures" of past failures and failed to generalize to the novel failure states in the future timeline. 

In traditional machine learning, concept drift is mitigated via Walk-Forward Validation and continuous rolling retraining. However, in an AIOps environment, this requires highly paid DevOps engineers to manually investigate and label every future server anomaly to feed the algorithm. Because continuous manual labeling in production is an operational impossibility, attempting to patch the supervised pipeline is an architectural dead end.

## Phase Two: Unsupervised Signal Reconstruction
To bypass the labeling bottleneck and the vulnerability to novel failure signatures, the architecture pivots to an Unsupervised Signal Reconstruction paradigm, heavily inspired by the methodology deployed by NASA.

Instead of classifying binary failure labels, this architecture trains an LSTM strictly on healthy, anomaly-free server operations to forecast the subsequent time step. When a real incident occurs, the actual metrics diverge drastically from the network's healthy prediction. By tracking the magnitude of these prediction errors (residuals), the system dynamically triggers alerts without ever requiring a single labeled failure state during training.