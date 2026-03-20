import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

def get_tree_ensemble():
    """
    Returns a configured Histogram-based Gradient Boosting Classifier.
    Used as the non-neural network baseline for supervised alerting.
    """
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )

class AlertingLSTM(nn.Module):
    """
    A recurrent neural network for multivariate time-series classification.
    Processes historical windows and outputs a raw score for incident probability.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super(AlertingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the final state of the sequence for the prediction
        final_timestep_out = out[:, -1, :]
        out = self.dropout(final_timestep_out)
        logits = self.fc(out)
        return logits