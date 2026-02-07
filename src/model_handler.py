import torch
import joblib
import torch.nn as nn
import numpy as np
import pandas as pd

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        device = x.device
        # Initial states (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the final 50ms window (Many-to-One)
        return self.fc(out[:, -1, :])

class ModelHandler:
    def __init__(self, weights_path, scaler_path):
        # input_dim=9 matches our engineered feature set
        self.model = TrafficLSTM(input_dim=9, hidden_dim=128, num_layers=2)
        
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.scaler = joblib.load(scaler_path)
        
        # 750ms / 50ms = 15 time steps required for a valid prediction
        self.required_sequence_length = 15
        self.feature_names = [
            'size', 'delay', 'bif', 'iat_mean', 'iat_std', 
            'throughput_mbps', 'delay_gradient', 'bif_gradient', 'is_silent'
        ]

    def predict(self, history_buffer):
        """
        history_buffer: List or array of shape (15, 9)
        """
        if len(history_buffer) < self.required_sequence_length:
            # Not enough data yet to make a valid 750ms-context prediction
            return None

        # 1. Convert to DataFrame for Scaler compatibility
        df_history = pd.DataFrame(history_buffer, columns=self.feature_names)
        
        # 2. Scale features using the 'ruler' from training
        seq_scaled = self.scaler.transform(df_history)
        
        # 3. Reshape: [Batch=1, Seq_len=15, Features=9]
        seq_reshaped = seq_scaled.reshape(1, self.required_sequence_length, 9)
        
        with torch.no_grad():
            input_tensor = torch.tensor(seq_reshaped).float()
            pred = self.model(input_tensor).item()
            
        return max(0, pred)
