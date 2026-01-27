import torch
import joblib
import torch.nn as nn
import numpy as np
import pandas as pd

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Using a dynamic device check is safer for future GPU use
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class ModelHandler:
    def __init__(self, weights_path, scaler_x_path, scaler_y_path):
        self.model = TrafficLSTM(input_dim=6, hidden_dim=128, num_layers=2)
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # Explicitly define feature names to satisfy the RobustScaler warning
        self.feature_names = ['size', 'delay', 'bif', 'throughput_mbps', 'delay_grad', 'bif_grad']

    def predict(self, history_buffer):
        # 1. Convert to DataFrame to include feature names
        # history_buffer is expected to be a list of 10 rows, each with 6 features
        df_history = pd.DataFrame(history_buffer, columns=self.feature_names)
        
        # 2. Scale using the DataFrame
        seq_scaled = self.scaler_x.transform(df_history)
        
        # 3. Reshape for LSTM: [Batch, Seq_len, Features] -> [1, 10, 6]
        seq_reshaped = seq_scaled.reshape(1, len(history_buffer), 6)
        
        with torch.no_grad():
            input_tensor = torch.tensor(seq_reshaped).float()
            pred_scaled = self.model(input_tensor)
            
            # 4. Inverse transform the prediction
            prediction = self.scaler_y.inverse_transform(pred_scaled.numpy())[0][0]
            
        return max(0, prediction) # Throughput can't be negative
