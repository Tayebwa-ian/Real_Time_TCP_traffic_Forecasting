import numpy as np
import pandas as pd
from collections import deque
import os
from datetime import datetime

class NetworkProcessor:
    def __init__(self, interface, duration, window_ms=50, lookback=15):
        self.interface = interface
        self.duration = duration
        self.window_ms = window_ms
        self.lookback = lookback
        self.history = deque(maxlen=lookback)
        self.log_data = [] 
        
        self.last_window_ts = 0
        self.prev_delay = 0
        self.running_bif = 0 # Track BIF across windows
        self.current_window_pkts = [] 

    def get_tshark_command(self):
        return [
            'tshark', '-i', self.interface, '-f', 'not (net 127.0.0.0/8)', '-t', 'e', '-T', 'fields',
            '-e', 'frame.time_epoch', '-e', 'frame.len', '-e', 'tcp.analysis.ack_rtt',
            '-a', f'duration:{self.duration}', '-l'
        ]

    def extract_features(self, line):
        parts = line.strip().split('\t')
        if len(parts) < 2: return None
        
        try:
            ts = float(parts[0])
            size = int(parts[1])
            # Handle empty RTT fields from Tshark
            delay = float(parts[2]) if (len(parts) > 2 and parts[2]) else self.prev_delay
            self.prev_delay = delay
        except ValueError:
            return None
        
        if self.last_window_ts == 0: self.last_window_ts = ts
        self.current_window_pkts.append([size, delay, ts])
        
        # Window Completion Check
        if (ts - self.last_window_ts) >= (self.window_ms / 1000.0):
            pkts = np.array(self.current_window_pkts)
            
            if len(pkts) > 0:
                total_size = np.sum(pkts[:, 0])
                avg_delay = np.mean(pkts[:, 1])
                # IAT logic
                iats = np.diff(pkts[:, 2]) if len(pkts) > 1 else [0]
                iat_mean = np.mean(iats)
                iat_std = np.std(iats) if len(iats) > 1 else 0
                is_silent = 0
            else:
                total_size = 0
                avg_delay = self.prev_delay
                iat_mean = 0
                iat_std = 0
                is_silent = 1

            throughput_mbps = (total_size * 8) / (self.window_ms / 1000.0) / 1e6
            
            # --- FIXED GRADIENT LOGIC ---
            # Compare current value to the value from the previous 50ms window
            if len(self.history) > 0:
                # history[-1] contains the full feature list of the previous window
                prev_feat = self.history[-1]
                delay_grad = avg_delay - prev_feat[1] # index 1 is delay
                bif_grad = total_size - prev_feat[2]  # index 2 is bif
            else:
                delay_grad = 0
                bif_grad = 0

            # 9 Features: [size, delay, bif, iat_mean, iat_std, throughput, delay_grad, bif_grad, is_silent]
            features = [
                float(total_size), float(avg_delay), float(total_size), 
                float(iat_mean), float(iat_std), float(throughput_mbps), 
                float(delay_grad), float(bif_grad), float(is_silent)
            ]
            
            # Reset for next window
            self.current_window_pkts = []
            self.last_window_ts = ts
            return features
            
        return None

    def log_step(self, features, prediction):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.log_data.append([timestamp] + features + [prediction])

    def save_to_csv(self, filename="inference_data/inference_results.csv"):
        if not self.log_data: return
            
        cols = [
            'timestamp', 'size', 'delay', 'bif', 'iat_mean', 'iat_std', 
            'throughput_mbps', 'delay_grad', 'bif_grad', 'is_silent', 'pred_thru'
        ]
        
        df = pd.DataFrame(self.log_data, columns=cols)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # --- FIXED HEADER LOGIC ---
        # Only skip header if file exists AND is not empty
        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        
        self.log_data = [] 
        print(f"\n[Disk] Data synchronized to {filename}")
