import numpy as np
import pandas as pd
from collections import deque
import os
from datetime import datetime

class NetworkProcessor:
    def __init__(self, interface, duration, window_ms=50, lookback=10):
        self.interface = interface
        self.duration = duration
        self.window_ms = window_ms
        self.lookback = lookback
        self.history = deque(maxlen=lookback)
        self.log_data = [] # For CSV export
        
        # State tracking
        self.last_window_ts = 0
        self.prev_delay = 0
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
        
        ts = float(parts[0])
        size = int(parts[1])
        delay = float(parts[2]) if (len(parts) > 2 and parts[2]) else self.prev_delay
        self.prev_delay = delay
        
        if self.last_window_ts == 0: self.last_window_ts = ts
        self.current_window_pkts.append([size, delay])
        
        # Check window completion
        if (ts - self.last_window_ts) >= (self.window_ms / 1000.0):
            throughput_mbps = (sum(p[0] for p in self.current_window_pkts) * 8) / (self.window_ms / 1000.0) / 1e6
            delay = np.mean([p[1] for p in self.current_window_pkts]) if self.current_window_pkts else 0
            bif = sum(p[0] for p in self.current_window_pkts)
            
            delay_grad = delay - self.history[-1][1] if self.history else 0
            bif_grad = bif - self.history[-1][2] if self.history else 0
            
            features = [sum(p[0] for p in self.current_window_pkts), delay, bif, throughput_mbps, delay_grad, bif_grad]
            
            # Reset window
            self.current_window_pkts = []
            self.last_window_ts = ts
            return features
        return None

    def log_step(self, features, prediction):
        # Grab human-readable time: 2026-01-27 18:30:05
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Store for CSV: [timestamp, size, delay, bif, throughput_mbps, delay_grad, bif_grad, pred_thru]
        self.log_data.append([timestamp] + features + [prediction])

    def save_to_csv(self, filename="inference_data/inference_results.csv"):
        if not self.log_data:
            return
            
        # Added 'timestamp' to the start of the list
        cols = ['timestamp', 'size', 'delay', 'bif', 'throughput_mbps', 'delay_grad', 'bif_grad', 'pred_thru']
        
        df = pd.DataFrame(self.log_data, columns=cols)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        
        self.log_data = [] 
        print(f"\n[Disk] Appended {len(df)} rows to {filename}")
