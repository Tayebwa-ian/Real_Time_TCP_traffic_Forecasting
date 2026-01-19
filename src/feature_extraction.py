import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
DATA_FOLDER = 'data/'
OUTPUT_FILE = 'src/train_data/training_data.csv'
WINDOW_SIZE = '50ms'

def process_logs():
    is_first_write = True
    
    for filename in os.listdir(DATA_FOLDER):
        if 'datalink' not in filename: continue
        
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"Processing {filename}...")
        
        egress_data = []
        bytes_in_flight = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                # Skip headers (lines starting with #) and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                ts = float(parts[0])
                event = parts[1]
                size = int(parts[2])
                
                if event == '+':
                    bytes_in_flight += size
                elif event == '-':
                    bytes_in_flight -= size
                    # In egress lines, delay is index 3
                    delay = float(parts[3])
                    egress_data.append([ts, size, delay, bytes_in_flight])

        # Convert the collected egress events into a DataFrame
        df = pd.DataFrame(egress_data, columns=['ts', 'size', 'delay', 'bif'])
        df['ts_dt'] = pd.to_timedelta(df['ts'], unit='ms')
        
        # Resample into 50ms windows
        resampled = df.set_index('ts_dt').resample(WINDOW_SIZE).agg({
            'size': 'sum',    # Used for throughput
            'delay': 'mean',  # Average Latency
            'bif': 'last'     # Current network pressure
        }).fillna(0)
        
        # Final Feature Engineering
        resampled['throughput_mbps'] = (resampled['size'] * 8) / 1e6 / 0.05
        resampled['target_next_throughput'] = resampled['throughput_mbps'].shift(-1)
        resampled = resampled.dropna()
        
        # Append to Master CSV
        if is_first_write:
            resampled.to_csv(OUTPUT_FILE, mode='w', index=False)
            is_first_write = False
        else:
            resampled.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

if __name__ == "__main__":
    process_logs()
    print(f"Success! Final dataset saved to {OUTPUT_FILE}")
