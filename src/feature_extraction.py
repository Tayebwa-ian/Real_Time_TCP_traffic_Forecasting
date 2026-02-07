import pandas as pd
import os
import numpy as np
import random

# --- CONFIGURATION ---
DATA_FOLDER = 'data/'
TRAIN_OUTPUT = 'src/train_data/train_data.csv'
TEST_OUTPUT = 'src/train_data/test_data.csv'
WINDOW_SIZE_MS = 50
WINDOW_SIZE = f'{WINDOW_SIZE_MS}ms'
TRAIN_SPLIT = 0.8  # 80% of files for training

def process_logs():
    # 1. Get and Shuffle Files
    # We shuffle BEFORE processing to ensure protocol diversity in train/test
    all_files = [f for f in os.listdir(DATA_FOLDER) if 'datalink' in f]
    random.seed(42) # For reproducibility
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * TRAIN_SPLIT)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    def process_file_list(file_list, output_path):
        is_first_write = True
        print(f"Starting processing for {output_path}...")
        
        for session_idx, filename in enumerate(file_list):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"  [{session_idx+1}/{len(file_list)}] Processing {filename}")
            
            egress_data = []
            bytes_in_flight = 0
            last_ts = 0
            
            # --- PACKET LEVEL EXTRACTION ---
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.split()
                    try:
                        ts = float(parts[0])
                        event = parts[1]
                        size = int(parts[2])
                        
                        iat = ts - last_ts
                        last_ts = ts
                        
                        if event == '+':
                            bytes_in_flight += size
                        elif event == '-':
                            bytes_in_flight -= size
                            delay = float(parts[3])
                            egress_data.append([ts, size, delay, bytes_in_flight, iat])
                    except (IndexError, ValueError):
                        continue

            if not egress_data:
                continue

            # --- WINDOWING & TEMPORAL ORDER ---
            df = pd.DataFrame(egress_data, columns=['ts', 'size', 'delay', 'bif', 'iat'])
            df = df.sort_values('ts') 
            df['ts_dt'] = pd.to_timedelta(df['ts'], unit='ms')
            
            resampled = df.set_index('ts_dt').resample(WINDOW_SIZE).agg({
                'size': 'sum',
                'delay': 'mean',
                'bif': 'last',
                'iat': ['mean', 'std']
            })
            
            resampled.columns = ['size', 'delay', 'bif', 'iat_mean', 'iat_std']
            resampled = resampled.fillna(0)

            # --- FEATURE ENGINEERING ---
            # Throughput calculation
            resampled['throughput_mbps'] = (resampled['size'] * 8) / 1e6 / (WINDOW_SIZE_MS / 1000)
            
            # Gradients (Crucial for Protocol-Agnostic Learning)
            resampled['delay_gradient'] = resampled['delay'].diff().fillna(0)
            resampled['bif_gradient'] = resampled['bif'].diff().fillna(0)
            
            # Silence detection
            resampled['is_silent'] = (resampled['size'] == 0).astype(int)
            
            # Unique ID for this specific file/session
            resampled['session_id'] = session_idx 

            # Target: The throughput of the NEXT window
            resampled['target_next_throughput'] = resampled['throughput_mbps'].shift(-1)
            
            # Remove the last row as it won't have a 'next' throughput
            resampled = resampled.dropna()

            # --- SAVING ---
            if is_first_write:
                resampled.to_csv(output_path, mode='w', index=False)
                is_first_write = False
            else:
                resampled.to_csv(output_path, mode='a', index=False, header=False)

    # Execute for both splits
    process_file_list(train_files, TRAIN_OUTPUT)
    process_file_list(test_files, TEST_OUTPUT)

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)
    process_logs()
    print(f"\nSuccess! Training data: {TRAIN_OUTPUT}")
    print(f"Success! Testing data: {TEST_OUTPUT}")
