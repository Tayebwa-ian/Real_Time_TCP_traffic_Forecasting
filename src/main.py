import subprocess
import sys
from model_handler import ModelHandler
from network_processor import NetworkProcessor
from database_handler import DataVault

CONFIG = {
    "interface": "any", 
    "duration": 600, # 10min capture
    "model_path": "model_files/tcp_forecast_model.pth",
    "scaler_path": "model_files/scaler_x.pkl",
}

def run_engine():
    # Removed scaler_y as the model predicts raw Mbps now
    handler = ModelHandler(CONFIG["model_path"], CONFIG["scaler_path"])
    
    # Processor now defaults to lookback=15 (750ms)
    processor = NetworkProcessor(CONFIG["interface"], CONFIG["duration"])
    vault = DataVault()

    print(f"--- ENGINE START: {CONFIG['interface']} -> InfluxDB ---")
    proc = subprocess.Popen(processor.get_tshark_command(), stdout=subprocess.PIPE, text=True)

    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None: break
            
            feat = processor.extract_features(line)
            if feat:
                # feat order: [size, delay, bif, iat_mean, iat_std, throughput, d_grad, b_grad, silent]
                processor.history.append(feat)
                pred = 0
                
                # Check if we have the full 750ms (15 windows) of history
                if len(processor.history) == processor.lookback:
                    pred = handler.predict(list(processor.history))
                
                # Updated to pass the full feature list to the vault
                # The vault now handles indexing the features internally
                vault.push_metrics(feat, pred)

                # Log to local memory buffer
                processor.log_step(feat, pred)
                
                # Periodic Disk Save
                if len(processor.log_data) >= 100: 
                    processor.save_to_csv()
                
                # Update console (throughput is at index 5 in our new 9-feature list)
                sys.stdout.write(f"\r[LIVE] Actual: {feat[5]:.1f} | Pred: {pred:.1f} Mbps")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping Engine...")
    finally:
        processor.save_to_csv() # Final save before exit
        proc.terminate()

if __name__ == "__main__":
    run_engine()
