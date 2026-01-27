import subprocess
import sys
from model_handler import ModelHandler
from network_processor import NetworkProcessor
from database_handler import DataVault

CONFIG = {
    "interface": "any", # capture on all interfaces
    "duration": 600, # 10min capture
    "model_path": "model_files/tcp_forecast_model.pth",
    "scaler_x": "model_files/scaler_x.pkl",
    "scaler_y": "model_files/scaler_y.pkl"
}

def run_engine():
    handler = ModelHandler(CONFIG["model_path"], CONFIG["scaler_x"], CONFIG["scaler_y"])
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
                processor.history.append(feat)
                pred = 0
                if len(processor.history) == processor.lookback:
                    pred = handler.predict(processor.history)
                
                # Push to DB instead of blocking with CSV/Plotting
                vault.push_metrics(feat[3], pred, feat[1], feat[2])

                # CSV Buffer (Processor already has this list)
                processor.log_step(feat, pred)
                
                # Periodic Disk Save
                if len(processor.log_data) >= 100: 
                    processor.save_to_csv() # This now appends and clears
                
                sys.stdout.write(f"\r[LIVE] Actual: {feat[3]:.1f} | Pred: {pred:.1f} Mbps")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping Engine...")
    finally:
        proc.terminate()

if __name__ == "__main__":
    run_engine()
