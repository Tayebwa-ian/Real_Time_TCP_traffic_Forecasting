# Real-Time TCP Traffic Forecasting with LSTM

A modular, high-performance pipeline designed to capture live network traffic, perform time-series forecasting using an LSTM neural network, and visualize the "Phase Shift" between actual and predicted throughput in real-time.



## System Architecture
The system is decoupled into three distinct layers to ensure maximum performance and stability:

1. **The Engine (Python & Tshark):** Captures raw packets via Tshark using a non-loopback filter, extracts features (BIF, RTT, Gradients), and runs them through a PyTorch LSTM model.
2. **The Time-Series Vault (InfluxDB v2):** A high-speed database that stores actual vs. predicted metrics with nanosecond precision.
3. **The Dashboard (Grafana):** A web-based visualizer that queries InfluxDB every 100ms to plot real-time traffic flows.

## Key Features
* **Headless Operation:** Runs without a GUI to prevent Matplotlib/Qt overhead and environment-related crashes.
* **Dual-Path Storage:** Simultaneously streams data to InfluxDB for visualization and appends to a local CSV for offline model training and audit trails.
* **Intelligent Filtering:** Automatically ignores loopback (`127.0.0.1`) traffic using BPF filters to focus exclusively on external network health.
* **Advanced Feature Set:**
    * Throughput (Mbps)
    * Round Trip Time (RTT) Delay
    * Bytes in Flight (BIF)
    * Delay & BIF Gradients (Velocity indicators for forecasting)

---

## Installation & Setup

### 1. Prerequisites
* Docker & Docker Compose
* Python 3.10+
* `tshark` (Wireshark CLI)

### 2. Infrastructure Setup
Spin up the database and visualization layers:
```bash
docker-compose up -d
``` 
### 3. Python Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```  

## Usage
To start the real-time capture and forecasting engine on all interfaces:  

```bash
cd src
sudo python3 main.py
```  

## Visualizing the "Phase Shift"
To monitor the model performance, create a "Time Series" panel in Grafana and use the following Flux Query:
```
from(bucket: "network_stats")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "traffic_prediction")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
```  

## Project Structure
```
.
├── src/
│   ├── main.py                 # Orchestrator & Live Loop
│   ├── model_handler.py        # PyTorch LSTM Inference & Scaling logic
│   ├── network_processor.py    # Tshark parsing & Feature Engineering
│   └── database_handler.py     # InfluxDB & CSV logging
│   └── feature_extraction.py   # Extract features from raw pantheon logs
│   └── training.ipynb          # Training and Evaluating the model
├── model_files/
│   ├── tcp_forecast_model.pth
│   ├── scaler_x.pkl            # RobustScaler for features
│   └── scaler_y.pkl            # RobustScaler for target
├── train_data/
│   └── training_data.CSV       # stores extracted data from raw logs
├── inference_data/
│   └── inference_results.csv   # Stores data after model inference in a live network
├── docker-compose.yml          # Grafana & InfluxDB stack
├── .gitignore                  # specifies which file are not uploaded to remote repo
└── requirements.txt
```
