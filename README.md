# Real-Time TCP Traffic Forecasting with LSTM

A modular, high-performance pipeline designed to capture live network traffic, perform time-series forecasting using an LSTM neural network, and visualize network dynamics in real-time. This system is specifically tuned to detect congestion patterns and "micro-bursts" by analyzing TCP stream momentum.


## System Architecture
TThe system is decoupled into three distinct layers to ensure maximum performance and stability:

1. **The Engine (Python & Tshark):** Captures raw packets via Tshark, aggregates them into 50ms windows, and performs real-time feature engineering.
2. **The Time-Series Vault (InfluxDB v2):** A high-speed database that stores actual vs. predicted metrics with nanosecond precision.
3. **The Dashboard (Grafana):** A web-based visualizer that queries InfluxDB every 100ms to plot real-time traffic flows and "Phase Shift" analysis.

## Training Context: The Pantheon Dataset  
The model was trained using data from [Pantheon](https://pantheon.stanford.edu/), a community-shared platform for academic research on network protocols.

- **Diverse Network Conditions:** Training included traces from various real-world network paths (e.g., ethernet, cellular, and cross-continent links).
- **Protocol Variety:** The dataset captures the behavior of multiple congestion control algorithms, allowing the LSTM to learn generalized TCP "sawtooth" patterns.
- **High-Speed Baselines:** Training was conducted on high-throughput environments (often 100Mbps+), providing a robust baseline for identifying congestion-induced throughput drops.

## Advanced Feature Set:  
The model utilizes a 9-dimensional feature vector for every prediction step to capture the full state of the network:  

| Feature | Description | Importance |
|-------|------------|------------|
| Size | Total bytes transferred in the 50ms window. | Raw volume indicator. |
| Delay (RTT) | Mean Round Trip Time (from tcp.analysis.ack_rtt). | Primary congestion signal. |
| BIF | Total Bytes in Flight within the current window. | Measures network "fullness". |
| IAT Mean | Mean Inter-Arrival Time of packets. | Captures packet pacing consistency. |
| IAT Std | Standard Deviation of Inter-Arrival Times. | Detects jitter and network instability. |
| Throughput | Calculated Mbps for the current window. | Target variable for forecasting. |
| Delay Gradient | The velocity of RTT change (Current - Previous). | Predicts queue buildup before it happens. |
| BIF Gradient | The velocity of BIF change. | Detects sudden protocol throttling. |
| Is Silent | Boolean flag for windows with zero traffic. | Prevents model hallucination during idle periods. |

## Key Model Parameters
- **Temporal Context (750ms Lookback):** The LSTM utilizes a rolling 15-step sequence (15 x 50ms windows) to maintain memory of recent network states.
- **Model Accuracy:** On the Pantheon validation set, the model achieves an $R^2 \approx 0.91$ and an AUC of 0.93 for congestion event classification.

## Evaluation Metrics
The following results were achieved on the Pantheon test set, validating the model's ability to navigate complex TCP dynamics:

1. **R-Squared ($R^2$) - 0.9192:** The model explains approximately 92% of the variance in network throughput, showing a near-perfect fit for the underlying TCP congestion control logic.
2. **Congestion Warning AUC - 0.9596:** With a "Crash Threshold" set at 20 Mbps, the model achieves a 96% success rate in distinguishing between normal traffic flow and imminent congestion events.
3. **Average Prediction Error(MAE) - 69.17 Mbps:** Given the high-speed nature of the training data, this error margin reflects the model's performance during high-velocity "micro-bursts" common in modern TCP streams.

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
python3 main.py
```  

## Visualizing Performance
Access Grafana at http://localhost:3000. The "Phase Shift" is best viewed by plotting actual_throughput and predicted_throughput on the same axis. A high-performing model will show the predicted line slightly leading or tightly following the actual peaks during congestion events.

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
