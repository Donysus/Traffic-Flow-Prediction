# AI‑Based Traffic Flow Prediction

This project develops a High tier traffic flow prediction model using an advanced bidirectional LSTM network. It generates synthetic traffic data that incorporates daily cycles, weekly patterns, rush-hour peaks, and random incidents, and then forecasts traffic over a one‑hour horizon.

## Features

- **Realistic Data Generation:** Simulates traffic with daily and weekly cycles plus random events.
- **Advanced LSTM Model:** Uses bidirectional LSTM layers with dropout for robust prediction.
- **Evaluation Metrics:** Assesses model performance with RMSE and MAE.
- **Deployment Ready:** Saves model artifacts and scaler for future use.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python traffic_flow_prediction.py
```

## Files

1. traffic_flow_prediction.py - Main project code.
2. requirements.txt - List of dependencies.
3. .gitignore - Git ignore file.

## Author

Raghav Jha

raghavmrparadise@gmail.com



