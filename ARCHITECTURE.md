# Enterprise AIOps CPU Forecasting Project Architecture

## Overview

This project is a production-oriented CPU utilization forecasting pipeline designed for thousands of servers. It combines data ingestion, time-series and feature-based forecasting, MLflow experiment tracking, and model deployment via FastAPI.

## Project Layers

### 1. Data Layer

- `src/forecasting/data.py`
  - `generate_synthetic_data`: synthetic sample data for CPU, RAM, disk, network, and server metadata.
  - `validate_data`: ensures required columns exist, timestamps are parsed, duplicates are removed, and chronological order is maintained.

### 2. Feature Engineering

- `src/forecasting/features.py`
  - Missing value handling with interpolation, forward/backward fill while preserving true 0.0 values.
  - Lag features: 1, 3, 7, 14, 30 days.
  - Rolling statistics: mean, std, min, max for rolling windows of 3, 7, 14, 30 days.
  - Calendar features: day of week, month, weekend flag.
  - Ratio features: `cpu_ram_ratio`, `cpu_disk_ratio`.

### 3. Modeling

- `src/forecasting/models.py`
  - `ARIMAForecaster` and `SARIMAForecaster`: baseline classical models for per-server time-series forecasting.
  - `XGBoostForecaster`: primary production model using gradient-boosted regression.
  - `TorchForecaster` base class and derived `GRUForecaster`, `LSTMForecaster`, `TFTForecaster` for sequence-based deep learning.

### 4. Evaluation

- `src/forecasting/evaluation.py`
  - Metrics supported: MAE, MSE, RMSE, MAPE, SMAPE, R², explained variance.
  - `compare_models`: create comparison table sorted by MAE.

### 5. MLflow Tracking and Registry

- `src/forecasting/mlflow_utils.py`
  - Wraps MLflow experiment run creation, metrics, parameters, artifacts, and model logging.
- `src/forecasting/training.py`
  - Logs experiments to MLflow during training.
  - Registers the best XGBoost model to the MLflow model registry under `cpu_forecast`.

### 6. Prediction Service

- `src/forecasting/api.py`
  - FastAPI JSON endpoint `/predict` for runtime inference.
  - Loads the registered production model from `FORECAST_MODEL_URI` or `models:/cpu_forecast/Production`.
- `src/forecasting/app.py`
  - Uvicorn server launcher for the REST API.

## Workflow

1. Data ingestion and validation
   - Load CSV using `src.forecasting.data.load_data`, or generate sample data with `generate_synthetic_data`.
   - Ensure chronological order per `server_id` and remove duplicates.

2. Preprocessing and feature engineering
   - Call `build_features` to create lag, rolling, calendar, and ratio features.
   - Fill missing values with interpolation and forward/backward fill.

3. Train/validation/test split
   - `time_split` splits chronological data into training, validation, and test periods.
   - This supports time-based evaluation and avoids lookahead bias.

4. Model training and comparison
   - Train baseline and production models in `training.py`.
   - Evaluate each model on test data using multiple metrics.
   - Track parameters, metrics, and prediction artifacts in MLflow.

5. Model registration
   - The XGBoost model is logged and registered to MLflow model registry.
   - The best-performing model can be promoted to a production stage.

6. Deploy prediction service
   - FastAPI service loads the production model from MLflow and exposes `/predict`.
   - Clients submit feature payloads and receive CPU utilization forecasts.

## Run Instructions

### 1. Install dependencies

```bash
cd "d:\ml project -mlflow-models comp"
python -m pip install -r requirements.txt
```

### 2. Run training

```bash
cd "d:\ml project -mlflow-models comp"
python -m src.forecasting.training
```

This performs:

- Synthetic data generation
- Feature engineering
- Model training for ARIMA, SARIMA, XGBoost, and GRU
- MLflow experiment logging
- MLflow model registry registration for the best XGBoost model

### 3. Start the API

```bash
cd "d:\ml project -mlflow-models comp"
python -m src.forecasting.app
```

The API runs on `http://0.0.0.0:8000` by default.

### 4. Make a prediction

Send a POST request to `/predict` with JSON payload:

```json
{
  "server_id": "server-001",
  "horizon": 1,
  "features": {
    "cpu_utilization": 35.2,
    "ram_utilization": 42.1,
    "disk_utilization": 55.0,
    "network_utilization": 23.4,
    "cpu_utilization_lag_1": 34.0,
    "cpu_utilization_lag_3": 30.2,
    "cpu_utilization_lag_7": 25.8,
    "cpu_utilization_lag_14": 22.1,
    "cpu_utilization_lag_30": 20.0,
    "cpu_utilization_roll_mean_3": 33.2,
    "cpu_utilization_roll_std_3": 2.1,
    "cpu_utilization_roll_min_3": 31.0,
    "cpu_utilization_roll_max_3": 35.8,
    "cpu_utilization_roll_mean_7": 30.1,
    "cpu_utilization_roll_std_7": 3.4,
    "cpu_utilization_roll_min_7": 28.0,
    "cpu_utilization_roll_max_7": 33.5,
    "cpu_ram_ratio": 0.83,
    "cpu_disk_ratio": 0.64,
    "day_of_week": 2,
    "month": 7,
    "is_weekend": 0
  }
}
```

### 5. Optional: override model URI

Set `FORECAST_MODEL_URI` to a different registered model before starting the API:

```bash
set FORECAST_MODEL_URI=models:/cpu_forecast/Production
python -m src.forecasting.app
```

## Notes

- The current pipeline is designed for daily server CPU forecasting.
- You can extend it by adding LightGBM, TCN, hyperparameter tuning, SHAP explainability, and confidence interval generation.
- `mlruns/` stores MLflow experiments locally by default.
