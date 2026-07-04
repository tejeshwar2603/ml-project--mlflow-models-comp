# Enterprise AIOps Copilot Forecasting

Production-ready CPU forecasting pipeline with classical, machine learning, and deep learning models.

## What is included

- Data ingestion and validation
- Feature engineering with lag, rolling, and calendar features
- Time-based train/validation/test split and rolling window evaluation
- Model wrappers for ARIMA, SARIMA, XGBoost, GRU, LSTM, TFT
- MLflow experiment tracking with metrics, parameters, artifacts, and model registry
- FastAPI prediction service using the registered production model

## Install

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m src.forecasting.training
```

## Run API

```bash
python -m src.forecasting.app
```

## Notes

- Training uses synthetic sample data by default.
- MLflow artifacts are stored in `mlruns/` under the project directory.
- The REST API exposes a `/predict` endpoint for daily horizon predictions.
- See `ARCHITECTURE.md` for full workflow, architecture details, and step-by-step run instructions.
