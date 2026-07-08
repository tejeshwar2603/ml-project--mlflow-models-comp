import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from pathlib import Path
from datetime import timedelta
from .data import generate_synthetic_data, validate_data
from .features import build_features
from .models import ARIMAForecaster, SARIMAForecaster, XGBoostForecaster, GRUForecaster, LSTMForecaster, TFTForecaster
from .evaluation import evaluate_forecast, compare_models

MODEL_FACTORIES = {
    "arima": lambda: ARIMAForecaster(order=(3, 1, 1)),
    "sarima": lambda: SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)),
    "xgboost": lambda: XGBoostForecaster(params={"objective": "reg:squarederror", "verbosity": 0}, num_round=100),
    "gru": lambda: GRUForecaster(seq_len=14, hidden_size=32, lr=0.005, epochs=5, batch_size=16),
}
REGISTERED_MODEL_NAME = "cpu_forecast"


def time_split(df, date_col="timestamp", test_days=30, val_days=30):
    last_date = df[date_col].max()
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df[date_col] < val_start]
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)]
    test = df[df[date_col] >= test_start]
    return train, val, test


def _prepare_ml_data(features):
    cols = [
        "cpu_utilization",
        "ram_utilization",
        "disk_utilization",
        "network_utilization",
        "cpu_utilization_lag_1",
        "cpu_utilization_lag_3",
        "cpu_utilization_lag_7",
        "cpu_utilization_lag_14",
        "cpu_utilization_lag_30",
        "cpu_utilization_roll_mean_3",
        "cpu_utilization_roll_std_3",
        "cpu_utilization_roll_min_3",
        "cpu_utilization_roll_max_3",
        "cpu_utilization_roll_mean_7",
        "cpu_utilization_roll_std_7",
        "cpu_utilization_roll_min_7",
        "cpu_utilization_roll_max_7",
        "cpu_ram_ratio",
        "cpu_disk_ratio",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    return features[cols].fillna(0), features["cpu_utilization"]


def _load_split_data():
    data = generate_synthetic_data(n_servers=20, n_days=150)
    data = validate_data(data)
    features = build_features(data)
    return time_split(features, test_days=30, val_days=30)


def train_one_model(name: str) -> dict:
    """Train + evaluate + log a single named model to MLflow. Independent of the
    other models - this is what lets Airflow run all 4 as parallel tasks instead
    of the sequential loop in train_and_evaluate()."""
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_FACTORIES)}")
    train, val, test = _load_split_data()
    X_train, y_train = _prepare_ml_data(train)
    X_test, y_test = _prepare_ml_data(test)
    model = MODEL_FACTORIES[name]()

    mlflow.set_experiment("cpu_forecasting")
    with mlflow.start_run(run_name=name) as run:
        if name in {"arima", "sarima"}:
            model.fit(train[["server_id", "timestamp", "cpu_utilization"]])
            y_pred = model.predict(test[["server_id", "timestamp"]])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        metrics = evaluate_forecast(y_test, y_pred)
        mlflow.log_params({"model": name, "horizon_days": 1})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(save_predictions_csv(test, y_pred, name))
        model_version = None
        if name == "xgboost" and hasattr(model, "model"):
            info = mlflow.xgboost.log_model(
                model.model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            model_version = getattr(info, "registered_model_version", None)
        return {"model": name, "run_id": run.info.run_id, "model_version": model_version, **metrics}


def promote_best_xgboost_version(candidate_run_ids: list[str] | None = None) -> str | None:
    """Compare registered cpu_forecast versions by their run's test MAE and point the
    'champion' alias at the best one. Aliases are MLflow 3.x's replacement for the
    deprecated Production/Staging stages - api.py falls back to 'latest version' if
    no alias is set, so this is what makes that fallback unnecessary going forward."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except Exception:
        return None
    if not versions:
        return None

    best_version, best_mae = None, float("inf")
    for v in versions:
        try:
            run = client.get_run(v.run_id)
            mae = run.data.metrics.get("mae")
        except Exception:
            mae = None
        if mae is not None and mae < best_mae:
            best_mae, best_version = mae, v.version

    if best_version is not None:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", best_version)
        return best_version
    return None


def train_and_evaluate():
    results = {}
    best_model_name = None
    best_mae = float("inf")
    for name in MODEL_FACTORIES:
        metrics = train_one_model(name)
        results[name] = {k: v for k, v in metrics.items() if k not in {"model", "run_id", "model_version"}}
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model_name = name

    comparison = compare_models(results)
    summary_path = Path("mlruns") / "model_comparison.csv"
    comparison.to_csv(summary_path, index=False)
    promote_best_xgboost_version()
    return comparison, best_model_name


def save_predictions_csv(test_df: pd.DataFrame, y_pred, prefix):
    output_dir = Path("artifacts") / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test_df[["server_id", "timestamp"]].copy()
    pred_df["predicted_cpu_utilization"] = y_pred
    output_path = output_dir / "predictions.csv"
    pred_df.to_csv(output_path, index=False)
    return str(output_path)


def run_training():
    comparison, best = train_and_evaluate()
    print("Model comparison:\n", comparison)
    print(f"Best model: {best}")


if __name__ == "__main__":
    run_training()
