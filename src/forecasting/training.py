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
    # order=None -> ARIMAForecaster/SARIMAForecaster determine (p,d,q) and the
    # seasonal component per-series from ADF/ACF instead of a single fixed order.
    "arima": lambda: ARIMAForecaster(),
    "sarima": lambda: SARIMAForecaster(),
    "xgboost": lambda: XGBoostForecaster(params={"objective": "reg:squarederror", "verbosity": 0}, num_round=100),
    "gru": lambda: GRUForecaster(seq_len=14, hidden_size=32, lr=0.005, epochs=5, batch_size=16),
}
REGISTERED_MODEL_NAME = "cpu_forecast"

FEATURE_COLUMNS = [
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
    "day_of_week",
    "month",
    "is_weekend",
]


def time_split(df, date_col="timestamp", test_days=30):
    """Contiguous train/test split - test is the last test_days, train is
    everything before it. No gap in between: ARIMA/SARIMA forecast forward
    from where training ends, so train must end exactly where test begins,
    or the forecast gets evaluated against the wrong dates."""
    last_date = df[date_col].max()
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    train = df[df[date_col] < test_start]
    test = df[df[date_col] >= test_start]
    return train, test


def _prepare_ml_data(features):
    # cpu_utilization itself (and anything derived from it, like the old
    # cpu_ram_ratio) is NOT a feature - it's the label. Only lag/rolling values
    # and calendar features are used, matching exactly what's available at live
    # forecast time (see forecast_service.FEATURE_COLUMNS / recursive_xgboost_forecast).
    return features[FEATURE_COLUMNS].fillna(0), features["cpu_utilization"]


def _load_split_data(dataset_path: str | Path | None = None):
    if dataset_path:
        data = pd.read_csv(dataset_path)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
    else:
        data = generate_synthetic_data(n_servers=20, n_days=150)
    data = validate_data(data)
    features = build_features(data)
    # Uploaded datasets are often much shorter than the 150-day synthetic default;
    # scale the test window down instead of hardcoding 30, which would empty out
    # small datasets entirely.
    span_days = max(1, (data["timestamp"].max() - data["timestamp"].min()).days)
    window = min(30, max(3, span_days // 5))
    return time_split(features, test_days=window)


def train_one_model(name: str, dataset_path: str | Path | None = None, dataset_id: str | None = None) -> dict:
    """Train + evaluate + log a single named model to MLflow. Independent of the
    other models - this is what lets Airflow run all 4 as parallel tasks instead
    of the sequential loop in train_and_evaluate().

    dataset_path/dataset_id: train against an uploaded/streamed dataset (see
    forecast_service.UPLOADED_DATASETS_DIR) instead of the built-in synthetic data.
    Runs are tagged with dataset_id so they're identifiable in the MLflow UI; a
    model trained this way is registered as a new version but NOT auto-promoted
    to the 'champion' alias (see promote_best_xgboost_version) since an ad-hoc
    upload shouldn't silently replace the model serving live synthetic-data traffic.
    """
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_FACTORIES)}")
    train, test = _load_split_data(dataset_path)
    X_train, y_train = _prepare_ml_data(train)
    X_test, y_test = _prepare_ml_data(test)
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Not enough data to train '{name}' on this dataset (train rows={len(X_train)}, test rows={len(X_test)})."
        )
    model = MODEL_FACTORIES[name]()

    mlflow.set_experiment("cpu_forecasting")
    run_name = f"{name}_{dataset_id}" if dataset_id else name
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("dataset_id", dataset_id or "synthetic")
        if name in {"arima", "sarima"}:
            model.fit(train[["server_id", "timestamp", "cpu_utilization"]])
            y_pred = model.predict(test[["server_id", "timestamp"]])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        metrics = evaluate_forecast(y_test, y_pred)
        mlflow.log_params({"model": name, "horizon_days": 1, "dataset_id": dataset_id or "synthetic"})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(save_predictions_csv(test, y_pred, name, dataset_id=dataset_id or "synthetic"))
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
    no alias is set, so this is what makes that fallback unnecessary going forward.

    Only considers versions trained on the synthetic baseline (run tag dataset_id is
    missing or "synthetic") AND whose logged feature schema matches the current
    FEATURE_COLUMNS exactly. Two separate failure modes otherwise let an incompatible
    model silently become champion and break live serving with a feature-names
    mismatch: (1) ad-hoc dataset uploads can score a numerically lower MAE than the
    baseline (a tiny/easy uploaded dataset is often trivially easy to fit), and (2) a
    stale run from before a feature-set change (e.g. the old leaked cpu/ram/disk/
    network + ratio features) can *also* score an artificially low MAE - leaked
    features trivially "solve" the regression, so MAE alone can't tell a stale,
    incompatible model apart from a genuinely good one."""
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
            run_dataset_id = run.data.tags.get("dataset_id")
        except Exception:
            continue
        if run_dataset_id not in (None, "synthetic") or mae is None:
            continue
        try:
            model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL_NAME}/{v.version}")
            feature_names_in = getattr(model, "feature_names_in_", None)
            feature_names = list(feature_names_in) if feature_names_in is not None else []
        except Exception:
            continue
        if feature_names != FEATURE_COLUMNS:
            continue
        if mae < best_mae:
            best_mae, best_version = mae, v.version

    if best_version is not None:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", best_version)
        return best_version
    return None


def train_and_evaluate(dataset_path: str | Path | None = None, dataset_id: str | None = None, promote: bool = True):
    results = {}
    errors = {}
    best_model_name = None
    best_mae = float("inf")
    for name in MODEL_FACTORIES:
        try:
            metrics = train_one_model(name, dataset_path=dataset_path, dataset_id=dataset_id)
        except Exception as exc:
            errors[name] = str(exc)
            continue
        results[name] = {k: v for k, v in metrics.items() if k not in {"model", "run_id", "model_version"}}
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model_name = name

    comparison = compare_models(results) if results else None
    if comparison is not None:
        summary_name = "model_comparison.csv" if not dataset_id else f"model_comparison_{dataset_id}.csv"
        comparison.to_csv(Path("mlruns") / summary_name, index=False)
    if promote and not dataset_id:
        promote_best_xgboost_version()
    return comparison, best_model_name, errors


def train_on_dataset(dataset_id: str, dataset_path: str | Path) -> dict:
    """Entry point for the /datasets/{id}/train API route and the Airflow DAG's
    dataset-scoped runs: trains all 4 models against one specific dataset and
    returns a JSON-friendly summary (no DataFrame objects)."""
    comparison, best_model_name, errors = train_and_evaluate(dataset_path=dataset_path, dataset_id=dataset_id, promote=False)
    return {
        "dataset_id": dataset_id,
        "best_model": best_model_name,
        "results": comparison.to_dict(orient="records") if comparison is not None else [],
        "errors": errors,
    }


def save_predictions_csv(test_df: pd.DataFrame, y_pred, prefix, dataset_id: str = "synthetic"):
    # dataset_id in the filename, not a shared "predictions.csv" - otherwise any
    # ad-hoc dataset's training run silently clobbers the synthetic baseline's
    # batch predictions (and vice versa), which is what fleet_overview() reads from.
    output_dir = Path("artifacts") / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test_df[["server_id", "timestamp"]].copy()
    pred_df["predicted_cpu_utilization"] = y_pred
    output_path = output_dir / f"predictions_{dataset_id}.csv"
    pred_df.to_csv(output_path, index=False)
    return str(output_path)


def run_training():
    comparison, best, errors = train_and_evaluate()
    print("Model comparison:\n", comparison)
    print(f"Best model: {best}")
    if errors:
        print("Errors:", errors)


if __name__ == "__main__":
    run_training()
