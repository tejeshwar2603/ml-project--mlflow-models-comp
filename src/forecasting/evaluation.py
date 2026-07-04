import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.nan


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if np.any(mask):
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100
    return np.nan


def explained_variance_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def evaluate_forecast(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
    }


def compare_models(results: dict):
    return pd.DataFrame.from_records(
        [dict(model=k, **v) for k, v in results.items()]
    ).sort_values("mae")
