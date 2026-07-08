"""Live, on-demand forecasting + rightsizing recommendations.

Unlike ``artifacts/*/predictions.csv`` (fixed batch output from the last
training run), this module answers "forecast server X for N days with
model Y" dynamically, on every request:

- ARIMA / SARIMA: refit per-server on the fly (fast for a single series).
- XGBoost: uses the MLflow-registered model with a recursive multi-step
  loop, recomputing lag/rolling features after each predicted day.
- GRU is intentionally NOT live here: it was trained once as a single
  model across all servers and never persisted as a reusable artifact
  (see training.py), so refitting it per-request would be too slow for
  a synchronous UI call. Its numbers remain available only as the
  historical batch predictions in artifacts/gru/predictions.csv.

Rightsizing thresholds and the illustrative cost constant below are
demo defaults, not calibrated capacity-planning guidance - tune them
for a real environment.
"""

import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import generate_synthetic_data, validate_data
from .features import build_features
from .predictions import PredictionStore

# Must match api.FEATURE_COLUMNS / training._prepare_ml_data exactly -
# this is the exact column set + order the registered XGBoost model expects.
FEATURE_COLUMNS = [
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

LIVE_MODELS = ("xgboost", "arima", "sarima")
MAX_HORIZON_DAYS = 90
MIN_HISTORY_FOR_XGBOOST = 30  # matches the max lag window in features.py

# Rightsizing thresholds (illustrative demo defaults).
UNDERSIZED_PEAK_THRESHOLD = 85.0
OVERSIZED_AVG_THRESHOLD = 20.0
OVERSIZED_PEAK_THRESHOLD = 35.0
TARGET_UTIL_HIGH = 75.0
TARGET_UTIL_LOW = 50.0
COST_PER_CORE_MONTH_USD = 30.0  # illustrative only

MODEL_INFO = {
    "xgboost": {
        "label": "XGBoost",
        "description": (
            "Gradient-boosted trees over engineered lag/rolling/ratio features. "
            "Captures non-linear, multivariate patterns (CPU+RAM+Disk+Network)."
        ),
        "suited_for": "Medium-to-large history (90+ days), many servers, multivariate signals.",
        "live": True,
    },
    "arima": {
        "label": "ARIMA",
        "description": "Classical autoregressive model fit per-server on the CPU series alone.",
        "suited_for": "Short history (<60 days) or a single univariate series; quick baseline.",
        "live": True,
    },
    "sarima": {
        "label": "SARIMA",
        "description": "ARIMA plus explicit weekly seasonality (7-day cycle).",
        "suited_for": "Series with a clear weekly pattern and several full cycles of history.",
        "live": True,
    },
    "gru": {
        "label": "GRU (batch only)",
        "description": (
            "Recurrent neural net trained once across all servers. Not persisted as a "
            "reusable artifact, so it cannot be refit live within a request."
        ),
        "suited_for": "Large multivariate datasets trained offline; shown here for comparison only.",
        "live": False,
    },
}


def recursive_xgboost_forecast(hist: pd.DataFrame, horizon_days: int, predict_fn) -> list[tuple[pd.Timestamp, float]]:
    """Recursive multi-step XGBoost forecast, shared by ForecastService and the offline
    model-comparison analysis so both use the exact same forecasting logic.

    predict_fn: callable(DataFrame[FEATURE_COLUMNS]) -> array-like of predictions
    (works with both an mlflow.pyfunc model and a raw xgboost.XGBRegressor).
    """
    working = hist[["server_id", "timestamp", "cpu_utilization", "ram_utilization", "disk_utilization", "network_utilization"]].copy()
    ram_persist = float(working["ram_utilization"].tail(7).mean())
    disk_persist = float(working["disk_utilization"].tail(7).mean())
    net_persist = float(working["network_utilization"].tail(7).mean())
    last_date = working["timestamp"].max()

    results: list[tuple[pd.Timestamp, float]] = []
    for step in range(1, horizon_days + 1):
        next_date = last_date + timedelta(days=step)
        placeholder_cpu = float(working["cpu_utilization"].iloc[-1])  # proxy for this-day ratio features only
        new_row = pd.DataFrame(
            [
                {
                    "server_id": working["server_id"].iloc[0],
                    "timestamp": next_date,
                    "cpu_utilization": placeholder_cpu,
                    "ram_utilization": ram_persist,
                    "disk_utilization": disk_persist,
                    "network_utilization": net_persist,
                }
            ]
        )
        working = pd.concat([working, new_row], ignore_index=True)
        feats = build_features(working)
        row = feats[feats["timestamp"] == next_date]
        if row.empty:
            raise ValueError("Not enough history to compute lag/rolling features for the requested horizon.")
        X = row[FEATURE_COLUMNS]
        pred = predict_fn(X)
        pred = float(pred.tolist()[0]) if hasattr(pred, "tolist") else float(pred)
        pred = float(np.clip(pred, 0, 100))
        working.loc[working["timestamp"] == next_date, "cpu_utilization"] = pred
        results.append((next_date, pred))
    return results


class ForecastService:
    def __init__(
        self,
        xgb_model: Any = None,
        xgb_load_error: str | None = None,
        n_servers: int = 20,
        n_days: int = 150,
        seed: int = 42,
        artifacts_dir: str | Path = "artifacts",
        mlruns_comparison_path: str | Path = "mlruns/model_comparison.csv",
    ) -> None:
        self._xgb_model = xgb_model
        self._xgb_load_error = xgb_load_error
        self._raw = validate_data(generate_synthetic_data(n_servers=n_servers, n_days=n_days, seed=seed))
        self._server_meta = (
            self._raw.groupby("server_id")
            .agg(cpu_cores=("cpu_cores", "first"), installed_ram=("installed_ram", "first"))
            .to_dict(orient="index")
        )
        self._store = PredictionStore(artifacts_dir)
        self._mae_by_model = self._load_mae_comparison(mlruns_comparison_path)

    @staticmethod
    def _load_mae_comparison(path: str | Path) -> dict[str, float]:
        try:
            df = pd.read_csv(path)
            return {str(row["model"]): float(row["mae"]) for _, row in df.iterrows()}
        except Exception:
            return {}

    def _cores_for(self, server_id: str) -> int:
        meta = self._server_meta.get(server_id)
        return int(meta["cpu_cores"]) if meta else 8  # fallback for servers outside the synthetic set (e.g. App-101)

    def list_servers(self) -> list[dict[str, Any]]:
        latest = self._raw.sort_values("timestamp").groupby("server_id").tail(1)
        history_counts = self._raw.groupby("server_id").size()
        out = []
        for _, row in latest.iterrows():
            server_id = row["server_id"]
            out.append(
                {
                    "server_id": server_id,
                    "cpu_cores": int(row["cpu_cores"]),
                    "installed_ram_gb": int(row["installed_ram"]),
                    "current_cpu": round(float(row["cpu_utilization"]), 1),
                    "current_ram": round(float(row["ram_utilization"]), 1),
                    "history_days": int(history_counts.get(server_id, 0)),
                }
            )
        return sorted(out, key=lambda s: s["server_id"])

    def model_metrics(self) -> list[dict[str, Any]]:
        out = []
        for name, info in MODEL_INFO.items():
            row = {"model": name, **info}
            if name in self._mae_by_model:
                row["mae"] = round(self._mae_by_model[name], 3)
            out.append(row)
        return sorted(out, key=lambda r: r.get("mae", float("inf")))

    def _server_history(self, server_id: str) -> pd.DataFrame:
        return self._raw[self._raw["server_id"] == server_id].sort_values("timestamp").reset_index(drop=True)

    def _select_model(self, n_points: int) -> tuple[str, str]:
        if n_points < 60:
            return "arima", f"{n_points} historical days available (<60): ARIMA needs less data and is less likely to overfit a short series."
        candidates = {k: v for k, v in self._mae_by_model.items() if k in LIVE_MODELS}
        if candidates and self._xgb_model is not None:
            best = min(candidates, key=candidates.get)
            return best, (
                f"{n_points} historical days available; {MODEL_INFO[best]['label']} had the lowest MAE "
                f"({candidates[best]:.2f}) among live-capable models in the last training comparison."
            )
        if self._xgb_model is not None:
            return "xgboost", f"{n_points} historical days available (>=90): defaulting to XGBoost."
        return "arima", "XGBoost model is not loaded; falling back to ARIMA."

    def _forecast_statsmodels(self, hist: pd.DataFrame, horizon_days: int, seasonal: bool) -> list[tuple[pd.Timestamp, float]]:
        series = hist.sort_values("timestamp")["cpu_utilization"].astype(float).reset_index(drop=True)
        last_date = hist["timestamp"].max()
        if seasonal:
            fitted = SARIMAX(
                series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
        else:
            fitted = ARIMA(series, order=(3, 1, 1)).fit()
        raw_forecast = fitted.forecast(steps=horizon_days)
        values = np.clip(np.asarray(raw_forecast, dtype=float), 0, 100)
        return [(last_date + timedelta(days=i + 1), float(v)) for i, v in enumerate(values)]

    def _forecast_xgboost(self, hist: pd.DataFrame, horizon_days: int) -> list[tuple[pd.Timestamp, float]]:
        if self._xgb_model is None:
            raise ValueError(f"XGBoost model is not loaded: {self._xgb_load_error or 'unknown error'}")
        if len(hist) < MIN_HISTORY_FOR_XGBOOST:
            raise ValueError(f"Need at least {MIN_HISTORY_FOR_XGBOOST} historical days for XGBoost; only {len(hist)} available.")
        return recursive_xgboost_forecast(hist, horizon_days, self._xgb_model.predict)

    def _rightsizing(self, peak: float, avg: float, cpu_cores: int) -> dict[str, Any]:
        if peak >= UNDERSIZED_PEAK_THRESHOLD:
            recommended_cores = max(cpu_cores + 1, math.ceil(cpu_cores * peak / TARGET_UTIL_HIGH))
            delta = recommended_cores - cpu_cores
            return {
                "status": "undersized",
                "label": "Undersized / At Risk",
                "message": f"Peak predicted CPU is {peak:.1f}% (avg {avg:.1f}%) against {cpu_cores} vCPUs.",
                "action": f"Scale up from {cpu_cores} to {recommended_cores} vCPUs (+{delta}) to bring peak utilization down to roughly {TARGET_UTIL_HIGH:.0f}%.",
                "recommended_cores": recommended_cores,
                "core_delta": delta,
                "estimated_monthly_cost_delta_usd": round(delta * COST_PER_CORE_MONTH_USD, 2),
            }
        if avg <= OVERSIZED_AVG_THRESHOLD and peak <= OVERSIZED_PEAK_THRESHOLD:
            raw_target = math.floor(cpu_cores * max(avg, 1.0) / TARGET_UTIL_LOW)
            recommended_cores = max(2, min(raw_target, cpu_cores - 1) if cpu_cores > 2 else cpu_cores)
            delta = recommended_cores - cpu_cores
            return {
                "status": "oversized",
                "label": "Oversized / Underutilized",
                "message": f"Average predicted CPU is only {avg:.1f}% (peak {peak:.1f}%) against {cpu_cores} vCPUs.",
                "action": f"Downsize from {cpu_cores} to {recommended_cores} vCPUs ({delta}) or consolidate this workload onto another server.",
                "recommended_cores": recommended_cores,
                "core_delta": delta,
                "estimated_monthly_cost_delta_usd": round(delta * COST_PER_CORE_MONTH_USD, 2),
            }
        return {
            "status": "rightsized",
            "label": "Rightsized",
            "message": f"Predicted CPU stays within a healthy band (avg {avg:.1f}%, peak {peak:.1f}%) for {cpu_cores} vCPUs.",
            "action": "No sizing change recommended; continue routine monitoring.",
            "recommended_cores": cpu_cores,
            "core_delta": 0,
            "estimated_monthly_cost_delta_usd": 0.0,
        }

    def forecast(self, server_id: str, horizon_days: int = 7, model: str = "auto") -> dict[str, Any]:
        if horizon_days < 1 or horizon_days > MAX_HORIZON_DAYS:
            raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}.")
        hist = self._server_history(server_id)
        if hist.empty:
            raise ValueError(f"Unknown server_id: {server_id}")

        model_reason = None
        if model == "auto":
            model, model_reason = self._select_model(len(hist))
        elif model not in LIVE_MODELS:
            raise ValueError(f"Unsupported live model: {model}. Supported: auto, {', '.join(LIVE_MODELS)}.")
        else:
            model_reason = f"Manually selected {MODEL_INFO[model]['label']}."

        if model == "xgboost":
            raw_points = self._forecast_xgboost(hist, horizon_days)
        elif model == "sarima":
            raw_points = self._forecast_statsmodels(hist, horizon_days, seasonal=True)
        else:
            raw_points = self._forecast_statsmodels(hist, horizon_days, seasonal=False)

        points = [{"date": d.strftime("%Y-%m-%d"), "predicted_cpu": round(v, 2)} for d, v in raw_points]
        values = [p["predicted_cpu"] for p in points]
        peak, avg, low = max(values), sum(values) / len(values), min(values)
        cpu_cores = self._cores_for(server_id)
        recommendation = self._rightsizing(peak, avg, cpu_cores)

        return {
            "server_id": server_id,
            "model_used": model,
            "model_selection_reason": model_reason,
            "horizon_days": horizon_days,
            "points": points,
            "peak_predicted_cpu": round(peak, 1),
            "avg_predicted_cpu": round(avg, 1),
            "min_predicted_cpu": round(low, 1),
            "cpu_cores": cpu_cores,
            "recommendation": recommendation,
        }

    def fleet_overview(self, horizon_days: int = 7, model: str = "xgboost") -> list[dict[str, Any]]:
        frame = self._store.frame
        if frame.empty:
            return []
        subset = frame[frame["model"] == model]
        if subset.empty:
            subset = frame
        start = subset["date"].min()
        end = start + timedelta(days=horizon_days - 1)
        window = subset[(subset["date"] >= start) & (subset["date"] <= end)]
        grouped = window.groupby("server_id")["prediction"].agg(["max", "mean"]).reset_index()

        out = []
        for _, row in grouped.iterrows():
            server_id = row["server_id"]
            cpu_cores = self._cores_for(server_id)
            rec = self._rightsizing(float(row["max"]), float(row["mean"]), cpu_cores)
            out.append(
                {
                    "server_id": server_id,
                    "peak_predicted_cpu": round(float(row["max"]), 1),
                    "avg_predicted_cpu": round(float(row["mean"]), 1),
                    "cpu_cores": cpu_cores,
                    **rec,
                }
            )
        return sorted(out, key=lambda r: r["peak_predicted_cpu"], reverse=True)
