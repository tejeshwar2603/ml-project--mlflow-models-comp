"""Comprehensive offline model-comparison analysis.

Produces the data + plots behind 7 comparison "tabs" (forecast-vs-actual,
error distribution, horizon degradation, dataset-size impact, null-value
handling, a radar scorecard, and business metrics) plus a Diebold-Mariano
significance test and a per-server error heatmap.

Everything here is measured against THIS repo's actual synthetic dataset
and THIS repo's actual model/feature code (not generic ML folklore) - where
a result differs from common wisdom (e.g. "XGBoost needs 30+ days here
because of the lag_30 feature", not "XGBoost works fine on 7 days"), that's
reported honestly rather than forced to match expectations.

Run manually (not triggered by anything automatically - there's no
scheduler/Airflow in this repo):

    python -m src.forecasting.model_comparison

Writes:
  - artifacts/model_comparison/results.json   (consumed by GET /model-comparison)
  - artifacts/model_comparison/plots/*.png    (also logged to MLflow)
  - an MLflow run "model_comparison_analysis" with all plots + metrics + the JSON
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import generate_synthetic_data, validate_data
from .evaluation import evaluate_forecast
from .features import build_features
from .forecast_service import recursive_xgboost_forecast
from .models import ARIMAForecaster, GRUForecaster, SARIMAForecaster, XGBoostForecaster
from .training import _prepare_ml_data, time_split

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path("artifacts/model_comparison")
PLOTS_DIR = ARTIFACTS_DIR / "plots"

LIVE_MODELS = ["arima", "sarima", "xgboost"]
ALL_MODELS = ["arima", "sarima", "xgboost", "gru"]
STUDY_SERVERS = [f"server-{i:03d}" for i in range(1, 6)]  # 5 servers for expensive studies
EXAMPLE_SERVER = "server-001"

HORIZONS = [1, 3, 7, 14, 21, 30]
ORIGIN_FRACTIONS = [0.55, 0.7, 0.85]
MISSING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
DATASET_SIZE_TIERS = {"7d": 7, "28d": 28, "60d": 60, "90d": 90, "150d": 150, "300d": 300}
UNDERSIZED_PEAK_THRESHOLD = 85.0
DECISION_HORIZON = 7  # classic capacity-planning window for the business-metrics study


def _safe_mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.clip(np.abs(actual), 1.0, None)  # avoid blow-up near-zero CPU%
    return float(np.mean(np.abs(actual - pred) / denom) * 100)


def _safe_mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float))))


# ---------------------------------------------------------------------------
# Base fit: standard train/val/test split, all 4 models, used by tabs 1/2/6/DM/heatmap
# ---------------------------------------------------------------------------
def _fit_and_predict_all(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.Series]:
    X_train, y_train = _prepare_ml_data(train)
    X_test, y_test = _prepare_ml_data(test)

    models = {
        "arima": ARIMAForecaster(order=(3, 1, 1)),
        "sarima": SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)),
        "xgboost": XGBoostForecaster(params={"objective": "reg:squarederror", "verbosity": 0}, num_round=100),
        "gru": GRUForecaster(seq_len=14, hidden_size=32, lr=0.005, epochs=5, batch_size=16),
    }
    results = {}
    fitted_xgb_regressor = None
    for name, model in models.items():
        t0 = time.time()
        if name in ("arima", "sarima"):
            model.fit(train[["server_id", "timestamp", "cpu_utilization"]])
            y_pred = model.predict(test[["server_id", "timestamp"]])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if name == "xgboost":
                fitted_xgb_regressor = model.model
        fit_seconds = time.time() - t0
        y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=float(y_train.mean()))
        results[name] = {"y_pred": y_pred, "fit_seconds": round(fit_seconds, 3)}
    results["_fitted_xgb_regressor"] = fitted_xgb_regressor
    return results, y_test.reset_index(drop=True)


def _forecast_vs_actual(test: pd.DataFrame, y_test: pd.Series, model_results: dict, server_id: str) -> dict:
    mask = (test["server_id"] == server_id).values
    dates = test.loc[mask, "timestamp"].dt.strftime("%Y-%m-%d").tolist()
    actual = np.asarray(y_test)[mask]
    series = {"server_id": server_id, "dates": dates, "actual": [round(v, 2) for v in actual.tolist()], "models": {}}
    for name in LIVE_MODELS + ["gru"]:
        pred = np.asarray(model_results[name]["y_pred"])[mask]
        residual = pred - actual
        series["models"][name] = {
            "predicted": [round(v, 2) for v in pred.tolist()],
            "residual": [round(v, 2) for v in residual.tolist()],
        }
    return series


def _error_distribution(y_test: pd.Series, model_results: dict) -> dict:
    y_arr = np.asarray(y_test)
    out = {}
    for name in ALL_MODELS:
        pred = np.asarray(model_results[name]["y_pred"])
        err = pred - y_arr
        counts, edges = np.histogram(err, bins=20)
        q1, med, q3 = np.percentile(err, [25, 50, 75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        within = err[(err >= lower_fence) & (err <= upper_fence)]
        whisker_low = float(within.min()) if len(within) else float(err.min())
        whisker_high = float(within.max()) if len(within) else float(err.max())
        out[name] = {
            "histogram": {"counts": counts.tolist(), "bin_edges": [round(e, 2) for e in edges.tolist()]},
            "boxplot": {
                "min": round(whisker_low, 2),
                "q1": round(float(q1), 2),
                "median": round(float(med), 2),
                "q3": round(float(q3), 2),
                "max": round(whisker_high, 2),
            },
            "skew": round(float(stats.skew(err)), 3),
            "mean_error": round(float(err.mean()), 3),
            "std_error": round(float(err.std()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# Walk-forward study: powers horizon degradation (tab 3) + business metrics (tab 7)
# ---------------------------------------------------------------------------
def _walk_forward_study(raw_df: pd.DataFrame, xgb_predict_fn) -> dict:
    horizon_errors = {m: {h: [] for h in HORIZONS} for m in LIVE_MODELS}
    decisions = {m: [] for m in LIVE_MODELS}  # list of (predicted_flag, actual_flag)

    for server_id in STUDY_SERVERS:
        hist = raw_df[raw_df["server_id"] == server_id].sort_values("timestamp").reset_index(drop=True)
        n = len(hist)
        for frac in ORIGIN_FRACTIONS:
            cutoff = int(n * frac)
            if n - cutoff <= HORIZONS[-1]:
                continue
            train_part = hist.iloc[:cutoff]
            future_part = hist.iloc[cutoff : cutoff + HORIZONS[-1]]
            actual_vals = future_part["cpu_utilization"].values

            forecasts = {}
            try:
                fitted = ARIMA(train_part["cpu_utilization"].astype(float), order=(3, 1, 1)).fit()
                forecasts["arima"] = np.clip(np.asarray(fitted.forecast(steps=HORIZONS[-1])), 0, 100)
            except Exception:
                forecasts["arima"] = None
            try:
                fitted = SARIMAX(
                    train_part["cpu_utilization"].astype(float),
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                forecasts["sarima"] = np.clip(np.asarray(fitted.forecast(steps=HORIZONS[-1])), 0, 100)
            except Exception:
                forecasts["sarima"] = None
            try:
                xgb_points = recursive_xgboost_forecast(train_part, HORIZONS[-1], xgb_predict_fn)
                forecasts["xgboost"] = np.asarray([v for _, v in xgb_points])
            except Exception:
                forecasts["xgboost"] = None

            for name, fc in forecasts.items():
                if fc is None:
                    continue
                for h in HORIZONS:
                    horizon_errors[name][h].append(_safe_mape(actual_vals[:h], fc[:h]))
                predicted_flag = bool(np.max(fc[:DECISION_HORIZON]) >= UNDERSIZED_PEAK_THRESHOLD)
                actual_flag = bool(np.max(actual_vals[:DECISION_HORIZON]) >= UNDERSIZED_PEAK_THRESHOLD)
                decisions[name].append((predicted_flag, actual_flag))

    horizon_curve = {
        m: {str(h): (round(float(np.mean(v)), 2) if v else None) for h, v in horizon_errors[m].items()}
        for m in LIVE_MODELS
    }
    horizon_stability = {
        m: (round(float(np.std([v for vals in horizon_errors[m].values() for v in vals])), 2) if horizon_errors[m][HORIZONS[0]] else None)
        for m in LIVE_MODELS
    }

    business = {}
    for m in LIVE_MODELS:
        pairs = decisions[m]
        n_total = len(pairs)
        if n_total == 0:
            business[m] = None
            continue
        correct = sum(1 for p, a in pairs if p == a)
        critical_misses = sum(1 for p, a in pairs if a and not p)  # real risk, model said hold
        false_alarms = sum(1 for p, a in pairs if p and not a)  # model said scale, no real risk
        actual_positives = sum(1 for _, a in pairs if a)
        predicted_positives = sum(1 for p, _ in pairs if p)
        business[m] = {
            "n_decisions": n_total,
            "decision_accuracy_pct": round(100 * correct / n_total, 1),
            "critical_miss_rate_pct": round(100 * critical_misses / actual_positives, 1) if actual_positives else 0.0,
            "false_alarm_rate_pct": round(100 * false_alarms / predicted_positives, 1) if predicted_positives else 0.0,
            "actual_risk_events": actual_positives,
        }

    return {"horizon_curve": horizon_curve, "horizon_mape_stddev": horizon_stability, "business": business}


# ---------------------------------------------------------------------------
# Tab 4: dataset size impact / learning curves
# ---------------------------------------------------------------------------
def _dataset_size_study() -> dict:
    big = validate_data(generate_synthetic_data(n_servers=len(STUDY_SERVERS), n_days=340, seed=42))
    test_block = big.groupby("server_id", group_keys=False).apply(lambda g: g.tail(30))
    train_pool = big.groupby("server_id", group_keys=False).apply(lambda g: g.iloc[:-30])

    results = {m: {} for m in ALL_MODELS}
    notes = {m: {} for m in ALL_MODELS}
    for tier_label, tier_days in DATASET_SIZE_TIERS.items():
        tier_train = train_pool.groupby("server_id", group_keys=False).apply(lambda g: g.tail(tier_days))
        actual_by_server = {s: g.sort_values("timestamp")["cpu_utilization"].values for s, g in test_block.groupby("server_id")}

        # ARIMA / SARIMA: per-server refit
        for name, cls, kwargs in (
            ("arima", ARIMAForecaster, {"order": (3, 1, 1)}),
            ("sarima", SARIMAForecaster, {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)}),
        ):
            maes = []
            try:
                for server_id, actual in actual_by_server.items():
                    series = tier_train[tier_train["server_id"] == server_id].sort_values("timestamp")["cpu_utilization"]
                    fc = cls(**kwargs)._fit_single(series.astype(float))
                    pred = np.clip(np.asarray(fc.forecast(steps=30)), 0, 100)
                    maes.append(_safe_mae(actual, pred))
                results[name][tier_label] = round(float(np.mean(maes)), 2)
            except Exception as exc:
                results[name][tier_label] = None
                notes[name][tier_label] = str(exc)[:160]

        # XGBoost: needs lag_30 -> structurally requires >30 days of history per server
        try:
            feats = build_features(tier_train)
            if feats.empty:
                raise ValueError(f"build_features dropped all rows: needs >30 days of history (lag_30), tier has {tier_days} days.")
            X_tr, y_tr = _prepare_ml_data(feats)
            xgb = XGBoostForecaster(params={"objective": "reg:squarederror", "verbosity": 0}, num_round=100)
            xgb.fit(X_tr, y_tr)
            maes = []
            for server_id, actual in actual_by_server.items():
                hist = tier_train[tier_train["server_id"] == server_id].sort_values("timestamp")
                if len(hist) <= 30:
                    raise ValueError("insufficient per-server history for recursive forecast")
                points = recursive_xgboost_forecast(hist, 30, xgb.model.predict)
                pred = np.asarray([v for _, v in points])
                maes.append(_safe_mae(actual, pred))
            results["xgboost"][tier_label] = round(float(np.mean(maes)), 2)
        except Exception as exc:
            results["xgboost"][tier_label] = None
            notes["xgboost"][tier_label] = str(exc)[:160]

        # GRU: needs seq_len=14 sliding windows -> structurally requires >14 days
        try:
            feats = build_features(tier_train)
            if feats.empty or len(feats) < 30:
                raise ValueError(f"Not enough post-feature-engineering rows ({len(feats)}) for a GRU sequence window at {tier_days} days.")
            X_tr, y_tr = _prepare_ml_data(feats)
            gru = GRUForecaster(seq_len=14, hidden_size=32, lr=0.005, epochs=5, batch_size=16)
            gru.fit(X_tr, y_tr)
            # Evaluate on the held-out test block's features (built against the tier's tail for lag continuity)
            combined = pd.concat([tier_train, test_block[test_block["server_id"].isin(tier_train["server_id"].unique())]])
            combined_feats = build_features(combined)
            X_eval, y_eval = _prepare_ml_data(combined_feats)
            y_pred = gru.predict(X_eval)
            y_pred = np.nan_to_num(y_pred, nan=float(y_tr.mean()))
            eval_mask = combined_feats["timestamp"].isin(test_block["timestamp"])
            if eval_mask.sum() == 0:
                raise ValueError("no overlapping evaluation rows survived feature engineering")
            results["gru"][tier_label] = round(_safe_mae(y_eval[eval_mask.values], y_pred[eval_mask.values]), 2)
        except Exception as exc:
            results["gru"][tier_label] = None
            notes["gru"][tier_label] = str(exc)[:160]

    return {"tiers": list(DATASET_SIZE_TIERS.keys()), "tier_days": DATASET_SIZE_TIERS, "mae_by_model": results, "failure_notes": notes}


# ---------------------------------------------------------------------------
# Tab 5: null-value handling robustness
# ---------------------------------------------------------------------------
def _null_handling_study(raw_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    results = {m: {} for m in LIVE_MODELS}
    for rate in MISSING_RATES:
        maes = {m: [] for m in LIVE_MODELS}
        for server_id in STUDY_SERVERS:
            hist = raw_df[raw_df["server_id"] == server_id].sort_values("timestamp").reset_index(drop=True)
            n = len(hist)
            cutoff = n - 30
            train_part = hist.iloc[:cutoff].copy()
            actual = hist.iloc[cutoff:]["cpu_utilization"].values
            mask = rng.random(len(train_part)) < rate
            noisy = train_part.copy()
            noisy.loc[noisy.index[mask], "cpu_utilization"] = np.nan

            # ARIMA / SARIMA fed the raw (unimputed) series directly - statsmodels' missing="drop"
            # just excises missing points rather than truly imputing them.
            try:
                fitted = ARIMA(noisy["cpu_utilization"].astype(float), order=(3, 1, 1), missing="drop").fit()
                fc = np.clip(np.asarray(fitted.forecast(steps=30)), 0, 100)
                maes["arima"].append(_safe_mae(actual, fc))
            except Exception:
                maes["arima"].append(None)
            try:
                fitted = SARIMAX(
                    noisy["cpu_utilization"].astype(float),
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    missing="drop",
                ).fit(disp=False)
                fc = np.clip(np.asarray(fitted.forecast(steps=30)), 0, 100)
                maes["sarima"].append(_safe_mae(actual, fc))
            except Exception:
                maes["sarima"].append(None)

            # XGBoost: goes through the existing build_features -> validate_and_fill pipeline,
            # which already interpolates/ffills/bfills missing values before training.
            try:
                feats = build_features(noisy)
                if feats.empty:
                    raise ValueError("all rows dropped after imputation + lag_30 requirement")
                X_tr, y_tr = _prepare_ml_data(feats)
                xgb = XGBoostForecaster(params={"objective": "reg:squarederror", "verbosity": 0}, num_round=80)
                xgb.fit(X_tr, y_tr)
                points = recursive_xgboost_forecast(noisy, 30, xgb.model.predict)
                fc = np.asarray([v for _, v in points])
                maes["xgboost"].append(_safe_mae(actual, fc))
            except Exception:
                maes["xgboost"].append(None)

        for m in LIVE_MODELS:
            vals = [v for v in maes[m] if v is not None]
            results[m][f"{int(rate * 100)}%"] = round(float(np.mean(vals)), 2) if vals else None
    return results


# ---------------------------------------------------------------------------
# Tab 6: radar scorecard (mix of measured + documented-qualitative axes)
# ---------------------------------------------------------------------------
def _normalize_inverse(values: dict[str, float | None]) -> dict[str, float]:
    """Lower raw value = better -> higher 0-100 score. None -> 0 (worst, missing data)."""
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return {k: 0.0 for k in values}
    lo, hi = min(valid.values()), max(valid.values())
    span = (hi - lo) or 1.0
    return {k: (round(100 * (1 - (v - lo) / span), 1) if v is not None else 0.0) for k, v in values.items()}


QUALITATIVE_SCORES = {
    # Documented judgment calls, NOT measured from this demo's data - based on each
    # algorithm's known structural properties. Kept separate from measured axes in the UI.
    "scalability": {
        "xgboost": 90, "gru": 80, "arima": 30, "sarima": 25,
        "note": "ARIMA/SARIMA fit one model per server sequentially; XGBoost/GRU train once across all servers and parallelize.",
    },
    "interpretability": {
        "arima": 85, "sarima": 80, "xgboost": 55, "gru": 20,
        "note": "ARIMA/SARIMA coefficients are directly interpretable; XGBoost offers feature importance/SHAP; GRU is largely a black box.",
    },
}


def _radar_scores(model_results: dict, y_test: pd.Series, null_study: dict, walk_forward: dict) -> dict:
    y_arr = np.asarray(y_test)
    mae_by_model = {m: _safe_mae(y_arr, model_results[m]["y_pred"]) for m in ALL_MODELS}
    speed_by_model = {m: model_results[m]["fit_seconds"] for m in ALL_MODELS}
    null_50 = {m: null_study.get(m, {}).get("50%") for m in LIVE_MODELS}
    short_term = {m: walk_forward["horizon_curve"][m].get("1") for m in LIVE_MODELS}
    long_term = {m: walk_forward["horizon_curve"][m].get("30") for m in LIVE_MODELS}
    stability = {m: walk_forward["horizon_mape_stddev"][m] for m in LIVE_MODELS}

    accuracy = _normalize_inverse(mae_by_model)
    speed = _normalize_inverse(speed_by_model)
    null_handling = _normalize_inverse(null_50)
    short_term_score = _normalize_inverse(short_term)
    long_term_score = _normalize_inverse(long_term)
    stability_score = _normalize_inverse(stability)

    radar = {}
    for m in ALL_MODELS:
        radar[m] = {
            "accuracy": accuracy.get(m, 0.0),
            "speed": speed.get(m, 0.0),
            "null_handling": null_handling.get(m, 0.0) if m in LIVE_MODELS else None,
            "short_term": short_term_score.get(m, 0.0) if m in LIVE_MODELS else None,
            "long_term": long_term_score.get(m, 0.0) if m in LIVE_MODELS else None,
            "stability": stability_score.get(m, 0.0) if m in LIVE_MODELS else None,
            "scalability": QUALITATIVE_SCORES["scalability"].get(m),
            "interpretability": QUALITATIVE_SCORES["interpretability"].get(m),
        }
    return {
        "scores": radar,
        "axes_basis": {
            "accuracy": "measured", "speed": "measured", "null_handling": "measured",
            "short_term": "measured", "long_term": "measured", "stability": "measured",
            "scalability": "qualitative", "interpretability": "qualitative",
        },
        "qualitative_notes": {
            "scalability": QUALITATIVE_SCORES["scalability"]["note"],
            "interpretability": QUALITATIVE_SCORES["interpretability"]["note"],
        },
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano significance test + per-server heatmap
# ---------------------------------------------------------------------------
def _diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0 or n < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_5pct": False}
    dm_stat = dbar / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"statistic": round(float(dm_stat), 3), "p_value": round(float(p_value), 4), "significant_at_5pct": bool(p_value < 0.05)}


def _dm_matrix(y_test: pd.Series, model_results: dict) -> dict:
    y_arr = np.asarray(y_test)
    errors = {m: np.asarray(model_results[m]["y_pred"]) - y_arr for m in ALL_MODELS}
    out = {}
    for i, m1 in enumerate(ALL_MODELS):
        for m2 in ALL_MODELS[i + 1 :]:
            result = _diebold_mariano(errors[m1], errors[m2])
            result["interpretation"] = (
                f"{'Statistically significant difference' if result['significant_at_5pct'] else 'No statistically significant difference'} "
                f"between {m1} and {m2} (p={result['p_value']})."
            )
            out[f"{m1}_vs_{m2}"] = result
    return out


def _per_server_heatmap(test: pd.DataFrame, y_test: pd.Series, model_results: dict, raw: pd.DataFrame) -> dict:
    df = test[["server_id"]].copy().reset_index(drop=True)
    df["actual"] = np.asarray(y_test)
    servers = sorted(df["server_id"].unique())
    matrix = {}
    for name in ALL_MODELS:
        df["pred"] = np.asarray(model_results[name]["y_pred"])
        df["err"] = (df["pred"] - df["actual"]).abs()
        per_server = df.groupby("server_id")["err"].mean()
        matrix[name] = [round(float(per_server.get(s, np.nan)), 2) for s in servers]

    specs = raw.groupby("server_id").agg(cpu_cores=("cpu_cores", "first"), installed_ram=("installed_ram", "first"))
    server_info = [
        {
            "server_id": s,
            "cpu_cores": int(specs.loc[s, "cpu_cores"]) if s in specs.index else None,
            "installed_ram_gb": int(specs.loc[s, "installed_ram"]) if s in specs.index else None,
        }
        for s in servers
    ]
    return {"servers": servers, "server_info": server_info, "models": ALL_MODELS, "mae_matrix": matrix}


# ---------------------------------------------------------------------------
# Plotting (logged to MLflow + saved to artifacts/model_comparison/plots)
# ---------------------------------------------------------------------------
def _save_plots(results: dict) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # Tab 1: forecast vs actual + residuals
    fva = results["forecast_vs_actual"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(fva["dates"], fva["actual"], label="actual", color="black", linewidth=2)
    for name in LIVE_MODELS:
        ax1.plot(fva["dates"], fva["models"][name]["predicted"], label=name, alpha=0.8)
    ax1.set_title(f"Forecast vs Actual - {fva['server_id']}")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    for name in LIVE_MODELS:
        ax2.plot(fva["dates"], fva["models"][name]["residual"], label=name, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("Residuals (predicted - actual)")
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p = PLOTS_DIR / "01_forecast_vs_actual.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 2: error distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in ALL_MODELS:
        edges = results["error_distribution"][name]["histogram"]["bin_edges"]
        counts = results["error_distribution"][name]["histogram"]["counts"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        axes[0].plot(centers, counts, label=name, alpha=0.8)
    axes[0].set_title("Error distribution (predicted - actual)")
    axes[0].legend()
    box_data = [
        [
            results["error_distribution"][name]["boxplot"]["min"],
            results["error_distribution"][name]["boxplot"]["q1"],
            results["error_distribution"][name]["boxplot"]["median"],
            results["error_distribution"][name]["boxplot"]["q3"],
            results["error_distribution"][name]["boxplot"]["max"],
        ]
        for name in ALL_MODELS
    ]
    axes[1].boxplot(box_data, tick_labels=ALL_MODELS, showfliers=False)
    axes[1].set_title("Error box plot")
    fig.tight_layout()
    p = PLOTS_DIR / "02_error_distribution.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 3: horizon degradation
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in LIVE_MODELS:
        curve = results["walk_forward"]["horizon_curve"][name]
        xs = [int(h) for h in curve.keys()]
        ys = [curve[str(h)] for h in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Horizon degradation (walk-forward)")
    ax.legend()
    fig.tight_layout()
    p = PLOTS_DIR / "03_horizon_degradation.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 4: dataset size impact
    fig, ax = plt.subplots(figsize=(8, 5))
    tiers = results["dataset_size"]["tiers"]
    for name in ALL_MODELS:
        ys = [results["dataset_size"]["mae_by_model"][name].get(t) for t in tiers]
        xs = [results["dataset_size"]["tier_days"][t] for t in tiers]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], marker="o", label=name)
    ax.set_xlabel("Training history (days)")
    ax.set_ylabel("MAE")
    ax.set_title("Dataset size impact / learning curves")
    ax.legend()
    fig.tight_layout()
    p = PLOTS_DIR / "04_dataset_size_impact.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 5: null handling
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in LIVE_MODELS:
        rates = list(results["null_handling"][name].keys())
        ys = [results["null_handling"][name][r] for r in rates]
        xs = [int(r.strip("%")) for r in rates]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], marker="o", label=name)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("MAE")
    ax.set_title("Robustness to missing data")
    ax.legend()
    fig.tight_layout()
    p = PLOTS_DIR / "05_null_handling.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 6: radar
    axes_order = ["accuracy", "speed", "null_handling", "short_term", "long_term", "stability", "scalability", "interpretability"]
    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name in ALL_MODELS:
        scores = results["radar"]["scores"][name]
        vals = [scores.get(a) or 0 for a in axes_order]
        vals += vals[:1]
        ax.plot(angles, vals, label=name)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_order)
    ax.set_title("Model tradeoff radar (0-100)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    p = PLOTS_DIR / "06_radar_chart.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Tab 7: business metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics_names = ["decision_accuracy_pct", "critical_miss_rate_pct", "false_alarm_rate_pct"]
    x = np.arange(len(LIVE_MODELS))
    width = 0.25
    for i, metric in enumerate(metrics_names):
        vals = [results["walk_forward"]["business"][m][metric] if results["walk_forward"]["business"][m] else 0 for m in LIVE_MODELS]
        ax1.bar(x + i * width, vals, width, label=metric.replace("_pct", ""))
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(LIVE_MODELS)
    ax1.set_title("Business decision metrics (%)")
    ax1.legend(fontsize=8)
    for name in ALL_MODELS:
        mae = _safe_mae(np.asarray(results["_y_test"]), results["_model_results"][name]["y_pred"])
        ax2.scatter(results["_model_results"][name]["fit_seconds"], mae, s=80, label=name)
    ax2.set_xlabel("Training time (s)")
    ax2.set_ylabel("MAE")
    ax2.set_title("Training time vs accuracy")
    ax2.legend()
    fig.tight_layout()
    p = PLOTS_DIR / "07_business_metrics.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Extra: per-server heatmap
    hm = results["per_server_heatmap"]
    fig, ax = plt.subplots(figsize=(6, 8))
    matrix = np.array([hm["mae_matrix"][m] for m in hm["models"]]).T
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(hm["models"])))
    ax.set_xticklabels(hm["models"])
    ax.set_yticks(range(len(hm["servers"])))
    y_labels = [f"{info['server_id']} ({info['cpu_cores']} vCPU)" for info in hm["server_info"]]
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title("Per-server MAE heatmap")
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    p = PLOTS_DIR / "08_per_server_heatmap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_comparison() -> dict:
    t_start = time.time()
    print("Generating base dataset (20 servers x 150 days, seed=42)...")
    raw = validate_data(generate_synthetic_data(n_servers=20, n_days=150, seed=42))
    features = build_features(raw)
    train, val, test = time_split(features, test_days=30, val_days=30)
    raw_history = raw[raw["timestamp"] < test["timestamp"].min()]  # for walk-forward/size/null studies

    print("Fitting all 4 models on the standard train/test split...")
    model_results, y_test = _fit_and_predict_all(train, test)
    fitted_xgb_regressor = model_results.pop("_fitted_xgb_regressor")

    print("Tab 1: forecast vs actual...")
    forecast_vs_actual = _forecast_vs_actual(test, y_test, model_results, EXAMPLE_SERVER)
    print("Tab 2: error distribution...")
    error_distribution = _error_distribution(y_test, model_results)

    print("Tab 3 + 7: walk-forward horizon degradation + business metrics (this takes a few minutes)...")
    walk_forward = _walk_forward_study(raw_history, fitted_xgb_regressor.predict)

    print("Tab 4: dataset size impact / learning curves...")
    dataset_size = _dataset_size_study()

    print("Tab 5: null-value handling robustness...")
    rng = np.random.default_rng(7)
    null_handling = _null_handling_study(raw_history, rng)

    print("Tab 6: radar scorecard...")
    radar = _radar_scores(model_results, y_test, null_handling, walk_forward)

    print("Diebold-Mariano tests + per-server heatmap...")
    dm_tests = _dm_matrix(y_test, model_results)
    per_server_heatmap = _per_server_heatmap(test, y_test, model_results, raw)

    mae_summary = {m: round(_safe_mae(np.asarray(y_test), model_results[m]["y_pred"]), 3) for m in ALL_MODELS}

    results = {
        "generated_in_seconds": None,  # filled below
        "mae_summary": mae_summary,
        "fit_seconds": {m: model_results[m]["fit_seconds"] for m in ALL_MODELS},
        "forecast_vs_actual": forecast_vs_actual,
        "error_distribution": error_distribution,
        "walk_forward": walk_forward,
        "dataset_size": dataset_size,
        "null_handling": null_handling,
        "radar": radar,
        "diebold_mariano": dm_tests,
        "per_server_heatmap": per_server_heatmap,
        "_y_test": y_test.tolist(),
        "_model_results": {m: {"y_pred": model_results[m]["y_pred"].tolist(), "fit_seconds": model_results[m]["fit_seconds"]} for m in ALL_MODELS},
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_paths = _save_plots(results)

    elapsed = round(time.time() - t_start, 1)
    results["generated_in_seconds"] = elapsed
    # Strip internal-only fields before persisting the public JSON contract
    public_results = {k: v for k, v in results.items() if not k.startswith("_")}

    results_path = ARTIFACTS_DIR / "results.json"
    results_path.write_text(json.dumps(public_results, indent=2))
    print(f"Wrote {results_path} in {elapsed}s")

    print("Logging to MLflow...")
    mlflow.set_experiment("cpu_forecasting")
    with mlflow.start_run(run_name="model_comparison_analysis"):
        for m, mae in mae_summary.items():
            mlflow.log_metric(f"{m}_test_mae", mae)
        for m, secs in results["fit_seconds"].items():
            mlflow.log_metric(f"{m}_fit_seconds", secs)
        for pair, dm in dm_tests.items():
            mlflow.log_metric(f"dm_pvalue_{pair}", dm["p_value"])
        mlflow.log_artifact(str(results_path), artifact_path="model_comparison")
        for p in plot_paths:
            mlflow.log_artifact(str(p), artifact_path="model_comparison/plots")
    print("Done. Logged plots + results.json to the 'cpu_forecasting' MLflow experiment (run: model_comparison_analysis).")

    return public_results


if __name__ == "__main__":
    run_comparison()
