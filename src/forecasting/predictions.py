import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
SERVER_PATTERN = re.compile(r"\b(?:App|app|server|Server|host|Host)[-_ ]?(\d+)\b")
THRESHOLD_PATTERN = re.compile(r"(\d{1,3})\s*%\s*cpu|cpu\s*(?:above|over|exceed(?:s|ing)?|>|>=)\s*(\d{1,3})", re.I)
HORIZON_PATTERN = re.compile(r"(?:next|within)\s+(\d+)\s+days?", re.I)
WEEK_PATTERN = re.compile(r"next\s+week|7\s+days?", re.I)


def _normalize_server_id(raw: str) -> str:
    raw = raw.strip()
    match = re.match(r"^(?:App|app|server|Server|host|Host)[-_ ]?(\d+)$", raw)
    if match:
        number = int(match.group(1))
        return f"server-{number:03d}"
    return raw


def _extract_server_ids(question: str) -> list[str]:
    servers: list[str] = []
    for match in SERVER_PATTERN.finditer(question):
        number = int(match.group(1))
        servers.append(f"server-{number:03d}")
        servers.append(f"App-{number}")
    return list(dict.fromkeys(servers))


def _extract_threshold(question: str, default: float = 90.0) -> float:
    for match in THRESHOLD_PATTERN.finditer(question):
        value = match.group(1) or match.group(2)
        if value:
            return float(value)
    if "high cpu" in question.lower():
        return 80.0
    return default


def _extract_horizon_days(question: str, default: int = 7) -> int:
    week_match = WEEK_PATTERN.search(question)
    if week_match:
        return 7
    day_match = HORIZON_PATTERN.search(question)
    if day_match:
        return int(day_match.group(1))
    return default


def _is_capacity_question(question: str) -> bool:
    lowered = question.lower()
    keywords = (
        "exceed",
        "above",
        "over",
        "high cpu",
        "capacity",
        "which servers",
        "what servers",
        "servers will",
        "next week",
        "overutilized",
        "over-utilized",
        "over utilized",
        "underutilized",
        "under-utilized",
        "under utilized",
        "utilization",
        "list of servers",
        "list servers",
        "top servers",
        "recommendation",
    )
    return any(keyword in lowered for keyword in keywords)


class PredictionStore:
    def __init__(self, artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.frame = self._load_predictions()

    def _load_predictions(self) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for path in sorted(self.artifacts_dir.rglob("predictions*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["model"] = path.parent.name or path.stem
            server_col = next(
                (col for col in ("server_id", "server", "host") if col in df.columns),
                None,
            )
            if server_col is None:
                continue
            df["server_id"] = df[server_col].astype(str)
            date_col = next(
                (col for col in ("timestamp", "date") if col in df.columns),
                None,
            )
            if date_col is None:
                continue
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            pred_col = next(
                (
                    col
                    for col in (
                        "predicted_cpu_utilization",
                        "prediction",
                        "predicted",
                    )
                    if col in df.columns
                ),
                None,
            )
            if pred_col is None:
                continue
            df["prediction"] = pd.to_numeric(df[pred_col], errors="coerce")
            rows.append(df[["server_id", "date", "prediction", "model"]])
        if not rows:
            return pd.DataFrame(columns=["server_id", "date", "prediction", "model"])
        frame = pd.concat(rows, ignore_index=True)
        frame = frame.dropna(subset=["date", "prediction"])
        return frame.sort_values(["server_id", "date", "model"]).reset_index(drop=True)

    @property
    def available(self) -> bool:
        return not self.frame.empty

    def _match_server(self, server_id: str) -> pd.DataFrame:
        if self.frame.empty:
            return self.frame
        aliases = {_normalize_server_id(server_id), server_id}
        if server_id.startswith("App-"):
            aliases.add(f"server-{int(server_id.split('-')[1]):03d}")
        if server_id.startswith("server-"):
            aliases.add(f"App-{int(server_id.split('-')[1])}")
        return self.frame[self.frame["server_id"].isin(aliases)]

    def server_forecast(
        self,
        server_id: str,
        horizon_days: int = 7,
        model: str | None = None,
    ) -> pd.DataFrame:
        subset = self._match_server(server_id)
        if subset.empty:
            return subset
        if model:
            subset = subset[subset["model"] == model]
        if subset.empty:
            return subset
        start_date = subset["date"].min()
        end_date = start_date + timedelta(days=horizon_days - 1)
        return subset[(subset["date"] >= start_date) & (subset["date"] <= end_date)].copy()

    def peak_prediction(
        self,
        server_id: str,
        horizon_days: int = 7,
        model: str | None = "xgboost",
    ) -> dict[str, Any] | None:
        forecast = self.server_forecast(server_id, horizon_days=horizon_days, model=model)
        if forecast.empty and model:
            forecast = self.server_forecast(server_id, horizon_days=horizon_days)
        if forecast.empty:
            return None
        peak_row = forecast.loc[forecast["prediction"].idxmax()]
        return {
            "server_id": str(peak_row["server_id"]),
            "horizon": horizon_days,
            "prediction": round(float(peak_row["prediction"]), 2),
            "peak_date": peak_row["date"].strftime("%Y-%m-%d"),
            "model": str(peak_row["model"]),
            "forecast_rows": len(forecast),
        }

    def servers_above_threshold(
        self,
        threshold: float = 90.0,
        horizon_days: int = 7,
        model: str | None = "xgboost",
    ) -> list[dict[str, Any]]:
        if self.frame.empty:
            return []
        subset = self.frame.copy()
        if model:
            model_subset = subset[subset["model"] == model]
            if not model_subset.empty:
                subset = model_subset
        start_date = subset["date"].min()
        end_date = start_date + timedelta(days=horizon_days - 1)
        window = subset[(subset["date"] >= start_date) & (subset["date"] <= end_date)]
        grouped = (
            window.groupby("server_id", as_index=False)
            .agg(
                peak_prediction=("prediction", "max"),
                peak_date=("date", "max"),
                model=("model", "first"),
            )
            .sort_values("peak_prediction", ascending=False)
        )
        hits = grouped[grouped["peak_prediction"] >= threshold]
        return [
            {
                "server_id": row["server_id"],
                "peak_prediction": round(float(row["peak_prediction"]), 2),
                "peak_date": pd.Timestamp(row["peak_date"]).strftime("%Y-%m-%d"),
                "model": row["model"],
            }
            for _, row in hits.iterrows()
        ]

    def resolve_context(
        self,
        question: str,
        ml_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if ml_output:
            return {"ml_output": ml_output, "insights": None}

        if not self.available:
            return {"ml_output": None, "insights": None}

        horizon_days = _extract_horizon_days(question)
        threshold = _extract_threshold(question)
        servers = _extract_server_ids(question)

        if servers:
            for server_id in servers:
                summary = self.peak_prediction(server_id, horizon_days=horizon_days)
                if summary:
                    return {"ml_output": summary, "insights": None}

        if _is_capacity_question(question):
            hits = self.servers_above_threshold(threshold=threshold, horizon_days=horizon_days)
            top_servers = (
                self.frame.groupby("server_id", as_index=False)["prediction"]
                .max()
                .sort_values("prediction", ascending=False)
                .head(5)
            )
            return {
                "ml_output": None,
                "insights": {
                    "type": "capacity_scan",
                    "threshold": threshold,
                    "horizon_days": horizon_days,
                    "matches": hits,
                    "top_servers": [
                        {
                            "server_id": row["server_id"],
                            "peak_prediction": round(float(row["prediction"]), 2),
                        }
                        for _, row in top_servers.iterrows()
                    ],
                    "date_range": {
                        "start": self.frame["date"].min().strftime("%Y-%m-%d"),
                        "end": self.frame["date"].max().strftime("%Y-%m-%d"),
                    },
                },
            }

        return {"ml_output": None, "insights": None}


def format_prediction_answer(
    question: str,
    ml_output: dict[str, Any] | None,
    insights: dict[str, Any] | None,
    analysis_mode: str,
) -> str | None:
    if insights and insights.get("type") == "capacity_scan":
        threshold = insights["threshold"]
        horizon = insights["horizon_days"]
        matches = insights["matches"]
        if matches:
            lines = [
                f"Based on indexed ML forecasts, these servers are predicted to exceed {threshold:.0f}% CPU within the next {horizon} day(s):"
            ]
            for item in matches:
                lines.append(
                    f"- {item['server_id']}: peak {item['peak_prediction']:.1f}% on {item['peak_date']} ({item['model']})"
                )
            lines.append("Recommended next step: review capacity for the listed servers and prepare scaling or workload throttling.")
            return "\n".join(lines)

        top_servers = insights.get("top_servers") or []
        date_range = insights.get("date_range") or {}
        lines = [
            f"No servers in the indexed ML forecasts exceed {threshold:.0f}% CPU over the next {horizon} day(s).",
        ]
        if date_range:
            lines.append(
                f"Forecast window: {date_range.get('start', 'unknown')} to {date_range.get('end', 'unknown')}."
            )
        if top_servers:
            lines.append("Highest predicted peaks:")
            for item in top_servers:
                lines.append(f"- {item['server_id']}: {item['peak_prediction']:.1f}%")
        lines.append("Recommended next step: continue monitoring; no immediate capacity action is indicated by the current forecasts.")
        return "\n".join(lines)

    if ml_output and ml_output.get("prediction") is not None:
        server_id = ml_output.get("server_id", "unknown")
        prediction = float(ml_output["prediction"])
        horizon = ml_output.get("horizon", 7)
        peak_date = ml_output.get("peak_date", "unknown")
        model = ml_output.get("model", "forecast model")
        if prediction >= 90:
            risk = "critical"
            action = "Treat this as a capacity risk and plan scaling or mitigation before the peak date."
        elif prediction >= 80:
            risk = "high"
            action = "Monitor closely and validate whether upcoming jobs or traffic explain the trend."
        elif prediction >= 70:
            risk = "medium"
            action = "Watch utilization trends and confirm there are no hidden workload spikes."
        else:
            risk = "low"
            action = "No immediate action is required based on the current forecast."
        return (
            f"CPU forecast analysis for {server_id} ({analysis_mode} mode):\n"
            f"- Peak predicted CPU: {prediction:.1f}% within the next {horizon} day(s)\n"
            f"- Peak date: {peak_date}\n"
            f"- Source model: {model}\n"
            f"- Risk level: {risk}\n"
            f"- Recommendation: {action}"
        )

    return None
