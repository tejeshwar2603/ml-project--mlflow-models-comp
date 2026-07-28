import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def check_stationarity(series, alpha: float = 0.05) -> tuple[bool, float]:
    """Augmented Dickey-Fuller test. H0: series has a unit root (non-stationary).
    p < alpha => reject H0 => stationary. Too few points for a meaningful ADF
    test defaults to "assume stationary" (d=0) rather than guessing d=1."""
    series = np.asarray(series, dtype=float)
    if len(series) < 8:
        return True, 1.0
    try:
        _, p_value = adfuller(series, autolag="AIC")[:2]
        return p_value < alpha, p_value
    except Exception:
        return True, 1.0


def detect_weekly_seasonality(series, period: int = 7) -> tuple[bool, float]:
    """ACF at the weekly lag, compared against the standard 1.96/sqrt(n)
    significance band. Needs at least 2 full periods to mean anything."""
    series = np.asarray(series, dtype=float)
    if len(series) < 2 * period:
        return False, 0.0
    try:
        nlags = min(21, len(series) // 2 - 1)
        if nlags <= period:
            return False, 0.0
        acf_values = acf(series, nlags=nlags, fft=True)
        acf_at_period = float(acf_values[period])
        threshold = 1.96 / np.sqrt(len(series))
        return abs(acf_at_period) > threshold, acf_at_period
    except Exception:
        return False, 0.0


def determine_arima_order(series, default_pq: tuple[int, int] = (1, 1)) -> tuple[int, int, int]:
    """d from the ADF test; p/q kept small and fixed - with the short, noisy
    per-server series this project deals with, searching p/q (e.g. auto_arima)
    overfits far more often than it helps."""
    is_stationary, _ = check_stationarity(series)
    p, q = default_pq
    return (p, 0 if is_stationary else 1, q)


def determine_sarima_order(series, period: int = 7) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """ARIMA order from ADF, seasonal order enabled only if ACF actually shows a
    weekly pattern with enough history to fit it - otherwise SARIMA degrades to
    plain ARIMA instead of fitting a seasonal component to noise."""
    order = determine_arima_order(series)
    has_seasonality, _ = detect_weekly_seasonality(series, period)
    if has_seasonality and len(series) >= 3 * period:
        seasonal_order = (1, 0, 1, period)
    else:
        seasonal_order = (0, 0, 0, 0)
    return order, seasonal_order


class BaseForecaster:
    def fit(self, X, y, **kwargs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def name(self):
        return self.__class__.__name__


class ServerSeriesForecaster(BaseForecaster):
    def __init__(self):
        self.models = {}

    def fit(self, df, target_col="cpu_utilization", server_col="server_id", time_col="timestamp"):
        self.target_col = target_col
        self.server_col = server_col
        self.time_col = time_col
        for server, group in df.groupby(server_col):
            series = group.sort_values(time_col)[target_col].astype(float)
            self.models[server] = self._fit_single(series)
        return self

    def predict(self, df, forecast_horizon=1):
        preds = []
        for server, group in df.groupby(self.server_col):
            model = self.models.get(server)
            if model is None:
                preds.extend([np.nan] * len(group))
                continue
            values = self._predict_single(model, len(group), forecast_horizon)
            preds.extend(values)
        return np.array(preds)

    def _fit_single(self, series):
        raise NotImplementedError()

    def _predict_single(self, model, n_points, forecast_horizon):
        raise NotImplementedError()


class ARIMAForecaster(ServerSeriesForecaster):
    def __init__(self, order=None):
        super().__init__()
        # None = determine (p,d,q) per-series from an ADF stationarity test
        # instead of assuming one fixed order fits every server's series.
        self.order = order

    def _fit_single(self, series):
        order = self.order or determine_arima_order(series)
        try:
            return ARIMA(series, order=order).fit()
        except Exception:
            return ARIMA(series, order=(1, 0, 0)).fit()

    def _predict_single(self, fitted_model, n_points, forecast_horizon):
        return fitted_model.forecast(steps=n_points)


class SARIMAForecaster(ServerSeriesForecaster):
    def __init__(self, order=None, seasonal_order=None):
        super().__init__()
        # None = determine both orders per-series from ADF (trend) + ACF (weekly
        # seasonality) instead of forcing the same seasonal component onto every
        # server regardless of whether its series actually shows one.
        self.order = order
        self.seasonal_order = seasonal_order

    def _fit_single(self, series):
        if self.order is not None and self.seasonal_order is not None:
            order, seasonal_order = self.order, self.seasonal_order
        else:
            order, seasonal_order = determine_sarima_order(series)
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False)
        except Exception:
            model = ARIMA(series, order=order)
            return model.fit()

    def _predict_single(self, fitted_model, n_points, forecast_horizon):
        return fitted_model.forecast(steps=n_points)


class XGBoostForecaster(BaseForecaster, RegressorMixin):
    def __init__(self, params=None, num_round=100):
        self.params = params or {"objective": "reg:squarederror", "tree_method": "auto", "verbosity": 0}
        self.num_round = num_round
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = xgb.XGBRegressor(**self.params, n_estimators=self.num_round)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=14):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


class TorchForecaster(BaseForecaster):
    def __init__(self, seq_len=14, hidden_size=64, lr=1e-3, epochs=10, batch_size=32):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None

    def _build_model(self, input_dim):
        raise NotImplementedError()

    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X.shape[1]
        dataset = SequenceDataset(X_scaled, y, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = self._build_model(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            for seq, target in loader:
                optimizer.zero_grad()
                out = self.model(seq)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            seq_inputs = []
            for i in range(max(0, len(X_scaled) - self.seq_len + 1)):
                seq_inputs.append(X_scaled[i : i + self.seq_len])
            if not seq_inputs:
                return np.zeros(len(X_scaled))
            seq_tensor = torch.tensor(np.stack(seq_inputs), dtype=torch.float32)
            preds = self.model(seq_tensor).squeeze().numpy()
            return np.concatenate([np.zeros(self.seq_len - 1), preds])


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_size, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out[-1]
        return self.fc(out)


class GRUForecaster(TorchForecaster):
    def _build_model(self, input_dim):
        return GRUModel(input_dim, self.hidden_size)


class LSTMForecaster(TorchForecaster):
    def _build_model(self, input_dim):
        return LSTMModel(input_dim, self.hidden_size)


class TFTForecaster(TorchForecaster):
    def _build_model(self, input_dim):
        return TransformerModel(input_dim, self.hidden_size)
