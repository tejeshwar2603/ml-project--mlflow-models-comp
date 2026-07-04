import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, order=(5, 1, 0)):
        super().__init__()
        self.order = order

    def _fit_single(self, series):
        model = ARIMA(series, order=self.order)
        return model.fit()

    def _predict_single(self, fitted_model, n_points, forecast_horizon):
        return fitted_model.forecast(steps=n_points)


class SARIMAForecaster(ServerSeriesForecaster):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order

    def _fit_single(self, series):
        model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)

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
