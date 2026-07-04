import pandas as pd

def _lag_features(group, target_col, lags):
    for lag in lags:
        group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
    return group


def _rolling_features(group, target_col, windows):
    for window in windows:
        group[f"{target_col}_roll_mean_{window}"] = group[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        group[f"{target_col}_roll_std_{window}"] = group[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
        group[f"{target_col}_roll_min_{window}"] = group[target_col].shift(1).rolling(window=window, min_periods=1).min()
        group[f"{target_col}_roll_max_{window}"] = group[target_col].shift(1).rolling(window=window, min_periods=1).max()
    return group


def _calendar_features(df):
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def _ratio_features(df):
    df["cpu_ram_ratio"] = df["cpu_utilization"] / df["ram_utilization"].replace(0, 1)
    df["cpu_disk_ratio"] = df["cpu_utilization"] / df["disk_utilization"].replace(0, 1)
    return df


def build_features(df: pd.DataFrame):
    df = df.copy()
    df = validate_and_fill(df)
    df = _calendar_features(df)
    lag_cols = [1, 3, 7, 14, 30]
    roll_windows = [3, 7, 14, 30]
    df = df.groupby("server_id", group_keys=False).apply(lambda g: _lag_features(g, "cpu_utilization", lag_cols))
    df = df.groupby("server_id", group_keys=False).apply(lambda g: _rolling_features(g, "cpu_utilization", roll_windows))
    df = _ratio_features(df)
    df = df.dropna(subset=[f"cpu_utilization_lag_{lag}" for lag in lag_cols])
    return df.reset_index(drop=True)


def validate_and_fill(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(["server_id", "timestamp"])
    df["cpu_utilization"] = df.groupby("server_id")["cpu_utilization"].transform(lambda x: x.interpolate().ffill().bfill())
    for col in ["ram_utilization", "disk_utilization", "network_utilization"]:
        df[col] = df.groupby("server_id")[col].transform(lambda x: x.interpolate().ffill().bfill())
    return df
