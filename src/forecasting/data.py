import numpy as np
import pandas as pd
from datetime import timedelta


def generate_synthetic_data(n_servers=50, n_days=180, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    for server in range(1, n_servers + 1):
        cores = int(rng.choice([4, 8, 16, 24]))
        installed_ram = int(rng.choice([32, 64, 128, 256]))
        for day in range(n_days):
            ts = start + pd.Timedelta(days=day)
            cpu = np.clip(20 + 15 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 5), 0, 100)
            ram = np.clip(cpu * 0.6 + rng.normal(0, 8), 0, 100)
            disk = np.clip(30 + 10 * np.cos(2 * np.pi * day / 30) + rng.normal(0, 6), 0, 100)
            net = np.clip(15 + 8 * np.sin(2 * np.pi * day / 14) + rng.normal(0, 3), 0, 100)
            records.append(
                {
                    "server_id": f"server-{server:03d}",
                    "timestamp": ts,
                    "cpu_utilization": float(cpu),
                    "ram_utilization": float(ram),
                    "disk_utilization": float(disk),
                    "network_utilization": float(net),
                    "cpu_cores": cores,
                    "installed_ram": installed_ram,
                }
            )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame):
    required_columns = [
        "server_id",
        "timestamp",
        "cpu_utilization",
        "ram_utilization",
        "disk_utilization",
        "network_utilization",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing data columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["server_id", "timestamp"])
    return df


def load_data(path: str):
    df = pd.read_csv(path)
    return validate_data(df)
