import json
import urllib.request
import urllib.error

body = {
    "server_id": "server-001",
    "horizon": 1,
    "features": {
        "cpu_utilization": 35.2,
        "ram_utilization": 42.1,
        "disk_utilization": 55.0,
        "network_utilization": 23.4,
        "cpu_utilization_lag_1": 34.0,
        "cpu_utilization_lag_3": 30.2,
        "cpu_utilization_lag_7": 25.8,
        "cpu_utilization_lag_14": 22.1,
        "cpu_utilization_lag_30": 20.0,
        "cpu_utilization_roll_mean_3": 33.2,
        "cpu_utilization_roll_std_3": 2.1,
        "cpu_utilization_roll_min_3": 31.0,
        "cpu_utilization_roll_max_3": 35.8,
        "cpu_utilization_roll_mean_7": 30.1,
        "cpu_utilization_roll_std_7": 3.4,
        "cpu_utilization_roll_min_7": 28.0,
        "cpu_utilization_roll_max_7": 33.5,
        "cpu_ram_ratio": 0.83,
        "cpu_disk_ratio": 0.64,
        "day_of_week": 2,
        "month": 7,
        "is_weekend": 0
    }
}

url = 'http://127.0.0.1:8000/predict'
data = json.dumps(body).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        status = resp.getcode()
        text = resp.read().decode('utf-8')
        print('STATUS:', status)
        print('BODY:')
        print(text)
except urllib.error.HTTPError as e:
    print('HTTP ERROR', e.code)
    try:
        print(e.read().decode('utf-8'))
    except Exception:
        pass
except Exception as e:
    print('REQUEST FAILED:', repr(e))
