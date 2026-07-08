#!/usr/bin/env python
"""Kafka consumer that turns the streamed telemetry into a dataset the app can forecast against.

Reads from the 'server-metrics' topic, accumulates readings in memory, and
periodically upserts them into the running API as dataset_id="kafka-live" via
POST /datasets/upload - reusing the exact same upload/validation path a human
uploading an Excel file goes through (see api.py, forecast_service.py).

Caveat worth knowing: this project's feature engineering (features.py) treats
each row as one calendar day (lag_7 = "a week ago", day_of_week, etc). This
consumer ingests readings at whatever cadence the producer sends them (every
few seconds), so each row here is one tick, not one day - lag/rolling
features built from this data mean "N ticks ago", not "N days ago". That's
fine for demonstrating the streaming plumbing end-to-end; it isn't a
substitute for real daily telemetry.

Usage:
    python -m src.streaming.consumer [--flush-every 20] [--api http://localhost:8001]
"""
import argparse
import io
import time

import pandas as pd
import requests
from kafka import KafkaConsumer

from .producer import TOPIC, DEFAULT_BROKER

DATASET_ID = "kafka-live"


def run(broker: str, api_base: str, flush_every: int) -> None:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=broker,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: __import__("json").loads(v.decode("utf-8")),
        consumer_timeout_ms=-1,
    )
    buffer: list[dict] = []
    print(f"Consuming topic '{TOPIC}' on {broker}; upserting dataset '{DATASET_ID}' to {api_base} every {flush_every} messages.")
    try:
        for message in consumer:
            buffer.append(message.value)
            if len(buffer) >= flush_every:
                _flush(buffer, api_base)
    except KeyboardInterrupt:
        print("Stopping; flushing remaining buffer...")
        if buffer:
            _flush(buffer, api_base)


def _flush(buffer: list[dict], api_base: str) -> None:
    df = pd.DataFrame(buffer)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    try:
        resp = requests.post(
            f"{api_base}/datasets/upload",
            files={"file": ("kafka_stream.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"dataset_label": "Live Kafka stream", "dataset_id": DATASET_ID},
            timeout=30,
        )
        resp.raise_for_status()
        summary = resp.json()
        print(f"Upserted '{DATASET_ID}': {summary['rows']} rows across {summary['servers']} servers.")
    except Exception as exc:
        print(f"Failed to upsert dataset: {exc}")
    finally:
        buffer.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--api", default="http://localhost:8001")
    parser.add_argument("--flush-every", type=int, default=20, help="messages to buffer before upserting the dataset")
    args = parser.parse_args()
    run(args.broker, args.api, args.flush_every)
