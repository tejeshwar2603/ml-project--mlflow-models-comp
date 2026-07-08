#!/usr/bin/env python
"""Simulated per-server telemetry producer.

There's no real fleet to scrape metrics from in this project, so this
generates plausible CPU/RAM/disk/network readings per server on an interval
and publishes them to Kafka. It's a stand-in for whatever real agent
(node_exporter, CloudWatch, Azure Monitor, etc.) would produce this data in
production - the point is to demonstrate the streaming ingestion shape, not
to be a realistic load generator.

Usage:
    python -m src.streaming.producer [--servers 5] [--interval 2] [--once]
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from kafka import KafkaProducer

TOPIC = "server-metrics"
DEFAULT_BROKER = "localhost:9095"


def make_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )


def generate_reading(server_id: str, tick: int, rng: np.random.Generator) -> dict:
    cpu = float(np.clip(35 + 25 * np.sin(2 * np.pi * tick / 24) + rng.normal(0, 6), 0, 100))
    ram = float(np.clip(cpu * 0.65 + rng.normal(0, 6), 0, 100))
    disk = float(np.clip(25 + rng.normal(0, 4), 0, 100))
    net = float(np.clip(12 + rng.normal(0, 3), 0, 100))
    return {
        "server_id": server_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_utilization": round(cpu, 2),
        "ram_utilization": round(ram, 2),
        "disk_utilization": round(disk, 2),
        "network_utilization": round(net, 2),
    }


def run(broker: str, n_servers: int, interval: float, once: bool) -> None:
    producer = make_producer(broker)
    servers = [f"stream-srv-{i:02d}" for i in range(1, n_servers + 1)]
    rng = np.random.default_rng()
    tick = 0
    print(f"Producing to topic '{TOPIC}' on {broker} for {n_servers} servers every {interval}s. Ctrl+C to stop.")
    try:
        while True:
            for server_id in servers:
                reading = generate_reading(server_id, tick, rng)
                producer.send(TOPIC, key=server_id, value=reading)
            producer.flush()
            print(f"[tick {tick}] sent {n_servers} readings")
            tick += 1
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--servers", type=int, default=5)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between ticks")
    parser.add_argument("--once", action="store_true", help="send a single tick and exit")
    args = parser.parse_args()
    run(args.broker, args.servers, args.interval, args.once)
