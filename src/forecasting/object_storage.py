"""MinIO (S3-compatible) object-storage ingestion.

A third way datasets enter the system, alongside manual upload (POST
/datasets/upload) and Kafka streaming (src/streaming/consumer.py): drop a
.csv/.xlsx file into the configured MinIO bucket, and MinIO's own bucket
notification fires a webhook at POST /storage/events (see api.py) the moment
the object is created. That handler calls fetch_object() below to pull the
bytes back out via the S3 API, then feeds them through the exact same
ForecastService.read_tabular_bytes() -> register_uploaded_dataset() path a
human upload goes through - MinIO is just a different source for the same
pipeline, not a separate one.
"""
import os

import boto3
from botocore.config import Config


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        # MinIO is path-style only (bucket.endpoint virtual-hosted style doesn't
        # resolve for a local/self-hosted endpoint) - boto3 defaults to virtual-hosted
        # for real AWS, so this must be set explicitly or every request 404s.
        config=Config(s3={"addressing_style": "path"}),
    )


def fetch_object(bucket: str, key: str) -> bytes:
    obj = _client().get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()
