#!/usr/bin/env python
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Set port before importing the app
os.environ["API_PORT"] = os.environ.get("API_PORT", "8001")

from src.forecasting.app import main

if __name__ == "__main__":
    main()
