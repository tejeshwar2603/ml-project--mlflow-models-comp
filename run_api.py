import os
from dotenv import load_dotenv
import src.forecasting.app as app

# Load environment variables from .env for local development (optional).
# In production use your platform's secret store or CI/CD secrets instead.
load_dotenv()

os.environ.setdefault("FORECAST_MODEL_URI", "models:/cpu_forecast/Production")

if __name__ == "__main__":
    app.main()
