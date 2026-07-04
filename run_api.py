import os
import src.forecasting.app as app

os.environ.setdefault("FORECAST_MODEL_URI", "models:/cpu_forecast/Production")

if __name__ == "__main__":
    app.main()
