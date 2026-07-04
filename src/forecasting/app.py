import os
import uvicorn
from .api import create_app


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    model_uri = os.getenv("FORECAST_MODEL_URI", "models:/cpu_forecast/Production")
    app = create_app(model_uri=model_uri)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
