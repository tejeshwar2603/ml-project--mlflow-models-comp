#!/usr/bin/env python
"""
Standalone UI Server - reverse proxy in front of the API server's /ui dashboard.

The real UI lives in src/forecasting/static/chatbot_ui.html and is served
directly by the API server at API_SERVER/ui. This process just proxies every
request through so the UI is also reachable on its own port (useful when the
API server isn't meant to be exposed directly).
"""
import os
import logging

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIOps Platform UI Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_SERVER = os.getenv("API_SERVER_URL", "http://localhost:8001")
UI_PORT = int(os.getenv("UI_PORT", "8002"))
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")

logger.info(f"UI proxy configured to forward to API at: {API_SERVER}")


async def _proxy(method: str, path: str, request: Request) -> Response:
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            upstream = await client.request(
                method,
                f"{API_SERVER}/{path}",
                params=request.query_params,
                content=await request.body(),
                headers={"Content-Type": request.headers.get("content-type", "application/json")},
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to API server at {API_SERVER}. Make sure it's running.",
            )
        content_type = upstream.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(content=upstream.json(), status_code=upstream.status_code)
        return HTMLResponse(content=upstream.text, status_code=upstream.status_code)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return await _proxy("GET", "ui", request)


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_all(full_path: str, request: Request):
    return await _proxy(request.method, full_path, request)


def main():
    logger.info(f"Starting AIOps Platform UI proxy on {UI_HOST}:{UI_PORT}")
    logger.info(f"Backend API server: {API_SERVER}")
    logger.info(f"Open browser at: http://localhost:{UI_PORT}")
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="info")


if __name__ == "__main__":
    main()
