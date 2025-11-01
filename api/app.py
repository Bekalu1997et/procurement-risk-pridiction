"""FastAPI application exposing prediction, explanation, and auditing routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import alerts, audit, explain, predict


app = FastAPI(title="Supplier Risk Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(explain.router, prefix="/explain", tags=["explain"])
app.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
app.include_router(audit.router, prefix="/audit", tags=["audit"])


@app.get("/health", tags=["health"])
async def healthcheck() -> dict[str, str]:
    """Quick health endpoint enabling readiness probes."""

    return {"status": "ok"}

