"""Routes for retrieving alerting information from the mock database."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Query

from src import db_connector


router = APIRouter()


@router.get("/")
async def list_alerts(limit: int = Query(default=20, le=100)) -> List[dict[str, object]]:
    """Return recent high-risk supplier alerts from the prediction history."""

    records = db_connector.get_historical_scores(limit=limit)
    return records

