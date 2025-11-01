"""Routes exposing auditing data for transparency dashboards."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Query

from src import auditing


router = APIRouter()


@router.get("/")
async def recent_audit_events(limit: int = Query(default=50, le=200)) -> List[dict[str, object]]:
    """Return recent audit events as dictionaries for UI consumption."""

    df = auditing.db_connector.fetch_audit_trail(limit=limit)
    return df.to_dict(orient="records")

