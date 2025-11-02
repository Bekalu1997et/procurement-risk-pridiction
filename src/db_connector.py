"""Database connectivity utilities for the risk prediction demo platform.

The project leverages a PostgreSQL database to simulate an operational data
store (ODS) for supplier profiles, prediction history, and auditing metadata.
The functions exposed here intentionally wrap SQLAlchemy operations in a
pedagogical manner, showing how production systems often abstract persistence
concerns away from business logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


LOGGER = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "risk_demo")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_engine(echo: bool = False) -> Engine:
    """Instantiate (and cache) a SQLAlchemy engine for PostgreSQL operations.
    The function creates a SQLAlchemy engine for PostgreSQL operations.
    It uses the environment variables for the database connection.
    """

    connection_string = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string, echo=echo, future=True)
    return engine


def initialize_database() -> None:
    """Create necessary tables when the database file is first provisioned.
    The function creates the necessary tables for the database.
    It uses the SQLAlchemy metadata to create the tables.
    """

    engine = get_engine()
    metadata = MetaData()

    Table(
        "supplier_profiles",
        metadata,
        Column("supplier_id", Integer, primary_key=True),
        Column("supplier_name", String(255)),
        Column("region", String(64)),
        Column("industry", String(64)),
        Column("credit_score", Integer),
        Column("annual_spend", Float),
        Column("contract_criticality", String(32)),
        Column("updated_at", DateTime(timezone=True)),
    )

    Table(
        "prediction_history",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("supplier_id", Integer),
        Column("model_name", String(128)),
        Column("risk_label", String(32)),
        Column("risk_score", Float),
        Column("explanation", String(1024)),
        Column("created_at", DateTime(timezone=True)),
    )

    Table(
        "audit_events",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("event_type", String(128)),
        Column("payload", JSON),
        Column("created_at", DateTime(timezone=True)),
    )

    metadata.create_all(engine)


def load_supplier_profile(supplier_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a supplier profile from the PostgreSQL database.
    The function retrieves a supplier profile from the PostgreSQL database.
    It uses the SQLAlchemy engine to execute the query.
    """

    engine = get_engine()
    query = text(
        """
        SELECT * FROM supplier_profiles WHERE supplier_id = :supplier_id
        """
    )

    with engine.begin() as conn:
        row = conn.execute(query, {"supplier_id": supplier_id}).mappings().first()
    return dict(row) if row else None


def save_predictions(records: Iterable[Dict[str, Any]]) -> None:
    """Persist prediction events into the history table.
    The function persists the prediction events into the history table.
    It uses the SQLAlchemy engine to execute the insert.
    """

    engine = get_engine()
    metadata = MetaData()
    table = Table("prediction_history", metadata, autoload_with=engine)

    with engine.begin() as conn:
        conn.execute(insert(table), list(records))


def get_historical_scores(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch the most recent prediction history entries for dashboards.
    The function fetches the most recent prediction history entries for dashboards.
    It uses the SQLAlchemy engine to execute the query.
    """

    engine = get_engine()
    metadata = MetaData()
    table = Table("prediction_history", metadata, autoload_with=engine)

    query = select(table).order_by(table.c.created_at.desc()).limit(limit)

    with engine.begin() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def write_audit_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Insert an audit event. Errors are swallowed but logged for resilience.
    The function inserts an audit event into the audit_events table.
    It uses the SQLAlchemy engine to execute the insert.
    """

    engine = get_engine()
    metadata = MetaData()
    table = Table("audit_events", metadata, autoload_with=engine)

    event = {
        "event_type": event_type,
        "payload": payload,
        "created_at": pd.Timestamp.utcnow(),
    }

    try:
        with engine.begin() as conn:
            conn.execute(insert(table), [event])
    except SQLAlchemyError as exc:  # pragma: no cover - external dependency.
        LOGGER.warning("Failed to persist audit event %s: %s", event_type, exc)


def fetch_audit_trail(limit: int = 100) -> pd.DataFrame:
    """Return a pandas dataframe of recent audit events for analysis.
    The function fetches the recent audit events from the audit_events table.
    It uses the SQLAlchemy engine to execute the query.
    """

    engine = get_engine()
    query = text(
        """
        SELECT event_type, payload, created_at
        FROM audit_events
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    return df

try:
    initialize_database()
    print("Database initialized successfully.")
except SQLAlchemyError as exc:  # pragma: no cover - depends on external service.
    LOGGER.error("Failed to initialize database: %s", exc)
    print("Database initialization failed. Check logs for details.")

