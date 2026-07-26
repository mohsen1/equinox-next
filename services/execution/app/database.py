from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://equinox_ops:ops-local-only@postgres:5432/equinox",
)


def connection() -> psycopg.Connection[dict[str, Any]]:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def migrate() -> None:
    migrations_dir = Path(__file__).parents[1] / "migrations"
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version integer PRIMARY KEY,
              applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(migrations_dir.glob("*.sql")):
            version = int(path.stem.split("_", 1)[0])
            if version not in applied:
                conn.execute(path.read_text(encoding="utf-8"))


def rows(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with connection() as conn:
        return list(conn.execute(query, params).fetchall())


def row(query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(query, params).fetchone()
