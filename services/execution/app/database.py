from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg
from equinox_core.migrations import migrate_directory
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
        migrate_directory(conn, migrations_dir, authority="operational")


def rows(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with connection() as conn:
        return list(conn.execute(query, params).fetchall())


def row(query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(query, params).fetchone()
