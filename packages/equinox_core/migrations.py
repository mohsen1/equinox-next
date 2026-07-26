from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from psycopg import Connection


class MigrationStateError(RuntimeError):
    pass


def migrate_directory(
    conn: Connection[dict[str, Any]],
    migrations_dir: Path,
    *,
    authority: str,
) -> None:
    conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (f"migrations:{authority}",))
    conn.commit()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version integer PRIMARY KEY,
              filename text,
              checksum text,
              dirty boolean NOT NULL DEFAULT false,
              duration_ms integer,
              applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS filename text")
        conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum text")
        conn.execute(
            "ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS dirty boolean "
            "NOT NULL DEFAULT false"
        )
        conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS duration_ms integer")
        conn.commit()

        paths = sorted(migrations_dir.glob("*.sql"))
        expected = {
            int(path.stem.split("_", 1)[0]): (
                path,
                f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            )
            for path in paths
        }
        applied = {
            row["version"]: row
            for row in conn.execute(
                "SELECT version, filename, checksum, dirty FROM schema_migrations"
            )
        }
        dirty = [str(version) for version, row in applied.items() if row["dirty"]]
        if dirty:
            raise MigrationStateError(
                f"{authority} schema has dirty migration(s): {', '.join(sorted(dirty))}"
            )
        for version, row in applied.items():
            if version not in expected:
                raise MigrationStateError(
                    f"{authority} schema records unknown migration version {version}"
                )
            path, checksum = expected[version]
            if row["checksum"] is None:
                conn.execute(
                    """
                    UPDATE schema_migrations
                    SET filename = %s, checksum = %s
                    WHERE version = %s
                    """,
                    (path.name, checksum, version),
                )
            elif row["checksum"] != checksum or row["filename"] != path.name:
                raise MigrationStateError(
                    f"{authority} migration {version} checksum or filename changed"
                )
        conn.commit()

        for version, (path, checksum) in expected.items():
            if version in applied:
                continue
            conn.execute(
                """
                INSERT INTO schema_migrations(version, filename, checksum, dirty)
                VALUES (%s, %s, %s, true)
                """,
                (version, path.name, checksum),
            )
            conn.commit()
            started = time.monotonic()
            try:
                conn.execute(path.read_text(encoding="utf-8"))
                duration_ms = round((time.monotonic() - started) * 1000)
                conn.execute(
                    """
                    UPDATE schema_migrations
                    SET dirty = false, duration_ms = %s, applied_at = now()
                    WHERE version = %s
                    """,
                    (duration_ms, version),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    finally:
        conn.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
            (f"migrations:{authority}",),
        )
        conn.commit()
