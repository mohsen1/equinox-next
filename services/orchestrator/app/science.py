from __future__ import annotations

from typing import Any

from equinox_core import ArtifactStore, canonical_digest, make_id, utc_now
from psycopg import Connection
from psycopg.types.json import Jsonb

artifact_store = ArtifactStore()


def record_artifact(
    conn: Connection[dict[str, Any]],
    artifact: dict[str, Any],
    *,
    entity_type: str,
    entity_id: str,
    role: str | None = None,
    ordinal: int = 0,
) -> str:
    artifact_id = artifact["artifact_id"]
    conn.execute(
        """
        INSERT INTO artifacts(artifact_id, digest, object_key, media_type, size_bytes)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (digest) DO NOTHING
        """,
        (
            artifact_id,
            artifact["digest"],
            artifact["object_key"],
            artifact["media_type"],
            artifact["size_bytes"],
        ),
    )
    canonical = conn.execute(
        "SELECT artifact_id FROM artifacts WHERE digest = %s",
        (artifact["digest"],),
    ).fetchone()
    actual_id = canonical["artifact_id"]
    conn.execute(
        """
        INSERT INTO artifact_refs(
          artifact_ref_id, artifact_id, entity_type, entity_id, role, ordinal,
          viewer_hint, visibility, trust_class
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (entity_type, entity_id, role, ordinal) DO NOTHING
        """,
        (
            make_id("artifact_ref"),
            actual_id,
            entity_type,
            entity_id,
            role or artifact["role"],
            ordinal,
            artifact.get("viewer_hint"),
            artifact["visibility"],
            artifact["trust_class"],
        ),
    )
    return actual_id


def store_json_artifact(
    conn: Connection[dict[str, Any]],
    value: Any,
    *,
    role: str,
    entity_type: str,
    entity_id: str,
    visibility: str = "OPERATOR",
    trust_class: str = "TRUSTED",
    viewer_hint: str | None = "structured-json",
    ordinal: int = 0,
) -> tuple[str, dict[str, Any]]:
    stored = artifact_store.put_json(
        value,
        role=role,
        visibility=visibility,
        trust_class=trust_class,
        viewer_hint=viewer_hint,
    )
    payload = {
        **stored.ref(),
        "object_key": stored.object_key,
        "size_bytes": stored.size_bytes,
    }
    artifact_id = record_artifact(
        conn,
        payload,
        entity_type=entity_type,
        entity_id=entity_id,
        role=role,
        ordinal=ordinal,
    )
    return artifact_id, payload


def emit_event(
    conn: Connection[dict[str, Any]],
    *,
    event_type: str,
    aggregate_type: str,
    aggregate_id: str,
    run_id: str,
    correlation_id: str,
    payload: dict[str, Any],
    causation_id: str | None = None,
    producer: str = "orchestrator",
) -> str:
    conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"{aggregate_type}:{aggregate_id}",),
    )
    sequence = conn.execute(
        """
        SELECT COALESCE(max(aggregate_sequence), 0) + 1 AS next
        FROM events WHERE aggregate_type = %s AND aggregate_id = %s
        """,
        (aggregate_type, aggregate_id),
    ).fetchone()["next"]
    event_id = make_id("event")
    occurred_at = utc_now()
    envelope = {
        "event_id": event_id,
        "event_type": event_type,
        "schema_version": 1,
        "aggregate_type": aggregate_type,
        "aggregate_id": aggregate_id,
        "aggregate_sequence": sequence,
        "run_id": run_id,
        "correlation_id": correlation_id,
        "causation_id": causation_id,
        "producer": producer,
        "occurred_at": occurred_at,
        "payload": payload,
    }
    conn.execute(
        """
        INSERT INTO events(
          event_id, event_type, schema_version, aggregate_type, aggregate_id,
          aggregate_sequence, run_id, correlation_id, causation_id, producer,
          payload, occurred_at
        ) VALUES (%s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event_id,
            event_type,
            aggregate_type,
            aggregate_id,
            sequence,
            run_id,
            correlation_id,
            causation_id,
            producer,
            Jsonb(payload),
            occurred_at,
        ),
    )
    conn.execute(
        "INSERT INTO outbox(event_id, topic, payload) VALUES (%s, %s, %s)",
        (event_id, event_type, Jsonb(envelope)),
    )
    return event_id


def record_operation_intent(
    conn: Connection[dict[str, Any]],
    *,
    operation_id: str,
    run_id: str,
    operation_type: str,
    operation_input: dict[str, Any],
    expected_version: int,
    budget_reservation_id: str | None = None,
) -> str:
    request_digest = canonical_digest({"operation_type": operation_type, "input": operation_input})
    existing = conn.execute(
        "SELECT request_digest FROM operations WHERE idempotency_key = %s",
        (operation_id,),
    ).fetchone()
    if existing and existing["request_digest"] != request_digest:
        raise ValueError("idempotency key reused with a different request digest")
    conn.execute(
        """
        INSERT INTO operations(
          operation_id, run_id, idempotency_key, request_digest, operation_type,
          operation_input, expected_version, status, budget_reservation_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'AUTHORIZED', %s)
        ON CONFLICT (operation_id) DO NOTHING
        """,
        (
            operation_id,
            run_id,
            operation_id,
            request_digest,
            operation_type,
            Jsonb(operation_input),
            expected_version,
            budget_reservation_id,
        ),
    )
    return request_digest


def accept_operation_result(
    conn: Connection[dict[str, Any]],
    *,
    operation_id: str,
    result: dict[str, Any],
) -> None:
    result_digest = canonical_digest(result)
    operation = conn.execute(
        "SELECT status, result_digest FROM operations WHERE operation_id = %s FOR UPDATE",
        (operation_id,),
    ).fetchone()
    if not operation:
        raise ValueError("operation result has no authorized intent")
    if operation["status"] == "ACCEPTED":
        if operation["result_digest"] != result_digest:
            raise ValueError("accepted operation received a conflicting result")
        return
    conn.execute(
        """
        UPDATE operations SET status = 'ACCEPTED', result_digest = %s, result = %s,
          updated_at = now() WHERE operation_id = %s
        """,
        (result_digest, Jsonb(result), operation_id),
    )
