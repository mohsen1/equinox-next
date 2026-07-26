from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .canonical import canonical_bytes, content_digest


@dataclass(frozen=True)
class StoredArtifact:
    artifact_id: str
    digest: str
    role: str
    media_type: str
    size_bytes: int
    object_key: str
    visibility: str
    trust_class: str
    viewer_hint: str | None

    def ref(self, *, ordinal: int) -> dict[str, Any]:
        if ordinal < 0:
            raise ValueError("artifact ordinal must be non-negative")
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "role": self.role,
            "ordinal": ordinal,
            "media_type": self.media_type,
            "viewer_hint": self.viewer_hint,
            "visibility": self.visibility,
            "trust_class": self.trust_class,
        }


class ArtifactStore:
    def __init__(self) -> None:
        self.bucket = os.getenv("S3_BUCKET", "equinox-artifacts")
        endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "equinox"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "equinox-local-only"),
            region_name="us-east-1",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            if os.getenv("S3_BOOTSTRAP_BUCKET", "true").lower() not in {"1", "true", "yes"}:
                raise RuntimeError(f"artifact bucket {self.bucket!r} does not exist") from exc
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(
        self,
        data: bytes,
        *,
        role: str,
        media_type: str,
        trust_class: str,
        visibility: str = "OPERATOR",
        viewer_hint: str | None = None,
    ) -> StoredArtifact:
        digest = content_digest(data)
        digest_hex = digest.removeprefix("sha256:")
        object_key = f"sha256/{digest_hex}"
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=data,
                ContentType=media_type,
                Metadata={"sha256": digest_hex},
                IfNoneMatch="*",
            )
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code not in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                raise
        metadata = self.client.head_object(Bucket=self.bucket, Key=object_key)
        stored_digest = metadata.get("Metadata", {}).get("sha256")
        if metadata.get("ContentLength") != len(data) or stored_digest != digest_hex:
            raise RuntimeError(f"immutable artifact metadata mismatch for {digest}")
        return StoredArtifact(
            artifact_id=f"art_sha256_{digest_hex}",
            digest=digest,
            role=role,
            media_type=media_type,
            size_bytes=len(data),
            object_key=object_key,
            visibility=visibility,
            trust_class=trust_class,
            viewer_hint=viewer_hint,
        )

    def put_json(
        self,
        value: Any,
        *,
        role: str,
        trust_class: str,
        visibility: str = "OPERATOR",
        viewer_hint: str | None = "structured-json",
    ) -> StoredArtifact:
        return self.put_bytes(
            canonical_bytes(value),
            role=role,
            media_type="application/json",
            visibility=visibility,
            trust_class=trust_class,
            viewer_hint=viewer_hint,
        )

    def get_bytes(self, digest: str) -> bytes:
        object_key = f"sha256/{digest.removeprefix('sha256:')}"
        response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        data = response["Body"].read()
        if content_digest(data) != digest:
            raise ValueError(f"artifact digest mismatch for {digest}")
        return data
