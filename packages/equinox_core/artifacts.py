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

    def ref(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "role": self.role,
            "ordinal": 0,
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
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(
        self,
        data: bytes,
        *,
        role: str,
        media_type: str,
        visibility: str = "OPERATOR",
        trust_class: str = "TRUSTED",
        viewer_hint: str | None = None,
    ) -> StoredArtifact:
        digest = content_digest(data)
        object_key = f"sha256/{digest.removeprefix('sha256:')}"
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=data,
            ContentType=media_type,
            Metadata={"sha256": digest.removeprefix("sha256:")},
        )
        return StoredArtifact(
            artifact_id=f"art_{digest.removeprefix('sha256:')[:32]}",
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
        visibility: str = "OPERATOR",
        trust_class: str = "TRUSTED",
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
