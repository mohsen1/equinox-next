"""Bounded RunPod launcher for the sealed revision-30 external evaluation."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - macOS operator compatibility.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from research.runpod.bootstrap_server import (
    EXTERNAL_EVALUATION_WORKLOAD as REVISION30_EXTERNAL_EVALUATION_WORKLOAD,
)
from research.runpod.bootstrap_server import (
    MAXIMUM_BUNDLE_BYTES,
    expected_bundle_files,
)
from research.runpod.external_eval_transport import sha256_file
from research.runpod.revision30_external_operator import load_input_manifest_from_value

EVALUATION_REVISION = os.environ.get("EQUINOX_EXTERNAL_EVALUATION_REVISION", "30")
if EVALUATION_REVISION == "31":
    from research.runpod.revision30_external_eval import canonical_json
    from research.runpod.revision31_external_eval import (
        MODEL_ID,
        MODEL_REVISION,
        PACK_ID,
        WORKLOAD,
        WORKLOAD_REVISION,
        validate_evaluation_manifest,
    )

    EXTERNAL_EVALUATION_WORKLOAD = "research/runpod/revision31_external_eval.py"
elif EVALUATION_REVISION == "30":
    from research.runpod.revision30_external_eval import (
        MODEL_ID,
        MODEL_REVISION,
        PACK_ID,
        WORKLOAD,
        WORKLOAD_REVISION,
        canonical_json,
        validate_evaluation_manifest,
    )

    EXTERNAL_EVALUATION_WORKLOAD = REVISION30_EXTERNAL_EVALUATION_WORKLOAD
else:
    raise RuntimeError("EQUINOX_EXTERNAL_EVALUATION_REVISION must be 30 or 31")

PROOF_CONTRACT_ERROR = "The remote result did not satisfy the declared proof contract."
DEFAULT_GPU = "NVIDIA RTX PRO 4500 Blackwell"
DEFAULT_TEMPLATE = "runpod-torch-v280"
DEFAULT_IMAGE_CEILING = 0.75
DEFAULT_MAXIMUM_LIFETIME_MINUTES = 180
DEFAULT_BOOT_TIMEOUT_SECONDS = 360
DEFAULT_INPUT_TIMEOUT_SECONDS = 1_800
DEFAULT_CONTAINER_DISK_GB = 30
DEFAULT_VOLUME_GB = 10
PROXY_PORT = 8000
REMOTE_WORK_DIRECTORY = "/workspace/equinox-state"
REMOTE_INPUT_DIRECTORY = f"{REMOTE_WORK_DIRECTORY}/inputs"


class LaunchFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def environment_integer(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or not minimum <= int(raw) <= maximum:
        raise LaunchFailure(f"{name} must be between {minimum} and {maximum}")
    return int(raw)


def environment_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise LaunchFailure(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise LaunchFailure(f"{name} must be between {minimum} and {maximum}")
    return value


def run_command(
    arguments: list[str],
    *,
    timeout: int = 60,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            arguments,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LaunchFailure(f"command failed to run: {arguments[0]}") from error


def runpod_json(*arguments: str, timeout: int = 60) -> Any:
    completed = run_command(["runpodctl", *arguments], timeout=timeout)
    if completed.returncode != 0:
        raise LaunchFailure(
            completed.stderr.decode(errors="replace").strip()
            or f"runpodctl {' '.join(arguments)} failed"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise LaunchFailure("runpodctl returned invalid JSON") from error


def request_json(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 15,
) -> Any:
    body = canonical_json(payload) if payload is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise LaunchFailure(f"JSON request failed: {url}") from error


def request_health(url: str) -> None:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read(1)
    except (OSError, urllib.error.URLError) as error:
        raise LaunchFailure(f"health request failed: {url}") from error


def proxy_get(
    proxy_root: str,
    path: str,
    token: str,
    *,
    timeout: int = 15,
    optional: bool = False,
) -> bytes | None:
    completed = run_command(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "-H",
            f"Authorization: Bearer {token}",
            f"{proxy_root}{path}",
        ],
        timeout=timeout + 5,
    )
    if completed.returncode != 0:
        if optional:
            return None
        raise LaunchFailure(
            completed.stderr.decode(errors="replace").strip() or f"proxy GET failed: {path}"
        )
    return completed.stdout


def proxy_put_file(
    proxy_root: str,
    path: str,
    token: str,
    source: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    completed = run_command(
        [
            "curl",
            "--silent",
            "--show-error",
            "--http1.1",
            "--max-time",
            str(timeout),
            "-X",
            "POST",
            "-H",
            f"Authorization: Bearer {token}",
            "-H",
            "Content-Type: application/octet-stream",
            "-H",
            "Expect:",
            "--data-binary",
            f"@{source}",
            "--write-out",
            "\n%{http_code}",
            f"{proxy_root}{path}",
        ],
        timeout=timeout + 5,
    )
    if completed.returncode != 0:
        raise LaunchFailure(
            completed.stderr.decode(errors="replace").strip() or f"proxy upload failed: {path}"
        )
    try:
        payload, status_bytes = completed.stdout.rsplit(b"\n", 1)
        status = int(status_bytes)
    except (ValueError, TypeError) as error:
        raise LaunchFailure(f"proxy upload returned no HTTP status: {path}") from error
    if not 200 <= status < 300:
        response = payload.decode(errors="replace").strip()
        raise LaunchFailure(
            f"proxy upload returned HTTP {status}: {path}"
            + (f": {response[:500]}" if response else "")
        )
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchFailure(f"proxy upload returned invalid JSON: {path}") from error


def ssh_endpoint_from_pod(pod: Any) -> tuple[str, int] | None:
    pending = [pod]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            private_port = value.get("privatePort", value.get("private_port"))
            public_port = value.get(
                "publicPort",
                value.get("public_port", value.get("port")),
            )
            host = value.get("ip", value.get("host"))
            is_public = value.get("isIpPublic", value.get("is_ip_public", True))
            cli_ssh_record = isinstance(value.get("ssh_command"), str)
            if (
                (private_port == 22 or cli_ssh_record)
                and is_public is not False
                and isinstance(host, str)
                and isinstance(public_port, int)
                and not isinstance(public_port, bool)
                and 1 <= public_port <= 65_535
            ):
                try:
                    ipaddress.ip_address(host)
                except ValueError:
                    pass
                else:
                    return host, public_port
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return None


def bootstrap_docker_arguments(
    encoded_bootstrap: str,
    *,
    input_timeout_seconds: int,
) -> str:
    if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded_bootstrap) is None:
        raise LaunchFailure("bootstrap payload is not valid base64")
    return (
        'bash -lc "set -e; '
        "mkdir -p /root/.ssh /run/sshd "
        f"{REMOTE_WORK_DIRECTORY}; "
        'echo \\"$PUBLIC_KEY\\" > /root/.ssh/authorized_keys; '
        "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
        "ssh-keygen -A; /usr/sbin/sshd; "
        f"echo {encoded_bootstrap} | base64 -d > "
        f"{REMOTE_WORK_DIRECTORY}/bootstrap_server.py; "
        f"EQUINOX_REMOTE_WORKDIR={REMOTE_WORK_DIRECTORY} "
        f"EQUINOX_WORKLOAD_FILE={EXTERNAL_EVALUATION_WORKLOAD} "
        f"EQUINOX_EXTERNAL_INPUT_TIMEOUT_SECONDS={input_timeout_seconds} "
        f'python3 {REMOTE_WORK_DIRECTORY}/bootstrap_server.py"'
    )


def bundle_sources(repository_root: Path) -> dict[str, Path]:
    common_sources = {
        "external_eval_remote_runner.sh": (
            repository_root / "research/runpod/external_eval_remote_runner.sh"
        ),
        "research/__init__.py": repository_root / "research/__init__.py",
        "research/runpod/__init__.py": repository_root / "research/runpod/__init__.py",
        "research/runpod/external_eval_transport.py": (
            repository_root / "research/runpod/external_eval_transport.py"
        ),
        "research/runpod/repository_repair_env.py": (
            repository_root / "research/runpod/repository_repair_env.py"
        ),
        "research/runpod/repository_repair_rl.py": (
            repository_root / "research/runpod/repository_repair_rl.py"
        ),
    }
    revision30_sources = {
        "research/external/revision30_task_pack.py": (
            repository_root / "research/external/revision30_task_pack.py"
        ),
        "research/frozen/revision30-external-pack.json": (
            repository_root / "research/frozen/revision30-external-pack.json"
        ),
        "research/runpod/revision30_external_eval.py": (
            repository_root / "research/runpod/revision30_external_eval.py"
        ),
    }
    revision31_sources = {
        "research/external/revision30_task_pack.py": (
            repository_root / "research/external/revision30_task_pack.py"
        ),
        "research/runpod/revision30_external_eval.py": (
            repository_root / "research/runpod/revision30_external_eval.py"
        ),
        "research/external/revision31_task_pack.py": (
            repository_root / "research/external/revision31_task_pack.py"
        ),
        "research/frozen/revision31-external-pack.json": (
            repository_root / "research/frozen/revision31-external-pack.json"
        ),
        "research/runpod/repository_repair_env_v31.py": (
            repository_root / "research/runpod/repository_repair_env_v31.py"
        ),
        "research/runpod/revision31_external_eval.py": (
            repository_root / "research/runpod/revision31_external_eval.py"
        ),
    }
    sources = {
        **common_sources,
        **(revision31_sources if EVALUATION_REVISION == "31" else revision30_sources),
    }
    if set(sources) != set(expected_bundle_files(EXTERNAL_EVALUATION_WORKLOAD)):
        raise LaunchFailure("external evaluation bundle does not match bootstrap allowlist")
    if not all(path.is_file() for path in sources.values()):
        raise LaunchFailure("external evaluation bundle source is incomplete")
    return sources


def build_bundle(repository_root: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, source in sorted(bundle_sources(repository_root).items()):
            payload = source.read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(member, io.BytesIO(payload))
    bundle = output.getvalue()
    if not bundle or len(bundle) > MAXIMUM_BUNDLE_BYTES:
        raise LaunchFailure("external evaluation bundle exceeds the inline transport ceiling")
    return bundle


def load_launch_inputs() -> tuple[str, Path, dict[str, Any], dict[str, Path]]:
    evaluation_id = os.environ.get("EQUINOX_EXTERNAL_EVALUATION_ID", "")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", evaluation_id) is None:
        raise LaunchFailure("external evaluation identity is invalid")
    manifest_path = Path(os.environ.get("EQUINOX_EXTERNAL_MANIFEST_PATH", "")).resolve()
    if not manifest_path.is_file():
        raise LaunchFailure("external evaluation manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_map_raw = json.loads(os.environ.get("EQUINOX_EXTERNAL_ARCHIVE_MAP", ""))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchFailure("external evaluation launch inputs are invalid") from error
    if not isinstance(manifest, dict) or not isinstance(archive_map_raw, dict):
        raise LaunchFailure("external evaluation launch inputs are invalid")
    validate_evaluation_manifest(load_input_manifest_from_value(manifest))
    if manifest["evaluation_id"] != evaluation_id:
        raise LaunchFailure("external evaluation identity does not match its manifest")
    expected_filenames = {adapter["archive_filename"] for adapter in manifest["adapters"]}
    if set(archive_map_raw) != expected_filenames:
        raise LaunchFailure("external adapter archive map is incomplete")
    archive_map = {}
    for adapter in manifest["adapters"]:
        path = Path(archive_map_raw[adapter["archive_filename"]]).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != adapter["size_bytes"]
            or sha256_file(path) != adapter["sha256"]
        ):
            raise LaunchFailure("external adapter archive evidence changed before launch")
        archive_map[adapter["archive_filename"]] = path
    return evaluation_id, manifest_path, manifest, archive_map


def load_frozen_pack(repository_root: Path) -> dict[str, Any]:
    filename = (
        "revision31-external-pack.json"
        if EVALUATION_REVISION == "31"
        else "revision30-external-pack.json"
    )
    return json.loads((repository_root / "research/frozen" / filename).read_text(encoding="utf-8"))


def expected_task_domains(
    repository_root: Path,
    frozen_pack: dict[str, Any],
) -> dict[str, str]:
    if isinstance(frozen_pack.get("tasks"), list):
        return {str(task["task_id"]): str(task["domain"]) for task in frozen_pack["tasks"]}
    if EVALUATION_REVISION != "31":
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    from research.external.revision31_task_pack import (
        canonical_json as task_canonical_json,
    )
    from research.external.revision31_task_pack import (
        external_tasks,
        task_descriptors,
    )

    tasks = external_tasks()
    descriptors = task_descriptors(tasks)
    observed_digest = hashlib.sha256(task_canonical_json(descriptors).encode()).hexdigest()
    if len(tasks) != frozen_pack.get("task_count") or observed_digest != frozen_pack.get(
        "task_descriptor_digest"
    ):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    return {task.task_id: task.domain for task in tasks}


def verify_policy_result(
    policy: dict[str, Any],
    *,
    policy_id: str,
    expected_tasks: dict[str, str],
) -> None:
    outcomes = policy.get("task_outcomes")
    if (
        policy.get("policy_id") != policy_id
        or policy.get("examples") != len(expected_tasks)
        or not isinstance(outcomes, list)
        or len(outcomes) != len(expected_tasks)
    ):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    observed = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise LaunchFailure(PROOF_CONTRACT_ERROR)
        task_id = outcome.get("task_id")
        if (
            task_id not in expected_tasks
            or task_id in observed
            or outcome.get("domain") != expected_tasks[task_id]
            or not isinstance(outcome.get("solved"), bool)
            or not isinstance(outcome.get("trajectory"), list)
            or outcome.get("actions") != len(outcome["trajectory"])
        ):
            raise LaunchFailure(PROOF_CONTRACT_ERROR)
        observed[task_id] = outcome
    successes = sum(outcome["solved"] for outcome in outcomes)
    total_actions = sum(outcome["actions"] for outcome in outcomes)
    malformed_actions = sum(outcome["malformed_actions"] for outcome in outcomes)
    expected_validity = (
        round((total_actions - malformed_actions) / total_actions, 6) if total_actions else None
    )
    if (
        policy.get("exact_successes") != successes
        or policy.get("exact_rate") != round(successes / len(expected_tasks), 6)
        or policy.get("total_actions") != total_actions
        or policy.get("malformed_actions") != malformed_actions
        or policy.get("action_protocol_validity_rate") != expected_validity
        or set(observed) != set(expected_tasks)
    ):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)


def paired_change_summary(
    base_outcomes: list[dict[str, Any]],
    adapter_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {item["task_id"]: bool(item["solved"]) for item in base_outcomes}
    adapter = {item["task_id"]: bool(item["solved"]) for item in adapter_outcomes}
    if base.keys() != adapter.keys() or not base:
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    improved = sum(not base[task_id] and adapter[task_id] for task_id in base)
    regressed = sum(base[task_id] and not adapter[task_id] for task_id in base)
    discordant = improved + regressed
    exact_p_value = 1.0
    if discordant:
        smaller_tail = (
            sum(math.comb(discordant, count) for count in range(min(improved, regressed) + 1))
            / 2**discordant
        )
        exact_p_value = min(1.0, 2 * smaller_tail)
    return {
        "examples": len(base),
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(base) - discordant,
        "net_improved": improved - regressed,
        "mcnemar_exact_p_value": exact_p_value,
    }


def verify_result_contract(
    result: dict[str, Any],
    manifest: dict[str, Any],
    frozen_pack: dict[str, Any],
) -> None:
    digest = result.get("result_digest")
    content = {key: value for key, value in result.items() if key != "result_digest"}
    observed_digest = "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()
    if (
        result.get("schema_version") != 1
        or result.get("workload") != WORKLOAD
        or result.get("workload_revision") != WORKLOAD_REVISION
        or result.get("external_evaluation_completed") is not True
        or result.get("device") != "cuda"
        or result.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}
        or result.get("pack") != frozen_pack
        or result.get("input_manifest") != manifest
        or result.get("adapter_count") != len(manifest["adapters"])
        or result.get("every_adapter_reported") is not True
        or result.get("task_count") != frozen_pack["task_count"]
        or result.get("domain_task_counts") != frozen_pack["domains"]
        or digest != observed_digest
    ):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    expected_tasks = expected_task_domains(Path(__file__).resolve().parents[2], frozen_pack)
    base = result.get("base")
    adapters = result.get("adapters")
    if not isinstance(base, dict) or not isinstance(adapters, list):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    verify_policy_result(
        base,
        policy_id="disabled_adapter_base",
        expected_tasks=expected_tasks,
    )
    if len(adapters) != len(manifest["adapters"]):
        raise LaunchFailure(PROOF_CONTRACT_ERROR)
    for index, expected in enumerate(manifest["adapters"]):
        observed = adapters[index]
        if (
            not isinstance(observed, dict)
            or any(observed.get(key) != value for key, value in expected.items())
            or observed.get("paired_change_vs_base")
            != paired_change_summary(
                base["task_outcomes"],
                observed.get("task_outcomes", []),
            )
        ):
            raise LaunchFailure(PROOF_CONTRACT_ERROR)
        verify_policy_result(
            observed,
            policy_id=expected["adapter_id"],
            expected_tasks=expected_tasks,
        )


def write_bytes_durably(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        handle.write(payload)
        if not payload.endswith(b"\n"):
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


class ExternalEvaluationLaunch:
    def __init__(
        self,
        repository_root: Path,
        *,
        evaluation_id: str,
        manifest_path: Path,
        manifest: dict[str, Any],
        archive_map: dict[str, Path],
    ) -> None:
        self.repository_root = repository_root
        self.evaluation_id = evaluation_id
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.archive_map = archive_map
        self.task_count = int(load_frozen_pack(repository_root)["task_count"])
        self.internal_token = self._internal_token()
        self.api_root = os.environ.get("EQUINOX_API_ROOT", "http://127.0.0.1:8180")
        self.dashboard_root = os.environ.get(
            "EQUINOX_DASHBOARD_ROOT",
            "http://127.0.0.1:3100",
        )
        self.gpu_id = os.environ.get("EQUINOX_RUNPOD_GPU", DEFAULT_GPU)
        self.template_id = os.environ.get("EQUINOX_RUNPOD_TEMPLATE", DEFAULT_TEMPLATE)
        self.cloud_type = os.environ.get("EQUINOX_RUNPOD_CLOUD_TYPE", "SECURE")
        if self.cloud_type not in {"COMMUNITY", "SECURE"}:
            raise LaunchFailure("EQUINOX_RUNPOD_CLOUD_TYPE is invalid")
        if not self.gpu_id:
            raise LaunchFailure("EQUINOX_RUNPOD_GPU cannot be empty")
        self.maximum_hourly_cost = environment_float(
            "EQUINOX_RUNPOD_MAX_HOURLY_COST",
            DEFAULT_IMAGE_CEILING,
            minimum=0.01,
            maximum=10,
        )
        self.maximum_lifetime_minutes = environment_integer(
            "EQUINOX_RUNPOD_MAX_LIFETIME_MINUTES",
            DEFAULT_MAXIMUM_LIFETIME_MINUTES,
            minimum=30,
            maximum=360,
        )
        self.boot_timeout_seconds = environment_integer(
            "EQUINOX_RUNPOD_BOOT_TIMEOUT_SECONDS",
            DEFAULT_BOOT_TIMEOUT_SECONDS,
            minimum=30,
            maximum=900,
        )
        self.input_timeout_seconds = environment_integer(
            "EQUINOX_EXTERNAL_INPUT_TIMEOUT_SECONDS",
            DEFAULT_INPUT_TIMEOUT_SECONDS,
            minimum=60,
            maximum=3_600,
        )
        self.container_disk_in_gb = environment_integer(
            "EQUINOX_RUNPOD_CONTAINER_DISK_GB",
            DEFAULT_CONTAINER_DISK_GB,
            minimum=10,
            maximum=100,
        )
        self.volume_in_gb = environment_integer(
            "EQUINOX_RUNPOD_VOLUME_GB",
            DEFAULT_VOLUME_GB,
            minimum=1,
            maximum=50,
        )
        self.image = self._template_image()
        self.bundle = build_bundle(repository_root)
        self.bootstrap_source = (
            repository_root / "research/runpod/bootstrap_server.py"
        ).read_bytes()
        self.proof_id = "runpod-proof-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.started_at = utc_now()
        self.started_epoch = time.time()
        self.provider_handle: str | None = None
        self.pod_id: str | None = None
        self.ssh_endpoint: tuple[str, int] | None = None
        self.hourly_cost: float | None = None
        self.execution_registered = False
        self.terminal_published = False
        self.last_progress: dict[str, Any] = {
            "phase": "requesting_capacity",
            "message": "Requesting one bounded RunPod worker.",
            "external_evaluation_id": evaluation_id,
            "adapter_count": len(manifest["adapters"]),
            "task_count": self.task_count,
            "elapsed_seconds": 0,
        }

    def _internal_token(self) -> str:
        token = os.environ.get("EQUINOX_INTERNAL_TOKEN", "")
        if not token:
            environment_path = self.repository_root / ".env"
            for line in environment_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("EQUINOX_INTERNAL_TOKEN="):
                    token = line.split("=", 1)[1]
        if len(token) < 32:
            raise LaunchFailure("EQUINOX_INTERNAL_TOKEN is missing or too short")
        return token

    def _template_image(self) -> str:
        template = runpod_json("template", "get", self.template_id)
        image = template.get("imageName") if isinstance(template, dict) else None
        if not isinstance(image, str) or not image:
            raise LaunchFailure("RunPod template did not resolve to a container image")
        return image

    def resource_profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "gpu_id": self.gpu_id,
            "gpu_count": 1,
            "image": self.image,
            "cloud_type": self.cloud_type,
            "persistent_volume_in_gb": self.volume_in_gb,
            "maximum_hourly_cost_usd": self.maximum_hourly_cost,
        }
        if self.hourly_cost is not None:
            profile["hourly_cost_usd"] = self.hourly_cost
        return profile

    def execution_payload(
        self,
        status: str,
        *,
        completed_at: str | None = None,
        teardown_confirmed: bool = False,
    ) -> dict[str, Any]:
        return {
            "name": "External adapter evaluation",
            "workload_id": WORKLOAD,
            "model_id": MODEL_ID,
            "branch_width": 4,
            "complexity_strategy": "adaptive",
            "status": status,
            "provider_name": "RunPod",
            "provider_handle": self.provider_handle,
            "resource_profile": self.resource_profile(),
            "progress": self.last_progress,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "teardown_confirmed": teardown_confirmed,
        }

    def publish(
        self,
        status: str,
        *,
        completed_at: str | None = None,
        teardown_confirmed: bool = False,
    ) -> None:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                request_json(
                    f"{self.api_root}/internal/research-compute-executions/{self.proof_id}",
                    method="PUT",
                    token=self.internal_token,
                    payload=self.execution_payload(
                        status,
                        completed_at=completed_at,
                        teardown_confirmed=teardown_confirmed,
                    ),
                )
                return
            except LaunchFailure as error:
                last_error = error
                time.sleep(1)
        raise LaunchFailure("observer API rejected external-evaluation state") from last_error

    def preflight(self) -> dict[str, Any]:
        return {
            "preflight_passed": True,
            "evaluation_id": self.evaluation_id,
            "adapter_count": len(self.manifest["adapters"]),
            "adapter_bytes": sum(adapter["size_bytes"] for adapter in self.manifest["adapters"]),
            "task_count": self.task_count,
            "gpu_id": self.gpu_id,
            "maximum_hourly_cost_usd": self.maximum_hourly_cost,
            "maximum_lifetime_minutes": self.maximum_lifetime_minutes,
            "bundle_size_bytes": len(self.bundle),
            "image": self.image,
        }

    def assert_idle_and_healthy(self) -> None:
        user = runpod_json("user")
        if float(user.get("currentSpendPerHr", -1)) != 0:
            raise LaunchFailure("RunPod already has active hourly spend")
        pods = runpod_json("pod", "list", "--all")
        if not isinstance(pods, list) or pods:
            raise LaunchFailure("RunPod already contains a pod")
        request_health(f"{self.api_root}/healthz")
        request_health(f"{self.dashboard_root}/runs")
        executions = request_json(f"{self.api_root}/v1/research-compute-executions")
        for execution in executions.get("items", []):
            if execution.get("status") not in {"PROVISIONING", "RUNNING", "FINALIZING"}:
                continue
            progress = dict(execution.get("progress") or {})
            progress.update(
                {
                    "message": "Reconciled after the operator process stopped.",
                    "error": "No RunPod pod or active hourly spend remained.",
                }
            )
            payload = {
                key: execution.get(key)
                for key in (
                    "name",
                    "workload_id",
                    "model_id",
                    "branch_width",
                    "complexity_strategy",
                    "provider_name",
                    "provider_handle",
                    "resource_profile",
                    "started_at",
                )
            }
            payload.update(
                {
                    "status": "FAILED",
                    "progress": progress,
                    "completed_at": utc_now(),
                    "teardown_confirmed": True,
                }
            )
            request_json(
                (
                    f"{self.api_root}/internal/research-compute-executions/"
                    f"{execution['execution_id']}"
                ),
                method="PUT",
                token=self.internal_token,
                payload=payload,
            )

    def create_pod(self) -> None:
        transport_token = hashlib.sha256(os.urandom(64)).hexdigest()
        self.transport_token = transport_token
        environment = json.dumps(
            {
                "EQUINOX_RESULT_TOKEN": transport_token,
                "EQUINOX_BUNDLE_B64": base64.b64encode(self.bundle).decode(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        bootstrap = base64.b64encode(self.bootstrap_source).decode()
        docker_arguments = bootstrap_docker_arguments(
            bootstrap,
            input_timeout_seconds=self.input_timeout_seconds,
        )
        terminate_after = (
            (datetime.now(UTC) + timedelta(minutes=self.maximum_lifetime_minutes))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        response = runpod_json(
            "pod",
            "create",
            "--name",
            f"equinox-{self.proof_id}",
            "--image",
            self.image,
            "--gpu-id",
            self.gpu_id,
            "--gpu-count",
            "1",
            "--cloud-type",
            self.cloud_type,
            "--container-disk-in-gb",
            str(self.container_disk_in_gb),
            "--volume-in-gb",
            str(self.volume_in_gb),
            "--volume-mount-path",
            "/workspace",
            "--ports",
            f"{PROXY_PORT}/http,22/tcp",
            "--ssh=true",
            "--env",
            environment,
            "--docker-args",
            docker_arguments,
            "--terminate-after",
            terminate_after,
            timeout=90,
        )
        pod_id = (
            response.get("id") or (response.get("pod") or {}).get("id")
            if isinstance(response, dict)
            else None
        )
        if not isinstance(pod_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", pod_id) is None:
            raise LaunchFailure("RunPod did not return a valid pod identity")
        self.pod_id = pod_id
        self.provider_handle = f"runpod://pods/{pod_id}"
        pod = runpod_json("pod", "get", pod_id)
        raw_cost = pod.get("adjustedCostPerHr") or pod.get("costPerHr") or pod.get("costPerHour")
        try:
            self.hourly_cost = float(raw_cost)
        except (TypeError, ValueError) as error:
            raise LaunchFailure("RunPod did not report a bounded hourly cost") from error
        if self.hourly_cost > self.maximum_hourly_cost:
            raise LaunchFailure(f"RunPod hourly cost {self.hourly_cost} exceeds configured ceiling")
        self.last_progress.update(
            {
                "phase": "waiting_for_capacity",
                "message": "Pod created; waiting for a live container endpoint.",
            }
        )
        self.publish("PROVISIONING")

    def proxy_root(self) -> str:
        if self.pod_id is None:
            raise LaunchFailure("RunPod pod identity is unavailable")
        return f"https://{self.pod_id}-{PROXY_PORT}.proxy.runpod.net"

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.boot_timeout_seconds
        while time.monotonic() < deadline:
            payload = proxy_get(
                self.proxy_root(),
                "/progress.json",
                self.transport_token,
                timeout=10,
                optional=True,
            )
            if payload is not None:
                try:
                    progress = json.loads(payload)
                except json.JSONDecodeError:
                    progress = None
                if isinstance(progress, dict):
                    self.last_progress.update(progress)
                    self.publish("RUNNING")
                    return
            self.last_progress["elapsed_seconds"] = round(time.time() - self.started_epoch, 3)
            self.publish("PROVISIONING")
            time.sleep(5)
        raise LaunchFailure("RunPod never exposed authenticated evaluation progress")

    def ssh_arguments(self, *remote_command: str) -> list[str]:
        if self.ssh_endpoint is None:
            raise LaunchFailure("RunPod SSH endpoint is unavailable")
        host, port = self.ssh_endpoint
        key_path = Path(
            os.environ.get(
                "EQUINOX_RUNPOD_SSH_KEY",
                "~/.runpod/ssh/runpodctl-ssh-key",
            )
        ).expanduser()
        if not key_path.is_file():
            raise LaunchFailure("RunPod SSH private key is unavailable")
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            "-i",
            str(key_path),
            "-p",
            str(port),
            f"root@{host}",
            *remote_command,
        ]

    def wait_until_ssh_ready(self) -> None:
        if self.pod_id is None:
            raise LaunchFailure("RunPod pod identity is unavailable")
        deadline = time.monotonic() + self.boot_timeout_seconds
        while time.monotonic() < deadline:
            pod = runpod_json("pod", "get", self.pod_id)
            endpoint = ssh_endpoint_from_pod(pod)
            if endpoint is not None:
                self.ssh_endpoint = endpoint
                completed = run_command(
                    self.ssh_arguments("true"),
                    timeout=25,
                )
                if completed.returncode == 0:
                    return
            time.sleep(5)
        raise LaunchFailure("RunPod never exposed a working full-SSH endpoint")

    def scp_input(
        self,
        source: Path,
        remote_name: str,
        *,
        finalize: bool = True,
    ) -> None:
        if self.ssh_endpoint is None:
            raise LaunchFailure("RunPod SSH endpoint is unavailable")
        if re.fullmatch(r"[a-zA-Z0-9._-]+", remote_name) is None:
            raise LaunchFailure("remote input filename is invalid")
        host, port = self.ssh_endpoint
        key_path = Path(
            os.environ.get(
                "EQUINOX_RUNPOD_SSH_KEY",
                "~/.runpod/ssh/runpodctl-ssh-key",
            )
        ).expanduser()
        pending_path = f"{REMOTE_INPUT_DIRECTORY}/{remote_name}.pending"
        completed = run_command(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "ConnectTimeout=20",
                "-i",
                str(key_path),
                "-P",
                str(port),
                str(source),
                f"root@{host}:{pending_path}",
            ],
            timeout=600,
        )
        if completed.returncode != 0:
            raise LaunchFailure(
                completed.stderr.decode(errors="replace").strip()
                or f"SCP upload failed: {remote_name}"
            )
        if not finalize:
            return
        completed = run_command(
            self.ssh_arguments(
                f"test ! -e {REMOTE_INPUT_DIRECTORY}/{remote_name} && "
                f"mv -- {pending_path} {REMOTE_INPUT_DIRECTORY}/{remote_name}"
            ),
            timeout=30,
        )
        if completed.returncode != 0:
            raise LaunchFailure(f"remote input commit failed: {remote_name}")

    def upload_inputs(self) -> None:
        total = len(self.manifest["adapters"])
        self.last_progress.update(
            {
                "phase": "waiting_for_secure_transfer",
                "message": "Container is ready; waiting for full-SSH file transfer.",
            }
        )
        self.publish("RUNNING")
        self.wait_until_ssh_ready()
        initialized = run_command(
            self.ssh_arguments(
                f"mkdir -p {REMOTE_INPUT_DIRECTORY} && "
                f"chmod 700 {REMOTE_INPUT_DIRECTORY} && "
                f'test -z "$(find {REMOTE_INPUT_DIRECTORY} -mindepth 1 '
                '-maxdepth 1 -print -quit)"'
            ),
            timeout=30,
        )
        if initialized.returncode != 0:
            raise LaunchFailure("remote input directory did not start empty")
        self.scp_input(self.manifest_path, "manifest.json")
        for index, adapter in enumerate(self.manifest["adapters"], start=1):
            filename = adapter["archive_filename"]
            self.last_progress.update(
                {
                    "phase": "uploading_adapters",
                    "message": f"Uploading retained adapter {index} of {total}.",
                    "adapter_upload_completed": index - 1,
                    "adapter_upload_total": total,
                    "elapsed_seconds": round(time.time() - self.started_epoch, 3),
                }
            )
            self.publish("RUNNING")
            self.scp_input(self.archive_map[filename], filename)
        ready = canonical_json({"manifest_sha256": sha256_file(self.manifest_path)})
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(ready)
            handle.flush()
            self.scp_input(Path(handle.name), "ready.json", finalize=False)
        ready_commit = run_command(
            self.ssh_arguments(
                f"cd {REMOTE_WORK_DIRECTORY} && "
                f"EQUINOX_REMOTE_WORKDIR={REMOTE_WORK_DIRECTORY} "
                f"PYTHONPATH={REMOTE_WORK_DIRECTORY} python3 -c "
                "'import os; from pathlib import Path; "
                "from research.runpod.external_eval_transport import "
                "verify_ready_payload; "
                f'root=Path("{REMOTE_INPUT_DIRECTORY}"); '
                'pending=root/"ready.json.pending"; '
                "verify_ready_payload(root, pending.read_bytes()); "
                'os.replace(pending, root/"ready.json")\''
            ),
            timeout=600,
        )
        if ready_commit.returncode != 0:
            error = ready_commit.stderr.decode(errors="replace").strip()
            raise LaunchFailure(
                "remote ready-fence verification failed" + (f": {error[-500:]}" if error else "")
            )
        self.last_progress.update(
            {
                "phase": "input_verification",
                "message": "All retained adapters uploaded; verifying sealed inputs.",
                "adapter_upload_completed": total,
                "adapter_upload_total": total,
            }
        )
        self.publish("RUNNING")

    def monitor(self) -> dict[str, Any]:
        deadline = self.started_epoch + self.maximum_lifetime_minutes * 60 - 60
        last_remote_progress: bytes | None = None
        while time.time() < deadline:
            exit_payload = proxy_get(
                self.proxy_root(),
                "/exit_code",
                self.transport_token,
                timeout=10,
                optional=True,
            )
            if exit_payload is not None:
                exit_code = exit_payload.decode(errors="replace").strip()
                if exit_code != "0":
                    error_payload = proxy_get(
                        self.proxy_root(),
                        "/error.log",
                        self.transport_token,
                        timeout=20,
                        optional=True,
                    )
                    tail = (
                        "\n".join(error_payload.decode(errors="replace").splitlines()[-40:])
                        if error_payload
                        else ""
                    )
                    raise LaunchFailure(
                        f"remote external evaluation exited {exit_code}"
                        + (f":\n{tail}" if tail else "")
                    )
                result_payload = proxy_get(
                    self.proxy_root(),
                    "/result.json",
                    self.transport_token,
                    timeout=120,
                )
                try:
                    result = json.loads(result_payload or b"")
                except json.JSONDecodeError as error:
                    raise LaunchFailure(PROOF_CONTRACT_ERROR) from error
                if not isinstance(result, dict):
                    raise LaunchFailure(PROOF_CONTRACT_ERROR)
                return result
            progress_payload = proxy_get(
                self.proxy_root(),
                "/progress.json",
                self.transport_token,
                timeout=10,
                optional=True,
            )
            if progress_payload is not None and progress_payload != last_remote_progress:
                try:
                    progress = json.loads(progress_payload)
                except json.JSONDecodeError:
                    progress = None
                if isinstance(progress, dict):
                    self.last_progress.update(progress)
                    self.publish("RUNNING")
                    last_remote_progress = progress_payload
            time.sleep(5)
        raise LaunchFailure("external evaluation exceeded the provider cleanup reserve")

    def pod_is_present(self) -> bool:
        if self.pod_id is None:
            return False
        pods = runpod_json("pod", "list", "--all")
        return any(isinstance(pod, dict) and pod.get("id") == self.pod_id for pod in pods)

    def teardown(self) -> bool:
        if self.pod_id is None:
            return True
        completed = run_command(
            ["runpodctl", "pod", "delete", self.pod_id],
            timeout=30,
        )
        if completed.returncode != 0:
            return False
        for _ in range(12):
            if not self.pod_is_present():
                self.pod_id = None
                return True
            time.sleep(5)
        return False

    def receipt(
        self,
        result: dict[str, Any],
        *,
        completed_at: str,
    ) -> dict[str, Any]:
        version = run_command(["runpodctl", "version"]).stdout.decode().split()
        cli_version = version[1] if len(version) >= 2 else "unknown"
        return {
            "provider_name": "RunPod",
            "provider_handle": self.provider_handle,
            "provider_cli_version": cli_version,
            "resource_profile": self.resource_profile(),
            "workload": {
                "id": result["workload"],
                "revision": result["workload_revision"],
                "algorithm": "greedy-base-and-lora-adapter-comparison",
                "static_branch_width": 4,
                "complexity_strategy": "adaptive",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "task_domains": list(result["domain_task_counts"]),
                "multi_step": True,
                "external_pack_id": PACK_ID,
                "adapter_count": result["adapter_count"],
            },
            "result": result,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "teardown_confirmed": True,
        }

    def run(self) -> dict[str, Any]:
        self.assert_idle_and_healthy()
        self.publish("PROVISIONING")
        self.execution_registered = True
        result: dict[str, Any] | None = None
        result_path = self.repository_root / "var/research-proofs" / f"{self.proof_id}.result.json"
        receipt_path = self.repository_root / "var/research-proofs" / f"{self.proof_id}.json"
        try:
            self.create_pod()
            self.wait_until_ready()
            self.upload_inputs()
            result = self.monitor()
            verify_result_contract(
                result,
                self.manifest,
                load_frozen_pack(self.repository_root),
            )
            write_bytes_durably(
                result_path,
                json.dumps(result, sort_keys=True, indent=2).encode(),
            )
            self.last_progress.update(
                {
                    "phase": "finalizing",
                    "message": "External result persisted; confirming RunPod teardown.",
                    "adapter_count": result["adapter_count"],
                    "task_count": result["task_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
            self.publish("FINALIZING")
            if not self.teardown():
                raise LaunchFailure("RunPod teardown could not be confirmed")
            completed_at = utc_now()
            receipt = self.receipt(result, completed_at=completed_at)
            write_bytes_durably(
                receipt_path,
                json.dumps(receipt, sort_keys=True, indent=2).encode(),
            )
            proof = request_json(
                f"{self.api_root}/internal/research-compute-proofs",
                method="POST",
                token=self.internal_token,
                payload=receipt,
                timeout=30,
            )
            self.terminal_published = True
            return {
                "execution_id": self.proof_id,
                "evaluation_id": self.evaluation_id,
                "proof": proof,
                "result_path": str(result_path),
                "receipt_path": str(receipt_path),
            }
        except Exception as error:
            teardown_confirmed = self.teardown()
            if self.execution_registered and not self.terminal_published:
                self.last_progress.update(
                    {
                        "phase": "failed",
                        "message": "External evaluation stopped before proof ingestion.",
                        "error": str(error),
                        "elapsed_seconds": round(time.time() - self.started_epoch, 3),
                    }
                )
                try:
                    self.publish(
                        "FAILED",
                        completed_at=utc_now(),
                        teardown_confirmed=teardown_confirmed,
                    )
                    self.terminal_published = True
                except Exception:
                    pass
            raise


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evaluation_id, manifest_path, manifest, archive_map = load_launch_inputs()
    launch = ExternalEvaluationLaunch(
        repository_root,
        evaluation_id=evaluation_id,
        manifest_path=manifest_path,
        manifest=manifest,
        archive_map=archive_map,
    )
    preflight_only = os.environ.get("EQUINOX_RUNPOD_PREFLIGHT_ONLY", "")
    if preflight_only not in {"", "1"}:
        raise LaunchFailure("EQUINOX_RUNPOD_PREFLIGHT_ONLY must be 1 when set")
    if preflight_only == "1":
        print(json.dumps(launch.preflight(), sort_keys=True, indent=2))
        return
    result = launch.run()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
