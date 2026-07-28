"""Evaluate frozen revision-30 adapters on the post-freeze external task pack."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.external.revision30_task_pack import (
    MAXIMUM_ACTIONS,
    PACK_ID,
    ExternalTaskEnvironment,
    external_tasks,
    pack_manifest,
)
from research.runpod import repository_repair_rl as frozen
from research.runpod.external_eval_transport import (
    load_input_manifest,
    sha256_file,
)

WORKLOAD = "revision30-post-freeze-external-adapter-evaluation"
WORKLOAD_REVISION = "revision30-external-adapter-evaluation@1"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
STUDY_ID = "repository-repair-confirmatory-study@1"
ADAPTER_SET = "all-retained-successful-study-adapters@1"
FROZEN_WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@30"
FROZEN_OBJECTIVE_ID = "verified-fix-coverage-retention-policy-gradient@15"
MAXIMUM_EXTRACTED_ADAPTER_BYTES = 180 * 1024 * 1024
PROGRESS_PATH = Path(os.environ.get("EQUINOX_PROGRESS_PATH", "/tmp/progress.json"))
EVALUATION_MANIFEST_KEYS = {
    "schema_version",
    "evaluation_id",
    "study_id",
    "adapter_set",
    "pack_id",
    "model",
    "adapters",
}
ADAPTER_MANIFEST_KEYS = {
    "adapter_id",
    "condition_id",
    "role",
    "optimization_seed",
    "source_execution_id",
    "source_result_sha256",
    "archive_filename",
    "size_bytes",
    "sha256",
    "adapter_manifest_digest",
}
ADAPTER_ROLES = {
    "frozen_reference",
    "fresh_replication",
    "matched_frozen_policy_control",
    "branch_width_ablation",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json_durably(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def emit_progress(phase: str, message: str, **values: Any) -> None:
    previous: dict[str, Any] = {}
    if PROGRESS_PATH.is_file():
        try:
            loaded = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        if isinstance(loaded, dict):
            previous = loaded
    write_json_durably(
        PROGRESS_PATH,
        {
            **previous,
            "schema_version": 1,
            "phase": phase,
            "message": message,
            "workload": WORKLOAD,
            "pack_id": PACK_ID,
            "model_id": MODEL_ID,
            **values,
        },
    )


def validate_evaluation_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if (
        set(manifest) != EVALUATION_MANIFEST_KEYS
        or not isinstance(manifest.get("evaluation_id"), str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", manifest["evaluation_id"]) is None
    ):
        raise ValueError("external-evaluation manifest has no evaluation identity")
    if manifest.get("study_id") != STUDY_ID or manifest.get("adapter_set") != ADAPTER_SET:
        raise ValueError("external-evaluation manifest references the wrong study")
    if manifest.get("pack_id") != PACK_ID:
        raise ValueError("external-evaluation manifest references the wrong task pack")
    model = manifest.get("model")
    if model != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("external-evaluation manifest references the wrong model")
    adapters = manifest["adapters"]
    source_execution_ids = set()
    condition_ids = set()
    for adapter in adapters:
        if (
            set(adapter) != ADAPTER_MANIFEST_KEYS
            or not isinstance(adapter["condition_id"], str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", adapter["condition_id"]) is None
            or adapter["condition_id"] in condition_ids
            or adapter["role"] not in ADAPTER_ROLES
            or isinstance(adapter["optimization_seed"], bool)
            or not isinstance(adapter["optimization_seed"], int)
            or adapter["optimization_seed"] < 0
            or not isinstance(adapter["source_execution_id"], str)
            or re.fullmatch(
                r"runpod-proof-[0-9]{8}T[0-9]{6}Z",
                adapter["source_execution_id"],
            )
            is None
            or adapter["source_execution_id"] in source_execution_ids
            or not isinstance(adapter["source_result_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", adapter["source_result_sha256"]) is None
            or not isinstance(adapter["adapter_manifest_digest"], str)
            or re.fullmatch(
                r"sha256:[a-f0-9]{64}",
                adapter["adapter_manifest_digest"],
            )
            is None
        ):
            raise ValueError("external-evaluation adapter evidence is incomplete")
        source_execution_ids.add(adapter["source_execution_id"])
        condition_ids.add(adapter["condition_id"])
    return manifest


def verify_external_pack(root: Path) -> dict[str, Any]:
    expected = json.loads(
        (root / "research/frozen/revision30-external-pack.json").read_text(encoding="utf-8")
    )
    observed = pack_manifest()
    if observed != expected:
        raise RuntimeError("the external task pack does not match its post-freeze manifest")
    return observed


def safe_extract_adapter(
    archive_path: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size: int,
    expected_manifest_digest: str,
) -> dict[str, Any]:
    if (
        archive_path.stat().st_size != expected_archive_size
        or sha256_file(archive_path) != expected_archive_sha256
    ):
        raise RuntimeError("adapter archive transport evidence does not match")
    pending = destination.with_name(destination.name + ".pending")
    if destination.exists() or pending.exists():
        raise RuntimeError("adapter extraction destination already exists")
    pending.mkdir(parents=True)
    extracted_bytes = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise RuntimeError("adapter archive is empty")
            for member in members:
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != "adapter"
                    or not (member.isdir() or member.isfile())
                ):
                    raise RuntimeError("adapter archive contains an unsafe member")
                if member.isdir():
                    continue
                extracted_bytes += member.size
                if extracted_bytes > MAXIMUM_EXTRACTED_ADAPTER_BYTES:
                    raise RuntimeError("adapter archive exceeds the extraction ceiling")
                relative = Path(*path.parts[1:])
                if not relative.parts:
                    raise RuntimeError("adapter archive contains an invalid file")
                target = pending / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("adapter archive member could not be read")
                with target.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
        manifest_path = pending / "adapter-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = {key: value for key, value in manifest.items() if key != "digest"}
        observed_digest = "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()
        if (
            manifest.get("digest") != expected_manifest_digest
            or observed_digest != expected_manifest_digest
            or manifest.get("schema_version") != 1
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("workload_revision") != FROZEN_WORKLOAD_REVISION
            or manifest.get("objective_id") != FROZEN_OBJECTIVE_ID
        ):
            raise RuntimeError("adapter manifest identity does not match")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError("adapter manifest has no files")
        declared_paths = set()
        for item in files:
            relative_path = item.get("path") if isinstance(item, dict) else None
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
                or relative_path in declared_paths
            ):
                raise RuntimeError("adapter manifest file identity is invalid")
            declared_paths.add(relative_path)
            target = pending / relative_path
            if (
                not target.is_file()
                or target.stat().st_size != item.get("size_bytes")
                or sha256_file(target) != item.get("sha256")
            ):
                raise RuntimeError("adapter file evidence does not match")
        required = {"adapter_config.json", "adapter_model.safetensors"}
        if not required.issubset(declared_paths):
            raise RuntimeError("adapter manifest omits required model files")
        observed_paths = {
            str(path.relative_to(pending)) for path in pending.rglob("*") if path.is_file()
        }
        if observed_paths != declared_paths | {"adapter-manifest.json"}:
            raise RuntimeError("adapter archive and manifest file sets do not match")
        os.replace(pending, destination)
        return manifest
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def extract_adapters(
    input_root: Path,
    manifest: dict[str, Any],
    extraction_root: Path,
) -> dict[str, Path]:
    paths = {}
    extraction_root.mkdir(parents=True, exist_ok=True)
    for adapter in manifest["adapters"]:
        destination = extraction_root / adapter["adapter_id"]
        safe_extract_adapter(
            input_root / adapter["archive_filename"],
            destination,
            expected_archive_sha256=adapter["sha256"],
            expected_archive_size=adapter["size_bytes"],
            expected_manifest_digest=adapter["adapter_manifest_digest"],
        )
        paths[adapter["adapter_id"]] = destination
    return paths


def evaluate_policy(
    policy_id: str,
    model: Any,
    tokenizer: Any,
    *,
    device: Any,
    started: float,
    torch: Any,
    stopping_criteria_type: Any,
    stopping_criteria_list_type: Any,
) -> dict[str, Any]:
    tasks = external_tasks()
    outcomes = []

    def sample(prompt: str) -> str:
        rendered = frozen.render_action_prompt(tokenizer, prompt)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=frozen.MAX_INPUT_TOKENS,
        ).to(device)
        input_width = encoded.input_ids.shape[1]

        class CompleteJsonObjectCriteria(stopping_criteria_type):
            def __call__(
                self,
                input_ids: Any,
                scores: Any,
                **kwargs: Any,
            ) -> Any:
                del scores, kwargs
                completed = [
                    frozen.complete_json_object(
                        frozen.ACTION_RESPONSE_PREFIX
                        + tokenizer.decode(row[input_width:], skip_special_tokens=True)
                    )
                    for row in input_ids
                ]
                return torch.tensor(completed, device=device, dtype=torch.bool)

        with torch.no_grad():
            sequence = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=frozen.MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria_list_type([CompleteJsonObjectCriteria()]),
            )
        response = (
            frozen.ACTION_RESPONSE_PREFIX
            + tokenizer.decode(sequence[0, input_width:], skip_special_tokens=True).strip()
        )
        del encoded, sequence
        return response

    model.eval()
    model.config.use_cache = True
    for index, task in enumerate(tasks):
        responses = []
        with ExternalTaskEnvironment(task) as environment:
            while not environment.terminal and len(environment.steps) < MAXIMUM_ACTIONS:
                response = sample(environment.policy_prompt())
                step = environment.step(response)
                responses.append(
                    {
                        "response": response,
                        "step": asdict(step),
                    }
                )
            solved, failures = environment.verify()
            outcomes.append(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "solved": solved,
                    "failed_checks": failures,
                    "terminal_reason": environment.terminal_reason,
                    "actions": len(environment.steps),
                    "accepted_actions": sum(step.accepted for step in environment.steps),
                    "malformed_actions": sum(step.action is None for step in environment.steps),
                    "trajectory": responses,
                }
            )
        emit_progress(
            "external_evaluation",
            f"Evaluating {policy_id}: {index + 1} of {len(tasks)} tasks.",
            policy_id=policy_id,
            policy_index=None,
            evaluation_completed=index + 1,
            evaluation_total=len(tasks),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    model.config.use_cache = False
    successes = sum(outcome["solved"] for outcome in outcomes)
    domain_successes = {
        domain: sum(outcome["solved"] for outcome in outcomes if outcome["domain"] == domain)
        for domain in ("micro_repository", "sqlite_data_repair", "filesystem_cli")
    }
    total_actions = sum(outcome["actions"] for outcome in outcomes)
    malformed_actions = sum(outcome["malformed_actions"] for outcome in outcomes)
    return {
        "policy_id": policy_id,
        "examples": len(outcomes),
        "exact_successes": successes,
        "exact_rate": round(successes / len(outcomes), 6),
        "domain_successes": domain_successes,
        "total_actions": total_actions,
        "accepted_actions": sum(outcome["accepted_actions"] for outcome in outcomes),
        "malformed_actions": malformed_actions,
        "action_protocol_validity_rate": (
            round((total_actions - malformed_actions) / total_actions, 6) if total_actions else None
        ),
        "task_outcomes": outcomes,
    }


def run_evaluation(root: Path, input_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    expected_pack = verify_external_pack(root)
    manifest = validate_evaluation_manifest(load_input_manifest(input_root / "manifest.json"))
    emit_progress(
        "input_verification",
        "Verifying uploaded adapter evidence.",
        adapter_count=len(manifest["adapters"]),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    adapter_paths = extract_adapters(
        input_root,
        manifest,
        root / "external-eval-adapters",
    )
    frozen.ensure_dependencies()
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    emit_progress(
        "model_loading",
        "Loading the frozen base model once.",
        gpu_name=torch.cuda.get_device_name(0),
        adapter_count=len(manifest["adapters"]),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device)
    base = evaluate_policy(
        "disabled_adapter_base",
        base_model,
        tokenizer,
        device=device,
        started=started,
        torch=torch,
        stopping_criteria_type=StoppingCriteria,
        stopping_criteria_list_type=StoppingCriteriaList,
    )
    adapters = manifest["adapters"]
    first = adapters[0]
    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[first["adapter_id"]],
        adapter_name=first["adapter_id"],
        is_trainable=False,
    )
    for adapter in adapters[1:]:
        model.load_adapter(
            adapter_paths[adapter["adapter_id"]],
            adapter_name=adapter["adapter_id"],
            is_trainable=False,
        )
    adapter_results = []
    for index, adapter in enumerate(adapters):
        model.set_adapter(adapter["adapter_id"])
        evaluation = evaluate_policy(
            adapter["adapter_id"],
            model,
            tokenizer,
            device=device,
            started=started,
            torch=torch,
            stopping_criteria_type=StoppingCriteria,
            stopping_criteria_list_type=StoppingCriteriaList,
        )
        adapter_results.append(
            {
                **adapter,
                **evaluation,
                "paired_change_vs_base": frozen.paired_change_summary(
                    base["task_outcomes"],
                    evaluation["task_outcomes"],
                ),
            }
        )
        emit_progress(
            "external_evaluation",
            f"Completed {adapter['adapter_id']}.",
            policy_id=adapter["adapter_id"],
            policy_index=index + 1,
            policy_total=len(adapters),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    result_without_digest = {
        "schema_version": 1,
        "workload": WORKLOAD,
        "workload_revision": WORKLOAD_REVISION,
        "external_evaluation_completed": True,
        "device": "cuda",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "pack": expected_pack,
        "input_manifest": manifest,
        "base": base,
        "adapters": adapter_results,
        "adapter_count": len(adapter_results),
        "every_adapter_reported": len(adapter_results) == len(adapters),
        "task_count": len(external_tasks()),
        "domain_task_counts": dict(Counter(task.domain for task in external_tasks())),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    result = {
        **result_without_digest,
        "result_digest": "sha256:"
        + hashlib.sha256(canonical_json(result_without_digest)).hexdigest(),
    }
    emit_progress(
        "complete",
        "External adapter evaluation completed.",
        adapter_count=len(adapter_results),
        elapsed_seconds=result["elapsed_seconds"],
    )
    return result


def main() -> None:
    root = Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp")).resolve()
    input_root = root / "inputs"
    result = run_evaluation(root, input_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
