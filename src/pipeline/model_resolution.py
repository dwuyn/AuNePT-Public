"""Hash-pinned model-resolution evidence for Vertex benchmark profiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.vertex_runtime import (
    GEMMA_ENDPOINT_URL,
    LOCKED_MODEL_INVOCATIONS,
    VertexContractError,
    validate_gemma_endpoint_url,
    validate_resolution_fields,
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay below the artifact root")
    return path


def validate_resolution_lock(
    lock: Mapping[str, Any], *, profiles: Sequence[ModelProfile], artifact_root: str | Path,
    strict: bool = False,
) -> None:
    """Rehash each selected provider catalog record and bind it to its profile."""
    allowed_schemas = {"1.0.0", "2.0.0"} if not strict else {"2.0.0"}
    if lock.get("schema_version") not in allowed_schemas or not str(lock.get("generated_at", "")).endswith("Z"):
        raise ValueError("model resolution lock schema or timestamp is invalid")
    if strict:
        dataset_hash = str(lock.get("dataset_lock_hash", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
            raise ValueError("model resolution lock requires dataset_lock_hash")
    entries = lock.get("models")
    if not isinstance(entries, list) or len(entries) != len(profiles):
        raise ValueError("model resolution lock must contain one record per profile")
    by_label = {str(entry.get("logical_label", "")): entry for entry in entries if isinstance(entry, Mapping)}
    if len(by_label) != len(entries) or set(by_label) != {profile.logical_label for profile in profiles}:
        raise ValueError("model resolution lock labels do not match profiles")
    root = Path(artifact_root).resolve()
    for profile in profiles:
        entry = by_label[profile.logical_label]
        for key, expected in (
            ("resource_id", profile.resource_id),
            ("resource_revision", profile.resource_revision),
            ("resolution_mode", profile.resolution_mode),
            ("resolution_evidence_hash", profile.resolution_evidence_hash),
            ("resolution_resolved_at", profile.resolution_resolved_at),
        ):
            if str(entry.get(key, "")) != expected:
                raise ValueError(f"model resolution lock {profile.logical_label}.{key} mismatch")
        try:
            validate_resolution_fields(
                profile.logical_label, profile.resource_revision, profile.resolution_mode,
                profile.resolution_evidence_hash, profile.resolution_resolved_at,
            )
        except VertexContractError as exc:
            raise ValueError(str(exc)) from exc
        evidence_path = root / _relative(entry.get("metadata_path"), "metadata_path")
        if not evidence_path.is_file():
            raise ValueError(f"model resolution metadata is missing for {profile.logical_label}")
        metadata_file_hash = _hash_file(evidence_path)
        expected_file_hash = str(entry.get("metadata_sha256", profile.resolution_evidence_hash))
        if metadata_file_hash != expected_file_hash:
            raise ValueError(f"model resolution metadata hash mismatch for {profile.logical_label}")
        if strict:
            try:
                metadata = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"model resolution metadata is not valid JSON for {profile.logical_label}") from exc
            if not isinstance(metadata, Mapping):
                raise ValueError(f"model resolution metadata must be an object for {profile.logical_label}")
            if str(metadata.get("logical_label", "")) != profile.logical_label:
                raise ValueError(f"model resolution metadata label mismatch for {profile.logical_label}")
            expected_model = LOCKED_MODEL_INVOCATIONS[profile.logical_label]
            if str(metadata.get("model_id", "")) != expected_model["model_id"]:
                raise ValueError(f"model resolution metadata model ID mismatch for {profile.logical_label}")
            if str(metadata.get("api_family", "")) != expected_model["api_family"]:
                raise ValueError(f"model resolution metadata provider surface mismatch for {profile.logical_label}")
            for key, expected in (
                ("resource_id", profile.resource_id),
                ("resource_revision", profile.resource_revision),
                ("location", profile.location),
            ):
                if str(metadata.get(key, "")) != expected:
                    raise ValueError(f"model resolution metadata {profile.logical_label}.{key} mismatch")
            if profile.logical_label == "gemma-4-26b-a4b-it":
                if profile.resource_revision != "001":
                    raise ValueError("Gemma MaaS must be pinned to immutable revision @001")
                try:
                    validate_gemma_endpoint_url(profile.endpoint_url)
                except ValueError as exc:
                    raise ValueError("Gemma MaaS endpoint must be the canonical verified endpoint") from exc
                if profile.endpoint_url != GEMMA_ENDPOINT_URL:
                    raise ValueError("Gemma MaaS endpoint must be canonical")
                for key in ("endpoint_snapshot", "endpoint_source"):
                    path = root / _relative(entry.get(f"{key}_path"), f"{key}_path")
                    if not path.is_file() or _hash_file(path) != str(entry.get(f"{key}_sha256", "")):
                        raise ValueError(f"Gemma MaaS {key} evidence hash mismatch")
            supplied_metadata_hash = metadata.get("metadata_hash")
            if supplied_metadata_hash is not None:
                canonical = {key: value for key, value in metadata.items() if key != "metadata_hash"}
                expected_hash = hashlib.sha256(
                    json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
                ).hexdigest()
                if str(supplied_metadata_hash) != expected_hash:
                    raise ValueError(f"model resolution metadata canonical hash mismatch for {profile.logical_label}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("metadata_sha256", ""))):
                raise ValueError(f"model resolution lock requires metadata_sha256 for {profile.logical_label}")
