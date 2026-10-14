"""Runtime-only locks and credential/signature gates.

This module contains checks that are deliberately independent from provider
SDKs.  It is safe to use during verify-only preflight: it never creates a
Vertex client and never performs inference.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.vertex_runtime import validate_gemma_endpoint_url

ALIAS_EXCEPTION_SCHEMA = "1.0.0"
GATEWAY_RELAY_LOCK_SCHEMA = "1.0.0"
LOCKED_MODEL_LABELS = tuple(sorted(ModelProfile.ALLOWED_MODELS))
ALIAS_EXCEPTION_KEYS = frozenset({
    "schema_version", "project", "dataset_lock_hash", "model_labels", "expires_at", "reason",
})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _parse_utc(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _verify_detached_minisign(payload: bytes, signature_path: str | Path, public_key: str) -> None:
    if not public_key.strip():
        raise ValueError("minisign public key is required")
    signature = Path(signature_path)
    if not signature.is_file():
        raise ValueError("detached minisign signature is missing")
    minisign = shutil.which("minisign")
    if minisign is None:
        raise ValueError("minisign is unavailable; refusing unsigned runtime exception")
    with tempfile.TemporaryDirectory(prefix="veriplanpt-runtime-signature-") as temp:
        message = Path(temp) / "signed.json"
        message.write_bytes(payload)
        result = subprocess.run(
            [minisign, "-Vm", str(message), "-P", public_key, "-x", str(signature)],
            capture_output=True, text=True, check=False,
        )
    if result.returncode != 0:
        raise ValueError("minisign verification failed")


def verify_alias_exception(
    exception: Mapping[str, Any], *, signature_path: str | Path, public_key: str,
    project: str, dataset_lock_hash: str, now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the narrowly scoped signed exception for Gemini ``@default``."""
    if set(exception) != ALIAS_EXCEPTION_KEYS:
        unexpected = sorted(set(exception).difference(ALIAS_EXCEPTION_KEYS))
        missing = sorted(ALIAS_EXCEPTION_KEYS.difference(exception))
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError("alias exception fields are not exact (" + "; ".join(detail) + ")")
    if exception["schema_version"] != ALIAS_EXCEPTION_SCHEMA:
        raise ValueError("alias exception schema_version is invalid")
    if str(exception["project"]) != project or not project.strip():
        raise ValueError("alias exception project mismatch")
    if str(exception["dataset_lock_hash"]) != dataset_lock_hash or not SHA256.fullmatch(dataset_lock_hash):
        raise ValueError("alias exception dataset lock hash mismatch")
    labels = exception["model_labels"]
    if labels != list(LOCKED_MODEL_LABELS):
        raise ValueError("alias exception must cover exactly the three locked model labels")
    reason = str(exception["reason"])
    if "@default" not in reason or not reason.strip():
        raise ValueError("alias exception reason must explain the @default exception")
    expires = _parse_utc(exception["expires_at"], "alias exception expires_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if current >= expires:
        raise ValueError("alias exception has expired")
    _verify_detached_minisign(canonical_json(exception), signature_path, public_key)
    return {
        "schema_version": ALIAS_EXCEPTION_SCHEMA,
        "project": project,
        "dataset_lock_hash": dataset_lock_hash,
        "model_labels": list(LOCKED_MODEL_LABELS),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
    }


def validate_runtime_profile(
    profile: ModelProfile, *, strict: bool = False, r10: bool = False,
) -> None:
    """Apply the stronger runtime rules without breaking legacy pilot fixtures."""
    if not strict:
        return
    if profile.location != "global":
        raise ValueError(f"runtime profile {profile.logical_label} must use global")
    if profile.logical_label == "gemma-4-26b-a4b-it":
        if profile.resource_revision != "001":
            raise ValueError("Gemma MaaS runtime profile must use @001")
        if profile.generation_parameters.get("thinking") in {True, "true", "enabled"}:
            raise ValueError("Gemma thinking must be disabled")
        if profile.generation_parameters.get("thinking_enabled") in {True, "true", "enabled"}:
            raise ValueError("Gemma thinking must be disabled")
        if r10 and profile.generation_parameters.get("thinking_enabled") is not False:
            raise ValueError("r10 Gemma profile must explicitly disable thinking")
        if "max_output_tokens" in profile.generation_parameters and int(profile.generation_parameters["max_output_tokens"]) != 2048:
            raise ValueError("runtime generation cap must be max_output_tokens=2048")
        if r10 and int(profile.generation_parameters.get("max_output_tokens", 0)) != 2048:
            raise ValueError("r10 runtime profile must pin max_output_tokens=2048")
        if profile.usage_semantics.get("total_formula") != "input+output":
            raise ValueError("Gemma runtime profile must not bill thinking tokens")
        if profile.usage_semantics.get("output_includes_reasoning") != "true":
            raise ValueError("Gemma runtime profile must declare output-only billing")
        try:
            validate_gemma_endpoint_url(profile.endpoint_url)
        except ValueError as exc:
            raise ValueError("Gemma runtime profile requires the canonical metadata-pinned MaaS endpoint") from exc
    else:
        if profile.resource_revision != "default" or profile.resolution_mode != "provider_alias":
            raise ValueError(f"{profile.logical_label} must use the explicit @default provider alias")
        if "max_output_tokens" in profile.generation_parameters and int(profile.generation_parameters["max_output_tokens"]) != 2048:
            raise ValueError("runtime generation cap must be max_output_tokens=2048")
        if r10 and int(profile.generation_parameters.get("max_output_tokens", 0)) != 2048:
            raise ValueError("r10 runtime profile must pin max_output_tokens=2048")
        thinking_config = profile.generation_parameters.get("thinking_config")
        if isinstance(thinking_config, Mapping) and thinking_config.get("thinking_level") != "MEDIUM":
            raise ValueError("Gemini runtime thinking_level must be MEDIUM")
        if r10 and (
            not isinstance(thinking_config, Mapping)
            or thinking_config.get("thinking_level") != "MEDIUM"
        ):
            raise ValueError("r10 Gemini profile must pin thinking_level=MEDIUM")


def verify_impersonated_adc(service_account: str, *, project: str) -> dict[str, str]:
    """Verify ADC can mint an impersonated token; no Vertex request is made."""
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.iam\.gserviceaccount\.com", service_account):
        raise ValueError("runtime requires a service-account impersonation target")
    if not project.strip():
        raise ValueError("runtime requires a GCP project")
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise ValueError("gcloud is unavailable; ADC preflight cannot pass")
    result = subprocess.run(
        [gcloud, "auth", "application-default", "print-access-token",
         "--impersonate-service-account", service_account],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("ADC service-account impersonation preflight failed")
    return {"project": project, "service_account": service_account, "token_obtained": "true"}


def validate_lock_reference(value: Any, *, name: str, artifact_root: str | Path) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} reference must be an object")
    path_value = value.get("artifact_path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{name} reference requires artifact_path")
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} artifact_path must be relative")
    expected = str(value.get("sha256", ""))
    if not SHA256.fullmatch(expected):
        raise ValueError(f"{name} reference requires a SHA-256")
    actual = sha256_file(Path(artifact_root).resolve() / path)
    if actual != expected:
        raise ValueError(f"{name} artifact hash mismatch")
    return actual


def validate_gateway_relay_lock(
    lock: Mapping[str, Any], *, artifact_root: str | Path | None = None,
    observed_image_digest: str = "", strict: bool = True,
) -> None:
    """Validate the immutable host-gateway/relay boundary.

    The relay is intentionally described as data rather than inferred from a
    compose file at runtime.  This makes a changed socket mount, network mode,
    UID policy, or image fail the same hash gate as any other runtime input.
    ``strict=False`` is reserved for reading historical lock files.
    """
    if str(lock.get("schema_version")) != GATEWAY_RELAY_LOCK_SCHEMA:
        raise ValueError("gateway relay lock schema_version must be 1.0.0")
    relay = lock.get("relay")
    socket = lock.get("socket")
    network = lock.get("network")
    if not isinstance(relay, Mapping) or not isinstance(socket, Mapping) or not isinstance(network, Mapping):
        raise ValueError("gateway relay lock requires relay, socket, and network objects")
    image_digest = str(relay.get("image_digest", ""))
    if not IMAGE_DIGEST.fullmatch(image_digest):
        raise ValueError("gateway relay lock requires an immutable relay image digest")
    if observed_image_digest and observed_image_digest != image_digest:
        raise ValueError("gateway relay image digest does not match the locked digest")
    for section, label in ((relay.get("recipe"), "relay recipe"), (relay.get("source"), "relay source")):
        if not isinstance(section, Mapping) or not str(section.get("path", "")).strip() or not SHA256.fullmatch(str(section.get("sha256", ""))):
            raise ValueError(f"gateway relay lock requires a hashed {label}")
        section_path = Path(str(section["path"]))
        if section_path.is_absolute() or ".." in section_path.parts:
            raise ValueError(f"gateway relay {label} path must stay relative")
        if artifact_root is not None:
            path = Path(artifact_root).resolve() / section_path
            if not path.is_file() or sha256_file(path) != str(section["sha256"]):
                raise ValueError(f"gateway relay {label} hash mismatch")
    if strict and str(lock.get("uid_policy")) != "host_euid_nonroot":
        raise ValueError("gateway relay UID policy must be host_euid_nonroot")
    if str(socket.get("path")) != "/run/veriplanpt-gateway/gateway.sock":
        raise ValueError("gateway relay socket path is not the pinned gateway.sock")
    if str(socket.get("mode")) not in {"0600", "0o600"}:
        raise ValueError("gateway relay socket must use mode 0600")
    if str(socket.get("parent_mode", "0700")) not in {"0700", "0o700"}:
        raise ValueError("gateway relay socket parent must use mode 0700")
    if socket.get("mount_read_only") is not True:
        raise ValueError("gateway relay socket mount must be read-only")
    if str(network.get("mode")) != "internal":
        raise ValueError("gateway relay network must be internal-only")
    if str(relay.get("alias")) != "gateway-relay":
        raise ValueError("gateway relay alias is not pinned to gateway-relay")
    if str(relay.get("endpoint")) != "http://gateway-relay:8080/v1/generate":
        raise ValueError("gateway relay endpoint is not the pinned internal endpoint")
    if strict:
        if relay.get("run_as") != "host_uid_gid_nonroot":
            raise ValueError("gateway relay must run as the host non-root UID/GID")
        if lock.get("baseline_socket_mount") is not False or lock.get("baseline_credentials") is not False:
            raise ValueError("baseline containers must not receive the gateway socket or credentials")
