"""
src/pipeline/manifest.py
────────────────────────
Versioned ``RunManifest`` and atomic run-directory management.

Every execution starts with a versioned ``RunManifest``. A run is staged in a
fresh temporary directory and atomically published under its unique run id so
that service- or target-named artifact directories are never reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "1.0.0"

# Frozen baseline tag for the pre-v6 PoC-only pipeline.
BASELINE_TAG = "baseline-poc-only-v1"

# Fields whose values are redacted when hashing a configuration mapping so that
# secrets never enter a manifest hash or published artifact.
_SECRET_KEY_TOKENS = (
    "key", "token", "secret", "password", "passwd", "credential", "api_key",
    "apikey", "auth", "private",
)


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    return any(tok in k for tok in _SECRET_KEY_TOKENS)


def redact_secrets(obj: Any) -> Any:
    """Return a deep copy of *obj* with any secret-looking values replaced."""
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _is_secret_key(str(k)):
                out[k] = "***REDACTED***"
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return obj


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a configuration mapping with secrets redacted."""
    safe = redact_secrets(config)
    blob = json.dumps(safe, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _short_id() -> str:
    return secrets.token_hex(8)


@dataclass
class Scope:
    """Network boundaries a run is permitted to touch."""

    allowed_hostnames: list[str] = field(default_factory=list)
    allowed_networks: list[str] = field(default_factory=list)  # CIDR strings
    allowed_ports: list[int] = field(default_factory=list)
    allowed_schemes: list[str] = field(default_factory=list)
    callback_endpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scope":
        return cls(**{k: list(data.get(k, []) or []) for k in (
            "allowed_hostnames", "allowed_networks", "allowed_ports",
            "allowed_schemes", "callback_endpoints",
        )})


@dataclass
class ResourceLimits:
    max_runtime_seconds: int = 1200      # 20 minutes per target
    max_tool_calls: int = 50
    max_executed_commands: int = 40
    max_cves_per_service: int = 5
    max_methods_per_cve: int = 2
    max_executed_candidates: int = 3
    max_attempts_per_candidate: int = 3
    # These are first-class limits, rather than dynamically attached fields.
    # That makes the serialized manifest a complete runtime contract.
    max_total_tokens: int = 300_000
    max_llm_calls: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceLimits":
        return cls(**{k: data.get(k, getattr(cls, k)) for k in (
            "max_runtime_seconds", "max_tool_calls", "max_executed_commands",
            "max_cves_per_service", "max_methods_per_cve",
            "max_executed_candidates", "max_attempts_per_candidate",
            "max_total_tokens", "max_llm_calls",
        )})


def capture_repo_state(repo_path: str | None = None) -> dict[str, Any]:
    """Capture commit, tag, and dirty indicator for the framework repository."""
    repo = repo_path or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    state: dict[str, Any] = {"path": repo, "commit": "", "tag": "", "dirty": False}

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", repo, *args],
                check=False, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""

    state["commit"] = _git("rev-parse", "HEAD")
    state["dirty"] = bool(_git("status", "--porcelain"))
    tags = _git("tag", "--points-at", "HEAD")
    state["tag"] = tags.splitlines()[0] if tags else ""
    if not state["commit"]:
        state["dirty"] = True
    return state


@dataclass
class RunManifest:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    created_at: float = 0.0

    # Target / experimental design
    target_id: str = ""
    lab_id: str = ""
    repetition: int = 1
    condition: str = ""              # "clean" | "noisy"
    variant: str = ""               # 1..4, or "oracle_assisted"

    # Scope and resources
    scope: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)

    # Provenance
    repo: dict[str, Any] = field(default_factory=dict)
    model_provider: str = ""
    model_id: str = ""
    temperature: float = 0.0
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    tool_versions: dict[str, str] = field(default_factory=dict)
    source_snapshot_ids: list[str] = field(default_factory=list)
    source_snapshot_hashes: dict[str, str] = field(default_factory=dict)

    # Runtime results (filled in during/after the run)
    candidate_ids: list[str] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    oracle_spec: dict[str, Any] = field(default_factory=dict)
    oracle_result: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def stable_hash(self) -> str:
        blob = self.to_json()
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def new_manifest(
    target_id: str,
    *,
    lab_id: str = "",
    repetition: int = 1,
    condition: str = "",
    variant: str = "",
    scope: Scope | None = None,
    limits: ResourceLimits | None = None,
    model_provider: str = "",
    model_id: str = "",
    temperature: float = 0.0,
    prompt_hashes: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    oracle_spec: Mapping[str, Any] | None = None,
    repo_path: str | None = None,
) -> RunManifest:
    now = time.time()
    return RunManifest(
        run_id=f"run-{int(now)}-{_short_id()}",
        created_at=now,
        target_id=target_id,
        lab_id=lab_id,
        repetition=repetition,
        condition=condition,
        variant=variant,
        scope=(scope or Scope()).to_dict(),
        limits=(limits or ResourceLimits()).to_dict(),
        repo=capture_repo_state(repo_path),
        model_provider=model_provider,
        model_id=model_id,
        temperature=temperature,
        prompt_hashes=dict(prompt_hashes or {}),
        config_hash=config_hash(config or {}),
        tool_versions=dict(tool_versions or {}),
        oracle_spec=dict(oracle_spec or {}),
    )


class RunContext:
    """
    Stages a run in a fresh temporary directory and publishes it atomically
    under its unique run id. Directories are never reused.
    """

    def __init__(self, manifest: RunManifest, root: str = "") -> None:
        self.manifest = manifest
        self._root = root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "runs",
        )
        self._published = False
        self._staged = False
        self._staging_dir: str | None = None
        self.staged_dir: str = ""
        self.published_dir: str = ""
        self._stage()

    def _stage(self) -> None:
        os.makedirs(self._root, exist_ok=True)
        self._staging_dir = tempfile.mkdtemp(prefix=f"{self.manifest.run_id}-", dir=self._root)
        self.staged_dir = self._staging_dir
        self._staged = True

    def write(self, relpath: str, data: bytes | str) -> str:
        if not self._staged or self._staging_dir is None:
            raise RuntimeError("RunContext not staged")
        path = os.path.join(self._staging_dir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(relpath) else None
        with open(path, "wb" if isinstance(data, bytes) else "w") as fh:
            fh.write(data if isinstance(data, bytes) else data)
        return path

    def write_json(self, relpath: str, obj: Any) -> str:
        return self.write(relpath, json.dumps(obj, sort_keys=True, indent=2, default=str))

    def publish(self) -> str:
        if self._published or self._staging_dir is None:
            raise RuntimeError("RunContext already published or not staged")
        final_dir = os.path.join(self._root, self.manifest.run_id)
        # Atomic publish: rename staging dir into its final run-id location.
        os.makedirs(self._root, exist_ok=True)
        if os.path.exists(final_dir):
            raise RuntimeError(f"Run directory already exists (would reuse): {final_dir}")
        os.replace(self._staging_dir, final_dir)
        self.manifest.run_dir = final_dir
        # Persist the manifest itself inside the published run.
        with open(os.path.join(final_dir, "manifest.json"), "w") as fh:
            fh.write(self.manifest.to_json())
        self._published = True
        self.published_dir = final_dir
        self._staging_dir = None
        return final_dir

    def cleanup_unpublished(self) -> None:
        """Remove the staging directory if publishing never happened."""
        if self._staging_dir and os.path.isdir(self._staging_dir):
            import shutil
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None


def load_manifest(run_dir: str) -> RunManifest:
    with open(os.path.join(run_dir, "manifest.json")) as fh:
        return RunManifest.from_dict(json.load(fh))
