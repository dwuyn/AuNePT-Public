"""Disposable, label-owned Docker topology for each runtime phase."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.pipeline.runtime_contract import validate_gateway_relay_lock

TOPOLOGY_SCHEMA = "1.0.0"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_IMAGE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TopologyError(RuntimeError):
    """A runtime topology could not be created or completely removed."""


class DockerBackend(Protocol):
    def run(
        self, args: Sequence[str], input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessDocker:
    def run(
        self, args: Sequence[str], input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["docker", *args], input=input_bytes,
            capture_output=True, text=False, check=False,
        )
        return subprocess.CompletedProcess(
            completed.args, completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )


@dataclass
class TopologyHandle:
    phase: str
    topology_id: str
    network_name: str
    socket_path: Path
    socket_parent: Path
    labels: dict[str, str]
    relay_container: str
    gateway_server: Any = None
    gateway_thread: threading.Thread | None = None
    baseline_containers: list[str] = field(default_factory=list)
    phase_token: str = ""
    token_expires_at: str = ""
    allowed_run_ids: set[str] = field(default_factory=set)
    epoch: str = ""


def _safe(value: str, name: str) -> str:
    if not _SAFE.fullmatch(value):
        raise TopologyError(f"{name} is not a safe Docker identifier")
    return value


def _label_args(labels: Mapping[str, str]) -> list[str]:
    values: list[str] = []
    for key, value in sorted(labels.items()):
        values.extend(["--label", f"{key}={value}"])
    return values


def _label_filters(labels: Mapping[str, str]) -> list[str]:
    values: list[str] = []
    for key, value in sorted(labels.items()):
        values.extend(["--filter", f"label={key}={value}"])
    return values


def validate_runtime_topology_evidence(value: Mapping[str, Any]) -> None:
    if str(value.get("schema_version")) != TOPOLOGY_SCHEMA:
        raise ValueError("runtime topology evidence schema_version must be 1.0.0")
    if value.get("success") is not True:
        raise ValueError("runtime topology evidence must record success=true")
    phases = value.get("phases")
    lock_hash = str(value.get("gateway_relay_lock_hash", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", lock_hash):
        raise ValueError("runtime topology evidence requires gateway relay lock hash")
    if not isinstance(phases, list) or [item.get("phase") for item in phases if isinstance(item, Mapping)] != ["canary", "smoke"]:
        raise ValueError("runtime topology evidence must contain canary then smoke phases")
    for phase in phases:
        if not isinstance(phase, Mapping) or phase.get("success") is not True:
            raise ValueError("runtime topology phase lifecycle did not succeed")
        if str(phase.get("gateway_relay_lock_hash")) != lock_hash or not str(phase.get("network_name", "")):
            raise ValueError("runtime topology phase is missing relay/network binding")


class TopologyLifecycle:
    """Start and remove one isolated gateway/relay/network per phase."""

    def __init__(
        self, *, artifact_root: str | Path, relay_lock: Mapping[str, Any],
        relay_image: str, docker: DockerBackend | None = None,
        host_uid: int | None = None, host_gid: int | None = None,
        source_snapshot_root: str | Path = "", source_snapshot_hash: str = "",
    ) -> None:
        self.root = Path(artifact_root).resolve()
        self.relay_lock = relay_lock
        self.relay_lock_hash = str(relay_lock.get("lock_hash", ""))
        self.docker = docker or SubprocessDocker()
        self.host_uid = os.geteuid() if host_uid is None else host_uid
        self.host_gid = os.getegid() if host_gid is None else host_gid
        self.source_snapshot_root = Path(source_snapshot_root) if source_snapshot_root else None
        self.source_snapshot_hash = source_snapshot_hash
        if self.host_uid == 0 or self.host_gid == 0:
            raise TopologyError("relay topology requires a non-root host UID/GID")
        if self.source_snapshot_root is not None:
            if self.source_snapshot_root.is_symlink() or not self.source_snapshot_root.is_dir():
                raise TopologyError("source snapshot mount must be a real directory")
        validate_gateway_relay_lock(relay_lock, strict=True)
        locked_relay = relay_lock.get("relay", {})
        locked_relay_tag = str(locked_relay.get("image", ""))
        locked_relay_digest = str(locked_relay.get("image_digest", ""))
        if relay_image not in {locked_relay_tag, locked_relay_digest}:
            raise TopologyError("runtime relay image does not match the relay lock")
        # The tag remains descriptive provenance; execution is always pinned
        # to the digest from the lock, even when legacy callers pass the tag.
        self.relay_image = locked_relay_digest
        self._active: dict[str, TopologyHandle] = {}

    def start(
        self, *, phase: str, run_ids: set[str], gateway_server: Any = None,
        gateway_factory: Any = None, gateway_token: str = "", token_expires_at: str = "", epoch: str = "",
    ) -> TopologyHandle:
        if phase not in {"canary", "smoke"}:
            raise TopologyError("runtime topology phase must be canary or smoke")
        if not run_ids or any(not _SAFE.fullmatch(run_id) for run_id in run_ids):
            raise TopologyError("runtime topology requires safe approved run IDs")
        if phase in self._active:
            raise TopologyError(f"{phase} topology is already active")
        topology_id = f"{phase}-{uuid.uuid4().hex[:12]}"
        network_name = _safe(f"veriplanpt-{topology_id}", "network")
        relay_name = _safe(f"veriplanpt-relay-{topology_id}", "relay container")
        socket_parent = self.root / "runtime-sockets" / topology_id
        socket_parent.mkdir(parents=True, exist_ok=False)
        os.chmod(socket_parent, 0o700)
        socket_path = socket_parent / "gateway.sock"
        labels = {
            "veriplanpt.managed": "true", "veriplanpt.phase": phase,
            "veriplanpt.topology_id": topology_id,
        }
        handle = TopologyHandle(
            phase=phase, topology_id=topology_id, network_name=network_name,
            socket_path=socket_path, socket_parent=socket_parent, labels=labels,
            relay_container=relay_name, gateway_server=gateway_server,
            phase_token=gateway_token, allowed_run_ids=set(run_ids),
            token_expires_at=token_expires_at,
            epoch=str(epoch),
        )
        try:
            network = self.docker.run(["network", "create", "--internal", *(_label_args(labels)), network_name])
            if network.returncode != 0:
                raise TopologyError(f"Docker network create failed: {network.stderr[-1000:]}")
            if gateway_server is None and gateway_factory is not None:
                try:
                    gateway_server = gateway_factory(socket_path, run_ids, gateway_token)
                except TypeError:
                    gateway_server = gateway_factory(socket_path, run_ids)
                handle.gateway_server = gateway_server
            if gateway_server is not None:
                handle.gateway_thread = threading.Thread(
                    target=gateway_server.serve_forever, name=f"veriplanpt-gateway-{phase}", daemon=True,
                )
                handle.gateway_thread.start()
            relay = self.docker.run([
                "run", "--detach", "--name", relay_name, "--network", network_name,
                "--network-alias", "gateway-relay", "--user", f"{self.host_uid}:{self.host_gid}",
                "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--mount", f"type=bind,src={socket_path},dst=/run/veriplanpt-gateway/gateway.sock,readonly",
                *(_label_args({**labels, "veriplanpt.run_ids": ",".join(sorted(run_ids))})),
                self.relay_image,
            ])
            if relay.returncode != 0:
                raise TopologyError(f"Docker relay start failed: {relay.stderr[-1000:]}")
            self._active[phase] = handle
            return handle
        except Exception:
            self.shutdown(handle)
            raise

    def run_baseline(
        self, handle: TopologyHandle, *, run_id: str, image: str, command: Sequence[str],
        environment: Mapping[str, str], public_payload: bytes,
        output_dir: str | Path,
    ) -> subprocess.CompletedProcess[str]:
        if handle.phase not in self._active or run_id not in handle.allowed_run_ids:
            raise TopologyError("baseline run is not bound to the active topology/run ID")
        if not _IMAGE.fullmatch(image) or not command:
            raise TopologyError("baseline image/command is not immutably pinned")
        required_runtime = {
            "VERIPLANPT_RUN_ID": run_id,
            "VERIPLANPT_PROVIDER_TOKEN": f"{handle.phase_token}~{run_id}~{environment.get('VERIPLANPT_PROFILE_HASH', '')}",
            "VERIPLANPT_PROVIDER_URL": "http://gateway-relay:8080/v1/generate",
            "VERIPLANPT_PROVIDER_BASE_URL": "http://gateway-relay:8080/v1",
        }
        if any(environment.get(key) != value for key, value in required_runtime.items()):
            raise TopologyError("baseline token, run-ID, or relay endpoint is not pinned")
        if not environment.get("VERIPLANPT_PROFILE_HASH"):
            raise TopologyError("baseline model profile hash is required")
        if not environment.get("VERIPLANPT_MODEL_LABEL"):
            raise TopologyError("baseline model label is required")
        if not environment.get("VERIPLANPT_PROVIDER_TOKEN_EXPIRES_AT"):
            raise TopologyError("baseline token expiry is required")
        env_args: list[str] = []
        for key, value in sorted(environment.items()):
            if key.startswith("GOOGLE_") or key in {"GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_ADC"}:
                raise TopologyError("baseline containers cannot receive Google credentials")
            env_args.extend(["--env", f"{key}={value}"])
        name = _safe(f"veriplanpt-{handle.phase}-{run_id}", "baseline container")
        labels = {**handle.labels, "veriplanpt.run_id": run_id}
        handle.baseline_containers.append(name)
        output = Path(output_dir).resolve()
        if not output.is_dir() or output.is_symlink():
            raise TopologyError("baseline output directory must be a real directory")
        env_args.extend(["--env", "VERIPLANPT_OUTPUT_DIR=/run/veriplanpt/output"])
        args = [
            "run", "--interactive", "--rm", "--name", name, "--user", f"{self.host_uid}:{self.host_gid}",
            "--network", handle.network_name,
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={output},dst=/run/veriplanpt/output",
        ]
        if self.source_snapshot_root is not None:
            args.extend(["--mount", f"type=bind,src={self.source_snapshot_root.resolve()},dst=/run/veriplanpt/source-snapshot,readonly"])
        args.extend([*(_label_args(labels)), *env_args, image, *command])
        return self.docker.run(args, input_bytes=public_payload)

    @staticmethod
    def runtime_environment(
        handle: TopologyHandle, *, run_id: str, model_label: str, profile_hash: str,
    ) -> dict[str, str]:
        if run_id not in handle.allowed_run_ids:
            raise TopologyError("runtime environment run-ID is not approved for this phase")
        if not handle.phase_token or not handle.token_expires_at:
            raise TopologyError("runtime environment token is incomplete")
        bound_token = f"{handle.phase_token}~{run_id}~{profile_hash}"
        return {
            "VERIPLANPT_RUN_ID": run_id,
            "VERIPLANPT_PROVIDER_TOKEN": bound_token,
            "VERIPLANPT_PROVIDER_TOKEN_EXPIRES_AT": handle.token_expires_at,
            "VERIPLANPT_PROVIDER_URL": "http://gateway-relay:8080/v1/generate",
            "VERIPLANPT_PROVIDER_BASE_URL": "http://gateway-relay:8080/v1",
            "OPENAI_BASE_URL": "http://gateway-relay:8080/v1",
            "OPENAI_API_BASE": "http://gateway-relay:8080/v1",
            "OPENAI_BASEURL": "http://gateway-relay:8080/v1",
            "OPENAI_API_KEY": bound_token,
            "VERIPLANPT_EPOCH": handle.epoch,
            "VERIPLANPT_PROFILE_HASH": profile_hash,
            "VERIPLANPT_MODEL_LABEL": model_label,
            "VERIPLANPT_GATEWAY_LIVE": "true",
        }

    def shutdown(self, handle: TopologyHandle) -> dict[str, Any]:
        """Always close host resources first, then remove only owned labels."""
        errors: list[str] = []
        if handle.gateway_server is not None:
            try:
                handle.gateway_server.shutdown()
            except Exception as exc:
                errors.append(f"gateway shutdown: {exc}")
            try:
                handle.gateway_server.server_close()
            except Exception as exc:
                errors.append(f"gateway close: {exc}")
        if handle.gateway_thread is not None:
            handle.gateway_thread.join(timeout=5)
            if handle.gateway_thread.is_alive():
                errors.append("gateway thread did not stop")
        try:
            if handle.socket_path.is_socket() or handle.socket_path.is_symlink():
                handle.socket_path.unlink()
            if handle.socket_parent.exists():
                handle.socket_parent.rmdir()
        except OSError as exc:
            errors.append(f"socket cleanup: {exc}")
        for kind in ("container", "network"):
            listed = self.docker.run([kind, "ls", "-q", *(_label_filters(handle.labels))])
            ids = [item for item in listed.stdout.splitlines() if item]
            if listed.returncode != 0:
                errors.append(f"{kind} list: {listed.stderr[-1000:]}")
            if ids:
                command = [kind, "rm", *ids]
                if kind == "container":
                    command.insert(2, "--force")
                removed = self.docker.run(command)
                if removed.returncode != 0:
                    errors.append(f"{kind} remove: {removed.stderr[-1000:]}")
        self._active.pop(handle.phase, None)
        return {
            "schema_version": TOPOLOGY_SCHEMA, "phase": handle.phase,
            "topology_id": handle.topology_id, "network_name": handle.network_name,
            "gateway_relay_lock_hash": self.relay_lock_hash,
            "resources": {"relay": handle.relay_container, "baselines": list(handle.baseline_containers)},
            "success": not errors, "errors": errors,
        }


def write_runtime_topology_evidence(
    path: str | Path, *, gateway_relay_lock_hash: str, phases: Sequence[Mapping[str, Any]],
) -> Path:
    value = {
        "schema_version": TOPOLOGY_SCHEMA, "gateway_relay_lock_hash": gateway_relay_lock_hash,
        "phases": [dict(item) for item in phases], "success": all(item.get("success") is True for item in phases),
    }
    validate_runtime_topology_evidence(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
