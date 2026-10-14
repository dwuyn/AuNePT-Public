"""Locked Docker executor for framework-owned 3+15 readiness."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.protocol import validate_run_artifact
from src.pipeline.runtime_executor import RuntimeCellResult
from src.pipeline.runtime_ledger import InvocationLedger
from src.pipeline.runtime_topology import TopologyHandle, TopologyLifecycle


class ReadinessContainerError(RuntimeError):
    """A readiness container failed its public boundary or evidence contract."""


def _require_sha256(value: Any, name: str) -> str:
    candidate = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ReadinessContainerError(f"{name} must be a lowercase SHA-256")
    return candidate


class ReadinessContainerExecutor:
    """Run `/runner/run` and bind its artifact to host-observed usage."""

    def __init__(
        self,
        *,
        profiles: Sequence[ModelProfile],
        topology: TopologyLifecycle,
        public_task: Mapping[str, Any],
        framework_identities: Mapping[str, Mapping[str, str]],
        evaluator_commit: str,
        training_protocol_hash: str,
        gateway_relay_lock_hash: str,
        source_snapshot_root: str = "",
        source_snapshot_hash: str = "",
    ) -> None:
        self.profiles = {profile.logical_label: profile for profile in profiles}
        self.topology = topology
        self.public_task = dict(public_task)
        self.framework_identities = {
            str(name): dict(identity) for name, identity in framework_identities.items()
        }
        self.evaluator_commit = evaluator_commit
        self.training_protocol_hash = training_protocol_hash
        self.gateway_relay_lock_hash = gateway_relay_lock_hash
        self.source_snapshot_root = source_snapshot_root
        self.source_snapshot_hash = source_snapshot_hash
        if not self.public_task or not str(self.public_task.get("case_id", "")):
            raise ReadinessContainerError("readiness requires one public validation task")

    def __call__(
        self,
        cell: Mapping[str, Any],
        run_dir: Path,
        labels: Mapping[str, str],
        _phase: str,
        handle: TopologyHandle,
        ledger: InvocationLedger,
    ) -> RuntimeCellResult:
        run_id = str(cell["run_id"])
        framework = str(cell.get("framework") or "VeriPlanPT")
        identity = self.framework_identities.get(framework)
        if identity is None:
            raise ReadinessContainerError(f"missing locked framework identity: {framework}")
        profile = self.profiles[str(cell["model_label"])]
        target_runtime_lock_hash = _require_sha256(
            cell.get("target_runtime_lock_hash", ""), "cell target runtime lock hash",
        )
        provenance = {
            "dataset_lock_hash": str(cell["dataset_lock_hash"]),
            "protocol_hash": self.training_protocol_hash,
            "framework_commit": str(identity["commit"]),
            "framework_image_digest": str(cell["image_digest"]),
            "framework_repository_url": str(identity["repository_url"]),
            "evaluator_commit": self.evaluator_commit,
            "target_runtime_lock_hash": target_runtime_lock_hash,
            "source_snapshot_hash": str(cell.get("source_snapshot_hash") or self.source_snapshot_hash),
        }
        transport_kind = str(cell.get("execution_kind") or cell.get("kind") or "")
        readiness_kind = str(cell.get("readiness_kind") or ("canary" if transport_kind == "vertex_canary" else "smoke"))
        condition = str(cell.get("condition") or transport_kind)
        evaluation_scope = str(cell.get("evaluation_scope") or "")
        invocation = {
            "schema_version": "2.0.0",
            "run_id": run_id,
            "framework": framework,
            "model_label": profile.logical_label,
            "case_id": str(self.public_task["case_id"]),
            "track": "blind",
            "execution_kind": transport_kind,
            "condition": condition,
            "evaluation_scope": evaluation_scope,
            "readiness_kind": readiness_kind,
            "task": self.public_task,
            "provenance": provenance,
            "labels": dict(labels),
            "model_profile": profile.to_dict(),
            "budget_tier": "medium",
            "repetition": 1,
            "parameters": {},
        }
        environment = self.topology.runtime_environment(
            handle, run_id=run_id, model_label=profile.logical_label,
            profile_hash=profile.profile_hash,
        )
        environment.update({
            "VERIPLANPT_STAGE": "canary_smoke",
            "VERIPLANPT_FRAMEWORK_NAME": framework,
            # The image root is read-only. Framework-owned checkpoints and
            # RunArtifact intermediates must stay inside the per-cell output
            # bind mount; never let the baked /run/veriplanpt default escape
            # the writable evidence boundary.
            "VERIPLANPT_RUN_DIR": "/run/veriplanpt/output",
            # Structured logging is initialized during graph import. Keep it
            # inside the same writable bind mount when the image root is
            # read-only.
            "LOG_DIR": "/run/veriplanpt/output",
            "VERIPLANPT_GATEWAY_RELAY_LOCK_HASH": self.gateway_relay_lock_hash,
            "VERIPLANPT_TARGET_RUNTIME_LOCK_HASH": target_runtime_lock_hash,
            "VERIPLANPT_ADAPTER_PRODUCTION": "true",
            "VERIPLANPT_IMAGE_DIGEST": str(cell["image_digest"]),
            "VERIPLANPT_DATASET_LOCK_HASH": str(cell["dataset_lock_hash"]),
            "VERIPLANPT_EVALUATOR_COMMIT": self.evaluator_commit,
            "VERIPLANPT_TRAINING_PROTOCOL_HASH": self.training_protocol_hash,
            "VERIPLANPT_MAX_LLM_CALLS": str(cell.get("max_llm_calls", 1)),
            "VERIPLANPT_REPOSITORY_URL": str(identity["repository_url"]),
            "VERIPLANPT_FRAMEWORK_COMMIT": str(identity["commit"]),
            # The host path is only a Docker bind source. Code inside the
            # container must use the stable container-side contract.
            "PENTEST_SOURCE_SNAPSHOT": "/run/veriplanpt/source-snapshot",
            "PENTEST_SOURCE_SNAPSHOT_HASH": str(cell.get("source_snapshot_hash") or self.source_snapshot_hash),
            "VERIPLANPT_EXECUTION_KIND": transport_kind,
            "VERIPLANPT_EVALUATION_SCOPE": evaluation_scope,
            "VERIPLANPT_READINESS_KIND": readiness_kind,
            "VERIPLANPT_METRIC_ELIGIBLE": "false" if cell.get("metric_eligible") is False else "true",
        })
        if transport_kind == "vertex_canary":
            environment["VERIPLANPT_CANARY_PROBE"] = "true"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = self.topology.run_baseline(
            handle, run_id=run_id, image=str(cell["image_digest"]),
            command=("/runner/run",), environment=environment,
            public_payload=json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode(),
            output_dir=run_dir,
        )
        if result.returncode != 0:
            raise ReadinessContainerError(f"readiness container failed: {result.stderr[-1000:]}")
        artifact_path = run_dir / "run_artifact.json"
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ReadinessContainerError("readiness container did not emit RunArtifact")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, Mapping):
            raise ReadinessContainerError("readiness RunArtifact is not an object")
        value = dict(artifact)
        validate_run_artifact(value, official=True, strict_runtime=True)
        if str(value.get("run_id", "")) != run_id:
            raise ReadinessContainerError("readiness RunArtifact run ID drifted")
        context = value.get("run_context")
        if not isinstance(context, Mapping) or str(context.get("target_runtime_lock_hash", "")) != target_runtime_lock_hash:
            raise ReadinessContainerError("readiness RunArtifact target lock binding drifted")
        observed = ledger.aggregate(run_id)
        value["usage"] = {
            **dict(value["usage"]),
            "input_tokens": int(observed["input_tokens"]),
            "output_tokens": int(observed["output_tokens"]),
            "total_tokens": int(observed["total_tokens"]),
            "total_usd": float(observed["usd"]),
        }
        validate_run_artifact(value, official=True, strict_runtime=True)
        artifact_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        event_ledger = value.get("transcript")
        proof = value.get("proof_submissions")
        if not isinstance(event_ledger, list) or not isinstance(proof, list):
            raise ReadinessContainerError("readiness RunArtifact omitted source evidence")
        cleanup = {
            "success": True,
            "run_id": run_id,
            "resources": {"container": {"ids": []}, "network": {"ids": []}},
            "errors": [],
        }
        usage = {
            "input_tokens": int(observed["input_tokens"]),
            "output_tokens": int(observed["output_tokens"]),
            "total_tokens": int(observed["total_tokens"]),
            "usd": float(observed["usd"]),
        }
        return RuntimeCellResult(
            run_artifact=value, event_ledger=event_ledger, proof=proof,
            usage=usage, cost={"billing_status": "known", "cost_usd": usage["usd"]},
            evaluator={}, cleanup=cleanup, billing_status="known", oracle_status="",
        )
