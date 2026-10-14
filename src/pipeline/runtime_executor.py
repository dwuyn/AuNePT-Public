"""Evidence-producing approved runtime executor; provider/evaluator are injected."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.pipeline.framework_adapter import ModelProfile, RunArtifact
from src.pipeline.protocol import validate_run_artifact
from src.pipeline.runtime_readiness import (
    execution_kind,
    is_readiness_contract,
    write_runtime_smoke_evidence,
)


@dataclass(frozen=True)
class RuntimeCellResult:
    """Evidence emitted by the real wrapper, evaluator, and Docker cleanup."""
    run_artifact: Mapping[str, Any]
    event_ledger: Any
    proof: Any
    usage: Mapping[str, Any]
    cost: Mapping[str, Any]
    evaluator: Mapping[str, Any]
    cleanup: Mapping[str, Any]
    billing_status: str
    oracle_status: str
    invocation_ledger: Mapping[str, Any] = field(default_factory=dict)
    oracle: Mapping[str, Any] = field(default_factory=dict)


CellExecutor = Callable[[Mapping[str, Any], Path], RuntimeCellResult]
Cleanup = Callable[[str], Mapping[str, Any]]


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute_runtime_plan(*, artifact_root: str | Path, plan: Mapping[str, Any],
                         profiles: Sequence[ModelProfile], training_protocol_hash: str,
                         baseline_lock_hash: str, pricing_snapshot_hash: str,
                         approval_hash: str, dataset_evidence_hash: str,
                         framework_commit: str, evaluator_commit: str,
                         gateway_relay_lock_hash: str = "",
                         runtime_topology_evidence_path: str = "",
                         runtime_topology_evidence_hash: str = "",
                         cell_executor: CellExecutor, cleanup: Cleanup) -> Path:
    """Run canaries sequentially then smokes; never manufacture evaluator/cleanup evidence."""
    root = Path(artifact_root).resolve()
    by_label = {profile.logical_label: profile for profile in profiles}
    cells = list(plan.get("cells", []))
    canaries = [cell for cell in cells if execution_kind(cell) == "vertex_canary"]
    smokes = [cell for cell in cells if execution_kind(cell) == "framework_model_smoke"]
    if len(canaries) != 3 or len(smokes) != 15:
        raise ValueError("runtime executor requires the approved 3+15 plan")
    production = bool(gateway_relay_lock_hash)
    if production and not runtime_topology_evidence_path:
        raise ValueError("production runtime execution requires topology evidence")
    records: list[dict[str, Any]] = []
    for cell in [*canaries, *smokes]:
        run_id, run_dir = str(cell["run_id"]), root / "runs" / str(cell["run_id"])
        result: RuntimeCellResult | None = None
        cleanup_evidence: Mapping[str, Any] | None = None
        try:
            result = cell_executor(cell, run_dir)
        finally:
            cleanup_evidence = cleanup(run_id)
        expected_oracle_status = "not_applicable" if is_readiness_contract(plan.get("runtime_contract")) else "passed"
        if result is None or result.billing_status != "known" or result.oracle_status != expected_oracle_status:
            raise RuntimeError("runtime stage halted by billing or oracle status")
        if production and not result.invocation_ledger:
            raise RuntimeError("production runtime requires observed invocation ledger")
        if cleanup_evidence.get("success") is not True:
            raise RuntimeError("runtime stage halted by cleanup failure")
        if result.cleanup != cleanup_evidence:
            raise ValueError("cell executor cleanup evidence must be the coordinator cleanup result")
        profile = by_label[str(cell["model_label"])]
        paths = {"run_artifact": run_dir / "run_artifact.json", "event_ledger": run_dir / "event-ledger.json",
                 "proof": run_dir / "proof.json", "usage": run_dir / "usage.json", "cost": run_dir / "cost.json", "evaluator": run_dir / "evaluator.json", "cleanup": run_dir / "cleanup.json"}
        if production:
            paths["invocation_ledger"] = run_dir / "invocation-ledger.json"
        if is_readiness_contract(plan.get("runtime_contract")):
            paths["oracle_applicability"] = run_dir / "oracle-applicability.json"
        ledger_hash, proof_hash = _write(paths["event_ledger"], result.event_ledger), _write(paths["proof"], result.proof)
        artifact = RunArtifact.from_dict(result.run_artifact)
        validate_run_artifact(result.run_artifact, official=True, strict_runtime=production)
        if artifact.model_profile.profile_hash != profile.profile_hash or artifact.model_profile.resource_revision != profile.resource_revision:
            raise ValueError("RunArtifact profile or resource revision drift")
        if artifact.event_ledger_hash != _canonical_hash(result.event_ledger) or artifact.proof_hash != _canonical_hash(result.proof):
            raise ValueError("RunArtifact ledger or proof hash does not match source evidence")
        if artifact.framework_identity.get("image_digest") != cell["image_digest"]:
            raise ValueError("RunArtifact framework image identity does not match cell")
        expected_context = {"dataset_lock_hash": cell["dataset_lock_hash"], "training_protocol_hash": training_protocol_hash,
                            "framework_commit": framework_commit, "evaluator_commit": evaluator_commit, "stage": "canary_smoke",
                            "target_runtime_lock_hash": cell["target_runtime_lock_hash"]}
        if production:
            expected_context["gateway_relay_lock_hash"] = gateway_relay_lock_hash
        if any(artifact.run_context.get(key) != value for key, value in expected_context.items()):
            raise ValueError("RunArtifact run context does not match pinned runtime inputs")
        usage = dict(result.usage)
        if {"input_tokens", "output_tokens", "total_tokens", "usd"}.difference(usage):
            raise ValueError("runtime usage is incomplete")
        if float(result.cost.get("cost_usd", -1)) != float(usage["usd"]) or result.cost.get("billing_status") != "known":
            raise ValueError("runtime usage and cost evidence mismatch")
        if any(float(artifact.usage[key]) != float(usage["usd"] if key == "total_usd" else usage[key]) for key in ("input_tokens", "output_tokens", "total_tokens", "total_usd")):
            raise ValueError("RunArtifact usage does not match usage evidence")
        if result.evaluator.get("status") != "passed":
            raise RuntimeError("runtime stage halted by evaluator failure")
        hashes = {"run_artifact": _write(paths["run_artifact"], result.run_artifact), "event_ledger": ledger_hash, "proof": proof_hash,
                  "usage": _write(paths["usage"], usage), "cost": _write(paths["cost"], result.cost), "evaluator": _write(paths["evaluator"], result.evaluator), "cleanup": _write(paths["cleanup"], cleanup_evidence)}
        if is_readiness_contract(plan.get("runtime_contract")):
            hashes["oracle_applicability"] = _write(paths["oracle_applicability"], result.oracle)
        if production:
            hashes["invocation_ledger"] = _write(paths["invocation_ledger"], result.invocation_ledger)
        record = {**dict(cell), "status": "passed", "plan_hash": plan["plan_hash"], "resource_id": profile.resource_id,
                  "resource_revision": profile.resource_revision, "resolution_mode": profile.resolution_mode,
                  "resolution_evidence_hash": profile.resolution_evidence_hash, "resolution_resolved_at": profile.resolution_resolved_at,
                  "artifact_path": str(paths["run_artifact"].relative_to(root)), "artifact_sha256": hashes["run_artifact"], "evidence_sha256": hashes["run_artifact"], "artifact_type": "run_artifact",
                  "proof_path": str(paths["proof"].relative_to(root)), "proof_sha256": hashes["proof"],
                  "billing_status": result.billing_status, "oracle_status": result.oracle_status,
                  "runtime_contract": str(plan.get("runtime_contract", ""))}
        if production:
            record["gateway_relay_lock_hash"] = gateway_relay_lock_hash
        for name in ("event_ledger", "usage", "cost", "evaluator", "cleanup"):
            record[f"{name}_path"] = str(paths[name].relative_to(root))
            record[f"{name}_sha256"] = hashes[name]
        if is_readiness_contract(plan.get("runtime_contract")):
            record["oracle_applicability_path"] = str(paths["oracle_applicability"].relative_to(root))
            record["oracle_applicability_sha256"] = hashes["oracle_applicability"]
        if production:
            record["invocation_ledger_path"] = str(paths["invocation_ledger"].relative_to(root))
            record["invocation_ledger_sha256"] = hashes["invocation_ledger"]
        if cell["kind"] == "framework_model_smoke":
            record["smoke_id"] = run_id
        records.append(record)
    return write_runtime_smoke_evidence(
        artifact_root=root, dataset_lock_hash=str(canaries[0]["dataset_lock_hash"]),
        dataset_evidence_hash=dataset_evidence_hash, canaries=records[:3], smokes=records[3:],
        plan_hash=str(plan["plan_hash"]), training_protocol_hash=training_protocol_hash,
        baseline_lock_hash=baseline_lock_hash,
        model_resolution_lock_hash=str(canaries[0]["model_resolution_lock_hash"]),
        pricing_snapshot_hash=pricing_snapshot_hash, approval_hash=approval_hash,
        gateway_relay_lock_hash=gateway_relay_lock_hash,
        runtime_topology_evidence_path=runtime_topology_evidence_path,
        runtime_topology_evidence_hash=runtime_topology_evidence_hash,
    )
