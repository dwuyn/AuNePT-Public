"""Production two-phase runtime runner with relay/topology fail-closed gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from src.pipeline.bundle_executor import IndependentBundleExecutor
from src.pipeline.experiment_runner import CellResult, ExperimentRunner
from src.pipeline.framework_adapter import ModelProfile, RunArtifact
from src.pipeline.protocol import validate_run_artifact
from src.pipeline.readiness_evidence import validate_smoke_evidence
from src.pipeline.runtime_contract import sha256_file, validate_gateway_relay_lock
from src.pipeline.runtime_executor import RuntimeCellResult
from src.pipeline.runtime_ledger import (
    BillableInvocationError,
    BillingUnknownError,
    InvocationLedger,
)
from src.pipeline.runtime_readiness import (
    execution_kind,
    is_readiness_contract,
    validate_canary_smoke_plan,
    write_runtime_smoke_evidence,
)
from src.pipeline.runtime_topology import (
    TopologyHandle,
    TopologyLifecycle,
    write_runtime_topology_evidence,
)

RuntimeCellExecutor = Callable[..., RuntimeCellResult]
BundleExecutor = Callable[[Path, Path, Mapping[str, Any]], Mapping[str, Any]]


def _bundle_hash(path: str | Path) -> str:
    root = Path(path).resolve()
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ValueError(f"runtime bundle is missing: {root}")
    digest = hashlib.sha256()
    for item in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RuntimeHalt(RuntimeError):
    """A runtime cell violated a production gate."""


class RuntimeRunner:
    """Run exactly three canaries, then fifteen smokes on a fresh topology."""

    def __init__(
        self, *, artifact_root: str | Path, plan: Mapping[str, Any], profiles: Sequence[ModelProfile],
        relay_lock: Mapping[str, Any], relay_lock_hash: str, relay_image: str,
        approval: Mapping[str, Any], signature_path: str | Path, public_key: str,
        evaluator_bundle: str | Path, oracle_bundle: str | Path,
        evaluator_bundle_hash: str, oracle_bundle_hash: str,
        cell_executor: RuntimeCellExecutor, bundle_executor: BundleExecutor | None = None,
        hidden_case_root: str | Path | None = None,
        dataset_evidence_hash: str = "", training_protocol_hash: str = "",
        pricing_snapshot_hash: str = "", approval_hash: str = "",
        topology: TopologyLifecycle | None = None,
        gateway_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.root = Path(artifact_root).resolve()
        self.plan = dict(plan)
        self.profiles = list(profiles)
        self.relay_lock = dict(relay_lock)
        self.relay_lock_hash = relay_lock_hash
        self.relay_image = relay_image
        self.approval = approval
        self.signature_path = signature_path
        self.public_key = public_key
        self.evaluator_bundle = Path(evaluator_bundle).resolve()
        self.oracle_bundle = Path(oracle_bundle).resolve()
        self.evaluator_bundle_hash = evaluator_bundle_hash
        self.oracle_bundle_hash = oracle_bundle_hash
        self.cell_executor = cell_executor
        hidden_root = hidden_case_root or os.environ.get("VERIPLANPT_HIDDEN_CASE_ROOT") or None
        self.bundle_executor = bundle_executor or IndependentBundleExecutor(
            evaluator_bundle=self.evaluator_bundle, oracle_bundle=self.oracle_bundle,
            hidden_case_root=hidden_root,
        )
        self.dataset_evidence_hash = dataset_evidence_hash
        self.training_protocol_hash = training_protocol_hash
        self.pricing_snapshot_hash = pricing_snapshot_hash
        self.approval_hash = approval_hash
        self.topology = topology or TopologyLifecycle(
            artifact_root=self.root, relay_lock={**self.relay_lock, "lock_hash": relay_lock_hash},
            relay_image=relay_image,
        )
        self.gateway_factory = gateway_factory
        self._result_lock = Lock()

    def _validate_inputs(self) -> None:
        validate_gateway_relay_lock(self.relay_lock, strict=True)
        if not isinstance(self.bundle_executor, IndependentBundleExecutor):
            raise RuntimeHalt("production runtime requires the independent bundle executor")
        if not re.fullmatch(r"[0-9a-f]{64}", self.relay_lock_hash):
            raise RuntimeHalt("runtime relay lock hash is invalid")
        if not Path(self.signature_path).is_file():
            raise RuntimeHalt("runtime approval signature is missing")
        if not self.evaluator_bundle.is_file() and not self.evaluator_bundle.is_dir():
            raise RuntimeHalt("evaluator bundle is missing")
        if not self.oracle_bundle.is_file() and not self.oracle_bundle.is_dir():
            raise RuntimeHalt("oracle bundle is missing")
        if _bundle_hash(self.evaluator_bundle) != self.evaluator_bundle_hash:
            raise RuntimeHalt("evaluator bundle hash mismatch")
        if _bundle_hash(self.oracle_bundle) != self.oracle_bundle_hash:
            raise RuntimeHalt("oracle bundle hash mismatch")
        if str(self.plan.get("schema_version")) not in {"1.1.0", "1.2.0"}:
            raise RuntimeHalt("production runtime requires canary plan 1.1.0 or 1.2.0")
        if str(self.plan.get("gateway_relay_lock_hash")) != self.relay_lock_hash:
            raise RuntimeHalt("runtime plan gateway relay lock hash mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", str(self.plan.get("target_runtime_lock_hash", ""))):
            raise RuntimeHalt("runtime plan target runtime lock hash is invalid")
        validate_canary_smoke_plan(self.plan, profiles=self.profiles, strict=True)

    @staticmethod
    def _phase_cells(plan: Mapping[str, Any], phase: str) -> list[Mapping[str, Any]]:
        kind = "vertex_canary" if phase == "canary" else "framework_model_smoke"
        return [cell for cell in plan["cells"] if execution_kind(cell) == kind]

    def _call_executor(
        self, cell: Mapping[str, Any], run_dir: Path, labels: Mapping[str, str],
        phase: str, topology: TopologyHandle, ledger: InvocationLedger,
    ) -> RuntimeCellResult:
        result = self.cell_executor(cell, run_dir, labels, phase, topology, ledger)
        if not isinstance(result, RuntimeCellResult):
            raise RuntimeHalt("runtime cell executor returned an invalid result")
        artifact = RunArtifact.from_dict(result.run_artifact)
        # The successor plan carries the runtime-contract marker. Legacy
        # fixture plans remain readable for migration tests, while r10 plans
        # take the strict termination/evidence gate.
        strict_runtime = bool(self.plan.get("runtime_contract"))
        validate_run_artifact(result.run_artifact, official=True, strict_runtime=strict_runtime)
        profile = next(profile for profile in self.profiles if profile.logical_label == cell["model_label"])
        if artifact.run_id != str(cell["run_id"]):
            raise RuntimeHalt("runtime RunArtifact run-ID mismatch")
        if artifact.model_profile.profile_hash != profile.profile_hash or artifact.model_profile.resource_revision != profile.resource_revision:
            raise RuntimeHalt("runtime RunArtifact profile mismatch")
        if artifact.framework_identity.get("image_digest") != cell["image_digest"]:
            raise RuntimeHalt("runtime RunArtifact image digest mismatch")
        if artifact.run_context.get("gateway_relay_lock_hash") != self.relay_lock_hash:
            raise RuntimeHalt("runtime RunArtifact relay lock binding mismatch")
        if artifact.run_context.get("target_runtime_lock_hash") != str(self.plan["target_runtime_lock_hash"]):
            raise RuntimeHalt("runtime RunArtifact target lock binding mismatch")
        observed = ledger.aggregate(str(cell["run_id"]))
        if result.billing_status != "known":
            raise RuntimeHalt("billing unknown halted runtime")
        if str(self.plan.get("runtime_contract")) in {
            "veriplanpt-runtime-v0.4.0-r10.5", "veriplanpt-runtime-v0.4.0-r10.6",
            "veriplanpt-runtime-v0.4.0-r10.8", "veriplanpt-runtime-v0.4.0-r10.10",
        }:
            observed_rows = [
                row for row in ledger.snapshot() if row.get("run_id") == str(cell["run_id"])
            ]
            if len(observed_rows) != 1 or int(observed_rows[0].get("call_index", -1)) != 0:
                raise RuntimeHalt("r10.7 readiness requires exactly one ledger response at call_index=0")
        if result.cleanup.get("success") is not True:
            raise RuntimeHalt("cell cleanup failed")
        resources = result.cleanup.get("resources", {})
        if isinstance(resources, Mapping) and any(
            item.get("ids") for item in resources.values() if isinstance(item, Mapping)
        ):
            raise RuntimeHalt("cell cleanup left managed Docker resources")
        reported = result.usage
        if reported and any(
            float(reported.get(key, -1)) != float(observed[key])
            for key in ("input_tokens", "output_tokens", "total_tokens", "usd")
        ):
            raise RuntimeHalt("runtime usage differs from observed gateway ledger")
        artifact_usage = artifact.usage
        if any(
            float(artifact_usage.get(key, -1))
            != float(observed["usd" if key == "total_usd" else key])
            for key in ("input_tokens", "output_tokens", "total_tokens", "total_usd")
        ):
            raise RuntimeHalt("RunArtifact usage differs from observed gateway ledger")
        if artifact.event_ledger_hash != _canonical_hash(result.event_ledger):
            raise RuntimeHalt("RunArtifact event ledger hash differs from source evidence")
        if artifact.proof_hash != _canonical_hash(result.proof):
            raise RuntimeHalt("RunArtifact proof hash differs from source evidence")
        normalized_cost = {"billing_status": "known", "cost_usd": observed["usd"]}
        # The independent evaluator validates the host-observed invocation
        # ledger.  Materialize the current phase ledger before invoking it;
        # the file is extended atomically as later cells complete.
        ledger_path = self.root / "runtime" / f"{phase}-invocation-ledger.json"
        ledger.write(ledger_path)
        # Evidence must exist before either independent verdict runs.  In
        # particular, never accept an executor-supplied oracle status: that
        # creates a self-verdict loop and permits an oracle pass before the
        # evaluator has seen the source evidence.
        source_paths = {
            "run_artifact": run_dir / "run_artifact.json",
            "event_ledger": run_dir / "event-ledger.json",
            "proof": run_dir / "proof.json",
            "usage": run_dir / "usage.json",
            "cost": run_dir / "cost.json",
            "cleanup": run_dir / "cleanup.json",
        }
        for name, value in (("run_artifact", result.run_artifact), ("event_ledger", result.event_ledger),
                            ("proof", result.proof), ("usage", observed),
                            ("cost", normalized_cost), ("cleanup", result.cleanup)):
            source_paths[name].parent.mkdir(parents=True, exist_ok=True)
            source_paths[name].write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # The independent evaluator consumes the framework-phase cleanup
        # seal. Readiness uses the same cleanup result as the per-cell
        # evidence, so persist it under the evaluator's protocol name before
        # invoking either independent bundle.
        (run_dir / "framework-cleanup.json").write_text(
            json.dumps(result.cleanup, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        evaluator_verdict = self.bundle_executor(self.evaluator_bundle, run_dir, result.run_artifact)
        if evaluator_verdict.get("status") != "passed" and evaluator_verdict.get("evaluator", {}).get("status") != "passed":
            raise RuntimeHalt("evaluator bundle failed")
        evaluator_path = run_dir / "evaluator.json"
        evaluator_path.write_text(json.dumps(dict(evaluator_verdict), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        oracle_verdict: Mapping[str, Any]
        if is_readiness_contract(self.plan.get("runtime_contract")):
            oracle_verdict = {
                "schema_version": "1.0.0",
                "status": "not_applicable",
                "reason": "no_target_condition_in_readiness",
                "osr": None,
                "plan_hash": str(self.plan["plan_hash"]),
                "run_id": str(cell["run_id"]),
                "readiness_kind": str(cell.get("readiness_kind", "")),
            }
            (run_dir / "oracle-applicability.json").write_text(
                json.dumps(oracle_verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            oracle_status = "not_applicable"
        else:
            oracle_verdict = self.bundle_executor(self.oracle_bundle, run_dir, result.run_artifact)
            if oracle_verdict.get("status") != "passed" and oracle_verdict.get("oracle", {}).get("status") != "passed":
                raise RuntimeHalt("oracle bundle failed")
            (run_dir / "oracle.json").write_text(json.dumps(dict(oracle_verdict), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            oracle_status = "passed"
        return replace(
            result, usage=observed, cost=normalized_cost,
            evaluator=dict(evaluator_verdict.get("evaluator", evaluator_verdict)),
            oracle=dict(oracle_verdict.get("oracle", oracle_verdict)), oracle_status=oracle_status,
        )

    def _run_phase(
        self, *, phase: str, records: list[dict[str, Any]],
        topology_evidence: list[Mapping[str, Any]],
    ) -> None:
        cells = self._phase_cells(self.plan, phase)
        expected_count = 3 if phase == "canary" else 15
        if len(cells) != expected_count:
            raise RuntimeHalt(f"{phase} phase has the wrong cell count")
        run_ids = {str(cell["run_id"]) for cell in cells}
        ledger_path = self.root / "runtime" / f"{phase}-invocation-ledger.json"
        ledger = InvocationLedger(
            phase=phase, gateway_relay_lock_hash=self.relay_lock_hash, path=ledger_path,
            epoch=str(self.plan.get("epoch", "")),
        )
        phase_token = secrets.token_urlsafe(32)
        token_expires_at = (datetime.now(UTC) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
        gateway_factory = self.gateway_factory
        def bound_gateway_factory(socket_path: Path, approved: set[str], token: str) -> Any:
            if gateway_factory is None:
                return None
            try:
                return gateway_factory(
                    socket_path, approved, token, token_expires_at, ledger,
                )
            except TypeError:
                return gateway_factory(socket_path, approved, token, token_expires_at)

        handle = self.topology.start(
            phase=phase, run_ids=run_ids,
            gateway_factory=bound_gateway_factory if gateway_factory is not None else None,
            gateway_token=phase_token, token_expires_at=token_expires_at,
            epoch=str(self.plan.get("epoch", "")),
        )
        coordinator = ExperimentRunner(
            artifact_root=self.root / f"coordinator-{phase}", workers=1,
        )
        result_by_id: dict[str, RuntimeCellResult] = {}
        try:
            def execute(cell: Mapping[str, Any], run_dir: Path, labels: Mapping[str, str]) -> CellResult:
                try:
                    result = self._call_executor(cell, run_dir, labels, phase, handle, ledger)
                except Exception as exc:
                    state = ledger.lookup(str(cell["run_id"]))
                    if state["billing_status"] == "unknown":
                        coordinator._halt("billing_unknown", str(exc))
                        raise BillingUnknownError(
                            f"billing unknown for {cell['run_id']} after provider response"
                        ) from exc
                    if state["billing_status"] == "known":
                        raise BillableInvocationError(
                            f"known-billed invocation failed for {cell['run_id']}",
                            cost_usd=float(state["cost_usd"]),
                        ) from exc
                    # No provider response was observed: the failure is a
                    # pre-response infrastructure fault.  Leave the phase
                    # running and let the coordinator retry (max 3) instead
                    # of preemptively halting the whole phase.
                    raise
                with self._result_lock:
                    result_by_id[str(cell["run_id"])] = result
                return CellResult(
                    "completed", cost_usd=float(result.cost["cost_usd"]),
                    artifact=dict(result.run_artifact), billable_model_response=True,
                    model_response_received=True, strict_artifact=bool(self.plan.get("runtime_contract")),
                )

            status = coordinator.execute(
                plan=self.plan, approval=self.approval, signature_path=self.signature_path,
                stage=phase, approval_scope="canary_smoke", public_key=self.public_key,
                executor=execute, eligible_run_ids=run_ids,
            )
            if status.get("halted") or status.get("completed") != expected_count:
                raise RuntimeHalt(f"{phase} phase did not complete exactly {expected_count} cells")
            if set(result_by_id) != run_ids:
                raise RuntimeHalt(f"{phase} phase result IDs do not match approved IDs")
            ledger_path = self.root / "runtime" / f"{phase}-invocation-ledger.json"
            ledger.write(ledger_path)
            ledger_hash = sha256_file(ledger_path)
            for cell in cells:
                result = result_by_id[str(cell["run_id"])]
                run_dir = self.root / "runs" / str(cell["run_id"])
                paths = {
                    "run_artifact": run_dir / "run_artifact.json", "event_ledger": run_dir / "event-ledger.json",
                    "proof": run_dir / "proof.json", "usage": run_dir / "usage.json",
                    "cost": run_dir / "cost.json", "evaluator": run_dir / "evaluator.json",
                    "oracle": run_dir / "oracle.json", "cleanup": run_dir / "cleanup.json",
                }
                if is_readiness_contract(self.plan.get("runtime_contract")):
                    paths["oracle_applicability"] = run_dir / "oracle-applicability.json"
                values = {
                    "event_ledger": result.event_ledger, "proof": result.proof, "usage": result.usage,
                    "cost": result.cost, "evaluator": result.evaluator, "oracle": result.oracle,
                    "cleanup": result.cleanup, "run_artifact": result.run_artifact,
                }
                if is_readiness_contract(self.plan.get("runtime_contract")):
                    values["oracle_applicability"] = result.oracle
                for name, value in values.items():
                    if name == "oracle" and is_readiness_contract(self.plan.get("runtime_contract")):
                        continue
                    paths[name].parent.mkdir(parents=True, exist_ok=True)
                    paths[name].write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
                ledger_rows = [
                    row for row in ledger.snapshot() if row.get("run_id") == str(cell["run_id"])
                ]
                counters = ledger.counter_snapshot(
                    str(cell["run_id"]), epoch=str(self.plan.get("epoch", "")),
                )
                if str(self.plan.get("runtime_contract", "")) in {
                    "veriplanpt-runtime-v0.4.0-r10.8", "veriplanpt-runtime-v0.4.0-r10.10",
                } and counters != {
                    "gateway_request_count": 1,
                    "provider_attempt_count": 1,
                    "provider_response_count": 1,
                }:
                    raise RuntimeHalt(
                        f"r10.7 readiness cell {cell['run_id']} has invalid gateway counters: {counters}"
                    )
                record = {
                    **dict(cell), "status": "passed", "plan_hash": self.plan["plan_hash"],
                    "resource_id": next(p for p in self.profiles if p.logical_label == cell["model_label"]).resource_id,
                    "resource_revision": next(p for p in self.profiles if p.logical_label == cell["model_label"]).resource_revision,
                    "resolution_mode": next(p for p in self.profiles if p.logical_label == cell["model_label"]).resolution_mode,
                    "resolution_evidence_hash": next(p for p in self.profiles if p.logical_label == cell["model_label"]).resolution_evidence_hash,
                    "resolution_resolved_at": next(p for p in self.profiles if p.logical_label == cell["model_label"]).resolution_resolved_at,
                    "artifact_path": str(paths["run_artifact"].relative_to(self.root)), "artifact_sha256": hashes["run_artifact"],
                    "evidence_sha256": hashes["run_artifact"], "artifact_type": "run_artifact",
                    "billing_status": "known", "oracle_status": result.oracle_status, "gateway_relay_lock_hash": self.relay_lock_hash,
                    "runtime_contract": str(self.plan.get("runtime_contract", "")),
                    "invocation_ledger_path": str(ledger_path.relative_to(self.root)), "invocation_ledger_sha256": ledger_hash,
                    "invocation_ledger_hash": ledger_hash,
                    "invocation_call_indices": [int(row["call_index"]) for row in ledger_rows if "call_index" in row],
                    "invocation_replay_count": sum(int(row.get("replay_count", 0)) for row in ledger_rows),
                    "invocation_response_count": len(ledger_rows),
                    **counters,
                    "smoke_id": str(cell["run_id"]) if phase == "smoke" else "",
                }
                evidence_names: tuple[str, ...] = ("event_ledger", "proof", "usage", "cost", "evaluator", "cleanup")
                if is_readiness_contract(self.plan.get("runtime_contract")):
                    evidence_names = (*evidence_names, "oracle_applicability")
                else:
                    evidence_names = (*evidence_names, "oracle")
                for name in evidence_names:
                    record[f"{name}_path"] = str(paths[name].relative_to(self.root))
                    record[f"{name}_sha256"] = hashes[name]
                records.append(record)
        finally:
            shutdown = self.topology.shutdown(handle)
            topology_evidence.append(shutdown)
            if shutdown.get("success") is not True and coordinator.status().get("halt_reason") != "billing_unknown":
                coordinator._halt("topology_cleanup_failure")
            coordinator.close()
            if shutdown.get("success") is not True:
                raise RuntimeHalt(f"{phase} topology cleanup failed")

    def run(self) -> Path:
        self._validate_inputs()
        evidence_path = self.root / "readiness" / "runtime-smoke-evidence.json"
        if evidence_path.exists():
            raise RuntimeHalt("refusing to overwrite existing runtime evidence")
        records: list[dict[str, Any]] = []
        topology_evidence: list[Mapping[str, Any]] = []
        self._run_phase(phase="canary", records=records, topology_evidence=topology_evidence)
        if len(records) != 3:
            raise RuntimeHalt("canary did not produce completed:3")
        self._run_phase(phase="smoke", records=records, topology_evidence=topology_evidence)
        if len(records) != 18:
            raise RuntimeHalt("smoke did not produce completed:15")
        topology_path = self.root / "runtime" / "runtime-topology-evidence.json"
        write_runtime_topology_evidence(
            topology_path, gateway_relay_lock_hash=self.relay_lock_hash,
            phases=topology_evidence,
        )
        topology_hash = sha256_file(topology_path)
        canaries = [record for record in records if execution_kind(record) == "vertex_canary"]
        smokes = [record for record in records if execution_kind(record) == "framework_model_smoke"]
        output = write_runtime_smoke_evidence(
            artifact_root=self.root, dataset_lock_hash=str(canaries[0]["dataset_lock_hash"]),
            dataset_evidence_hash=self.dataset_evidence_hash or str(self.plan.get("dataset_evidence_hash", "0" * 64)),
            canaries=canaries, smokes=smokes, plan_hash=str(self.plan["plan_hash"]),
            training_protocol_hash=self.training_protocol_hash or str(self.plan.get("training_protocol_hash", "0" * 64)),
            baseline_lock_hash=str(canaries[0]["baseline_identity_hash"]),
            model_resolution_lock_hash=str(canaries[0]["model_resolution_lock_hash"]),
            pricing_snapshot_hash=self.pricing_snapshot_hash or str(self.plan.get("pricing_snapshot_hash", "0" * 64)),
            approval_hash=self.approval_hash or hashlib.sha256(json.dumps(self.approval, sort_keys=True).encode()).hexdigest(),
            gateway_relay_lock_hash=self.relay_lock_hash,
            runtime_topology_evidence_path=str(topology_path.relative_to(self.root)),
            runtime_topology_evidence_hash=topology_hash,
        )
        validate_smoke_evidence(
            json.loads(output.read_text(encoding="utf-8")), base_case_ids=[],
            model_labels=[profile.logical_label for profile in self.profiles],
            mode="runtime-smoke", artifact_root=self.root,
        )
        return output
