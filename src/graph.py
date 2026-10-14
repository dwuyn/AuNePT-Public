"""
src/graph.py
─────────────
LangGraph StateGraph — the central orchestrator.

Topology (v7 — paper-conformant control flow, paper §3.2–§3.3)
──────────────────────────────────────────────────────────────
    START → recon → pipeline_hypotheses → pipeline_episode →
        pipeline_oracle → END
    pipeline_hypotheses → pipeline_finalize → END   (infrastructure failure)

``recon`` is deterministic and contract-driven; ``pipeline_hypotheses`` builds
the evidence-tiered candidate queue; ``pipeline_episode`` runs the budgeted
orchestration loop (Algorithm 2), which is the only path that dispatches a
target-facing action. Success is admitted only from the evaluator's oracle.

Human-in-the-loop
─────────────────
Fully autonomous mode. Safety is handled by internal quality gates
plus operator-controlled lab environment. No human approval interrupt.

Checkpointing
─────────────
Uses MemorySaver + pickle-to-disk. Resume via thread_id.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import logging
import os
import pickle
import tempfile
import threading
import time
from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.aunept_loop import DispatchOutcome, EpisodeResult, run_episode
from src.agents.critic import critique
from src.agents.recon import recon_node
from src.config import get_config
from src.design import SubgoalQueue, subgoals_from_preconditions
from src.design.contract import RuntimeContract
from src.design.subgoals import SubgoalKind
from src.memory.decision import DecisionMemory
from src.memory.episodic import EpisodicMemory
from src.memory.world_state import WorldState
from src.pipeline.budget import BudgetExceeded, BudgetTier, ResourceBudget
from src.pipeline.candidates import ExploitCandidate, ProcedureStep
from src.pipeline.collectors import (
    ExploitDbSpec,
    MetasploitSpec,
    NativeToolSpec,
    NmapNseSpec,
    NucleiSpec,
    PublicPocSpec,
    VendorRecipeSpec,
    collect_from_records,
    index_exploitdb,
    index_nse_scripts,
)
from src.pipeline.ledger import EventLedger
from src.pipeline.manifest import (
    ResourceLimits,
    RunManifest,
    Scope,
    new_manifest,
)
from src.pipeline.oracle import BenchmarkOracle, OracleResult, ProofArtifact, TargetTruth
from src.pipeline.queue import CandidateQueue
from src.pipeline.runner import (
    PipelineRunner,
    ReconObservation,
    dispatch_compiled_procedure,
)
from src.pipeline.sources import (
    CveListV5Adapter,
    NvdAdapter,
    RawCveRecord,
    SourceRegistry,
    VulnxAdapter,
)
from src.pipeline.vendor_advisories import (
    VendorAdvisoryAdapter,
    resolve_vendor_advisory_path,
)
from src.retrieval.applicability import applicability, contradicted, prerequisites_missing
from src.retrieval.registry import TEMPLATES, compile_procedure
from src.state import PentestState

logger = logging.getLogger(__name__)

# ── Checkpoint directory ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_DIR = os.path.join(
    os.environ.get("VERIPLANPT_RUN_DIR", os.path.join(_ROOT, "data")),
    "checkpoints",
)


def _to_dict(d):
    """Convert nested defaultdicts to regular dicts for pickling."""
    if isinstance(d, dict):
        return {k: _to_dict(v) for k, v in d.items()}
    return d

def _from_dict(d, target_dict):
    """Restore standard dicts back into nested defaultdicts."""
    for k, v in d.items():
        if isinstance(v, dict):
            _from_dict(v, target_dict[k])
        else:
            target_dict[k] = v

class _DiskBackedSaver(MemorySaver):
    """
    Thin wrapper over MemorySaver that also persists to a pickle file.
    This gives resume-from-crash without requiring langgraph-checkpoint-sqlite.
    Thread-safe: uses a lock, deep-copies stores, and writes atomically.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)

                if "storage" in data:
                    _from_dict(data["storage"], self.storage)
                if "writes" in data:
                    _from_dict(data["writes"], self.writes)
                if "blobs" in data and hasattr(self, "blobs"):
                    _from_dict(data["blobs"], self.blobs)

                logger.info("Restored checkpoint from %s", path)
            except Exception as exc:
                logger.warning("Could not load checkpoint %s: %s", path, exc)

    def put(self, config, checkpoint, metadata, new_versions):
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            try:
                data = {
                    "storage": _to_dict(copy.deepcopy(self.storage)),
                    "writes": _to_dict(copy.deepcopy(self.writes)),
                }
                if hasattr(self, "blobs"):
                    data["blobs"] = _to_dict(copy.deepcopy(self.blobs))
            except Exception as exc:
                logger.warning("Could not snapshot checkpoint: %s", exc)
                return result
        try:
            dir_name = os.path.dirname(self._path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(data, f)
                os.replace(tmp_path, self._path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Could not save checkpoint: %s", exc)
        return result


# ── v7 routing functions ─────────────────────────────────────────────────────


def _route_recon_v7(state: PentestState) -> str:
    """After recon: enter hypothesis construction, or stop on a terminal phase."""
    if state.get("current_phase") == "done":
        return "end"
    return "pipeline_hypotheses"


def _route_retrieve(state: PentestState) -> str:
    """Infrastructure retrieval failures terminate through finalize."""
    if state.get("retrieval_status") == "dataset_missing":
        return "pipeline_finalize"
    if state.get("current_phase") == "done":
        return "end"
    return "pipeline_queue"


def _route_hypotheses_v7(state: PentestState) -> str:
    """After hypothesis construction: run the episode, or finalize on infra failure."""
    if state.get("retrieval_status") == "dataset_missing":
        return "pipeline_finalize"
    if state.get("current_phase") == "done":
        return "end"
    return "pipeline_episode"


def _scope_from_state(state: PentestState) -> Scope:
    target = str(state.get("target_ip", "") or "")
    ports: set[int] = set()
    for value in [state.get("target_port")]:
        try:
            if value:
                ports.add(int(value))
        except (TypeError, ValueError):
            pass
    for svc in state.get("target_services", []) or []:
        try:
            ports.add(int(svc.get("port", 0) or 0))
        except (TypeError, ValueError, AttributeError):
            pass
    for port in (state.get("port_services", {}) or {}).keys():
        try:
            ports.add(int(port))
        except (TypeError, ValueError):
            pass
    hostnames: list[str] = []
    networks: list[str] = []
    if target:
        try:
            networks.append(f"{ipaddress.ip_address(target)}/32")
        except ValueError:
            hostnames.append(target)
    callbacks = [str(state.get("attacker_ip") or "")] if state.get("attacker_ip") else []
    return Scope(
        allowed_hostnames=hostnames,
        allowed_networks=networks,
        allowed_ports=sorted(p for p in ports if p),
        allowed_schemes=["http", "https"],
        callback_endpoints=callbacks,
    )


def _evaluator_truth(state: PentestState, manifest: RunManifest | None = None) -> TargetTruth | None:
    """Read private benchmark truth only in the evaluator node."""
    raw = {}
    path = str(state.get("evaluator_truth_path") or "")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    elif manifest is not None:
        raw = (manifest.oracle_spec or {}).get("truth", {})
    if isinstance(raw, TargetTruth):
        return raw
    if isinstance(raw, dict) and raw:
        return TargetTruth.from_dict(raw)
    return None


def _manifest_from_state(state: PentestState) -> RunManifest:
    if state.get("pipeline_manifest"):
        manifest = RunManifest.from_dict(state["pipeline_manifest"])
    else:
        scope = _scope_from_state(state)
        limits = ResourceLimits(max_runtime_seconds=int(state.get("max_runtime_seconds", 1200) or 1200))
        oracle_spec = dict(state.get("oracle_spec", {}) or {})  # optional runtime override
        if state.get("approved_lab_cves"):
            oracle_spec["approved_lab_cves"] = list(state.get("approved_lab_cves") or [])
        manifest = new_manifest(
            str(state.get("target_ip") or "target"),
            variant=str(state.get("variant") or "4"),
            condition=str(state.get("condition") or "clean"),
            scope=scope,
            limits=limits,
            oracle_spec=oracle_spec,
        )
    if not manifest.run_dir:
        manifest.run_dir = os.path.join(_ROOT, "data", "runs", manifest.run_id)
    os.makedirs(manifest.run_dir, exist_ok=True)
    return manifest


def _private_truth_path(manifest: RunManifest) -> str:
    return os.path.join(manifest.run_dir, "evaluator_truth.json")


def _public_manifest(manifest: RunManifest) -> RunManifest:
    public = RunManifest.from_dict(manifest.to_dict())
    public.oracle_spec = {k: v for k, v in public.oracle_spec.items() if k != "truth"}
    return public


def _role_llm(state: PentestState, role: str):
    """Load one configured role model; model_profile selects the experiment condition."""
    try:
        return get_config().get_role_llm(role, str(state.get("model_profile") or ""))
    except Exception as exc:
        if state.get("model_profile"):
            raise RuntimeError(f"Configured model profile cannot be loaded for {role}: {exc}") from exc
        logger.warning("%s model unavailable: %s", role, exc)
        return None


def _role_usage(state: PentestState, role: str, llm, usage: dict | None = None) -> list[dict]:
    """Append one telemetry record for an LLM call.

    *usage* may carry the token breakdown returned by the LLM response:
        input_tokens, cached_input_tokens, output_tokens, thinking_tokens,
        latency_ms, model_revision, usd_cost.
    These are forwarded to the BudgetState so the singleton budget is updated.
    """
    records = list(state.get("role_usage") or [])
    record: dict = {"role": role, "model_loaded": llm is not None}
    if usage:
        record.update(usage)
    records.append(record)

    # Update singleton budget with token usage (B1/B7).
    if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
        budget_state_dict = dict(state.get("budget_state") or {})
        tier = BudgetTier.from_str(str(state.get("budget_tier") or "medium"))
        budget = ResourceBudget.restore(tier.to_limits(), budget_state_dict) if budget_state_dict else ResourceBudget(tier.to_limits())
        try:
            budget.record_llm_usage(
                input_tokens=int(usage.get("input_tokens", 0)),
                cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                thinking_tokens=int(usage.get("thinking_tokens", 0)),
                usd=float(usage.get("usd_cost", 0.0)),
            )
        except BudgetExceeded:
            pass  # budget exceeded is checked separately per node
        # Nodes only return partial state updates, so persist the singleton
        # immediately as well as exposing it in telemetry for audit/replay.
        serialized = budget.state_to_dict()
        state["budget_state"] = serialized
        state["total_input_tokens"] = budget.state.total_input_tokens
        state["total_cached_input_tokens"] = budget.state.total_cached_input_tokens
        state["total_output_tokens"] = budget.state.total_output_tokens
        state["total_thinking_tokens"] = budget.state.total_thinking_tokens
        state["total_tokens"] = budget.state.total_tokens
        state["total_llm_requests"] = budget.state.llm_calls
        state["total_usd"] = budget.state.total_usd
        record["budget_state"] = serialized
    return records


def _save_manifest(manifest: RunManifest) -> None:
    os.makedirs(manifest.run_dir, exist_ok=True)
    with open(os.path.join(manifest.run_dir, "manifest.json"), "w") as fh:
        fh.write(manifest.to_json())


def _ledger_path(state: PentestState, manifest: RunManifest) -> str:
    return str(state.get("pipeline_ledger_path") or os.path.join(manifest.run_dir, "events.jsonl"))


def _open_ledger(state: PentestState, manifest: RunManifest) -> EventLedger:
    path = _ledger_path(state, manifest)
    if os.path.exists(path):
        return EventLedger.resume(path, run_id=manifest.run_id)
    return EventLedger(manifest.run_id, path=path)


def _source_registry(state: PentestState, manifest: RunManifest, ledger: EventLedger) -> SourceRegistry:
    mode = str(state.get("retrieval_mode") or state.get("dataset_mode") or "snapshot")
    if mode not in {"live", "snapshot", "replay"}:
        mode = "snapshot"
    snap = str(state.get("source_snapshot_dir") or "")
    raw_dir = os.path.join(manifest.run_dir, "source_raw")
    adapters: list[Any] = []
    # A vendor-advisory set is part of ``K`` too (paper §3.2): the experiment may
    # mount it when the pinned snapshot cannot supply a case's advisory. It is the
    # vendor's own publication, so for the CVE it names it outranks the mirrored
    # corpus (which may carry a placeholder product or an incomplete range).
    vendor_path = resolve_vendor_advisory_path(str(state.get("vendor_advisories_path") or "") or None)
    if vendor_path is not None and vendor_path.is_file():
        adapters.append(VendorAdvisoryAdapter(path=vendor_path, ledger=ledger))
    adapters += [
        NvdAdapter(mode=mode, snapshot_dir=snap, ledger=ledger, raw_dir=raw_dir),
        CveListV5Adapter(mode=mode, snapshot_dir=snap, ledger=ledger, raw_dir=raw_dir),
    ]
    adapters.append(
        VulnxAdapter(mode=mode, snapshot_dir=snap, ledger=ledger, raw_dir=raw_dir,
                     base_url=str(state.get("vulnx_base_url") or "")))
    return SourceRegistry(adapters, ledger=ledger)


def _runner(state: PentestState, manifest: RunManifest, ledger: EventLedger) -> PipelineRunner:
    limits = ResourceLimits(**manifest.limits)
    scope = Scope.from_dict(manifest.scope)
    serialized_budget = dict(state.get("budget_state") or {})
    return PipelineRunner(
        manifest=manifest,
        ledger=ledger,
        budget=ResourceBudget.restore(limits, serialized_budget) if serialized_budget else ResourceBudget(limits),
        scope=scope,
        sources=_source_registry(state, manifest, ledger),
    )


def _observations_from_state(state: PentestState) -> list[ReconObservation]:
    raw_value = state.get("pipeline_recon_observations") or state.get("recon_observations") or []
    raw = raw_value if isinstance(raw_value, list) else []
    out: list[ReconObservation] = []
    for item in raw:
        if isinstance(item, ReconObservation):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ReconObservation(
                target_ip=str(item.get("target_ip") or state.get("target_ip") or ""),
                port=int(item.get("port", 0) or 0),
                protocol=str(item.get("protocol", "tcp") or "tcp"),
                service_name=str(item.get("service_name") or item.get("name") or ""),
                banner=str(item.get("banner", "") or ""),
                version=str(item.get("version", "") or ""),
                observed_cpe=str(item.get("observed_cpe", "") or ""),
            ))
    seen = {(o.target_ip, o.port, o.service_name, o.observed_cpe) for o in out}
    for svc in state.get("target_services", []) or []:
        try:
            port = int(svc.get("port", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        name = str(svc.get("name") or svc.get("service_name") or "")
        cpe = str(svc.get("observed_cpe") or svc.get("cpe") or "")
        key = (str(state.get("target_ip") or ""), port, name, cpe)
        if port and key not in seen:
            version = str(svc.get("version", "") or "")
            out.append(ReconObservation(
                target_ip=key[0], port=port,
                protocol=str(svc.get("protocol", "tcp") or "tcp"),
                service_name=name,
                banner=str(svc.get("banner") or (f"{name}/{version}" if version else name)),
                version=version,
                observed_cpe=cpe,
            ))
            seen.add(key)
    for port, svc in (state.get("port_services", {}) or {}).items():
        if not isinstance(svc, dict):
            continue
        try:
            p = int(port)
        except (TypeError, ValueError):
            continue
        name = str(svc.get("name") or svc.get("service_name") or "")
        cpe = str(svc.get("cpe") or svc.get("observed_cpe") or "")
        key = (str(state.get("target_ip") or ""), p, name, cpe)
        if p and key not in seen:
            version = str(svc.get("version", "") or "")
            out.append(ReconObservation(
                target_ip=key[0], port=p,
                protocol=str(svc.get("protocol", "tcp") or "tcp"),
                service_name=name,
                banner=str(svc.get("banner") or (f"{name}/{version}" if version else name)),
                version=version,
                observed_cpe=cpe,
            ))
            seen.add(key)
    if not out and state.get("target_port"):
        out.append(ReconObservation(
            target_ip=str(state.get("target_ip") or ""),
            port=int(state.get("target_port") or 0),
            service_name=str(state.get("app_name") or state.get("keyword") or ""),
            version=str(state.get("app_version") or ""),
            observed_cpe=str(state.get("observed_cpe") or ""),
        ))
    return [obs for obs in out if obs.target_ip and obs.port]


def _records_path(manifest: RunManifest) -> str:
    return os.path.join(manifest.run_dir, "source_records.json")


def _proofs_path(manifest: RunManifest) -> str:
    return os.path.join(manifest.run_dir, "proofs.json")


def _write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2, default=str)
    return path


def _load_records(path: str) -> list[RawCveRecord]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return [RawCveRecord.from_dict(item) for item in json.load(fh)]


def _candidate_objs(items) -> list[ExploitCandidate]:
    out: list[ExploitCandidate] = []
    for item in items or []:
        if isinstance(item, ExploitCandidate):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ExploitCandidate.from_dict(item))
    return out


def _candidate_specs_from_state(state: PentestState) -> list:
    specs: list[Any] = []
    for key in ("exploit_candidate_specs", "candidate_specs"):
        raw = state.get(key) or []
        for item in (raw if isinstance(raw, list) else []):
            if not isinstance(item, dict):
                specs.append(item)
                continue
            spec = _spec_from_dict(item)
            if spec is not None:
                specs.append(spec)
    for key in ("exploitdb_dir", "exploitdb_root"):
        root = str(state.get(key) or "")
        if root:
            specs.extend(index_exploitdb(root))
    for key in ("nse_scripts_dir", "nmap_nse_dir"):
        root = str(state.get(key) or "")
        if root:
            specs.extend(index_nse_scripts(root))
    return specs


def _spec_from_dict(item: dict):
    kind = str(item.get("kind") or item.get("type") or "").lower()
    try:
        if kind == "poc":
            return PublicPocSpec(
                cve_id=item["cve_id"], repo=item["repo"], commit=item["commit"],
                entry_point=item["entry_point"],
                capability=item.get("capability", "code_execution"),
                license=item.get("license", "unknown"),
            )
        if kind == "exploitdb":
            return ExploitDbSpec(
                cve_id=item["cve_id"], edb_id=str(item["edb_id"]),
                local_path=item["local_path"],
                capability=item.get("capability", "code_execution"),
                language=item.get("language", "unknown"),
            )
        if kind == "metasploit":
            return MetasploitSpec(
                cve_id=item["cve_id"], module_name=item["module_name"],
                options=dict(item.get("options", {}) or {}),
                rank=item.get("rank", "manual"),
                check_supported=bool(item.get("check_supported", True)),
                capability=item.get("capability", "code_execution"),
            )
        if kind == "nuclei":
            return NucleiSpec(
                cve_id=item["cve_id"], template_id=item["template_id"],
                template_path=item["template_path"],
                classification=item.get("classification", "cve"),
                pinned_commit=item["pinned_commit"],
                capability=item.get("capability", "detection"),
            )
        if kind == "nmap_nse":
            return NmapNseSpec(
                cve_id=item["cve_id"], script_name=item["script_name"],
                script_path=item["script_path"],
                script_args=list(item.get("script_args", []) or []),
                capability=item.get("capability", "detection"),
            )
        if kind == "vendor_recipe":
            return VendorRecipeSpec(
                cve_id=item["cve_id"], vendor=item["vendor"], product=item["product"],
                steps=[ProcedureStep.from_dict(step) for step in item.get("steps", [])],
                references=list(item.get("references", []) or []),
                license=item.get("license", "vendor-documented"),
                capability=item.get("capability", "code_execution"),
                template_id=str(item.get("template_id", "") or ""),
                bindings=dict(item.get("bindings", {}) or {}),
                prerequisites=dict(item.get("prerequisites", {}) or {}),
                extra=dict(item.get("extra", {}) or {}),
            )
        if kind == "native_tool":
            return NativeToolSpec(
                cve_id=item["cve_id"], tool_name=item["tool_name"],
                argv=list(item.get("argv", []) or []),
                capability=item.get("capability", "detection"),
                references=list(item.get("references", []) or []),
                template_id=str(item.get("template_id", "") or ""),
                bindings=dict(item.get("bindings", {}) or {}),
                prerequisites=dict(item.get("prerequisites", {}) or {}),
                extra=dict(item.get("extra", {}) or {}),
            )
    except KeyError:
        return None
    return None


def _proof_objs(path: str) -> list[ProofArtifact]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return [ProofArtifact(**item) for item in json.load(fh)]


def pipeline_prepare_node(state: PentestState) -> dict:
    manifest = _manifest_from_state(state)
    truth = state.get("oracle_truth") or manifest.oracle_spec.get("truth")
    truth_path = ""
    if truth:
        truth_path = _private_truth_path(manifest)
        _write_json(truth_path, truth.to_dict() if isinstance(truth, TargetTruth) else truth)
    manifest = _public_manifest(manifest)
    snapshot = str(state.get("source_snapshot_dir") or "")
    snapshot_hash = str(state.get("source_snapshot_hash") or "")
    if snapshot:
        from src.pipeline.source_snapshot import validate_source_snapshot
        manifest_path = os.path.join(snapshot, "manifest.json")
        legacy_manifest = False
        if os.path.isfile(manifest_path):
            try:
                legacy_manifest = json.loads(open(manifest_path).read()).get("schema") == "veriplanpt-cve-source-snapshot-1.0"
            except (OSError, ValueError, TypeError):
                legacy_manifest = False
        if snapshot_hash or legacy_manifest:
            source_manifest = validate_source_snapshot(snapshot, full=True, expected_hash=snapshot_hash, official=bool(snapshot_hash))
            snapshot_hash = str(source_manifest["snapshot_hash"])
        else:
            digest = hashlib.sha256()
            for root, _, names in os.walk(snapshot):
                for name in sorted(names):
                    with open(os.path.join(root, name), "rb") as handle:
                        digest.update(handle.read())
            snapshot_hash = digest.hexdigest()
        manifest.source_snapshot_hashes = {snapshot: snapshot_hash}
    _save_manifest(manifest)
    ledger = _open_ledger(state, manifest)
    observations = _observations_from_state(state)
    ledger.record(phase="recon", stage="applicability",
                  detail=f"prepared {len(observations)} observation(s)")
    result = dict(state.get("pipeline_result", {}) or {})
    result.update({"run_dir": manifest.run_dir, "observation_count": len(observations)})
    return {
        "current_phase": "pipeline_prepare",
        "pipeline_manifest": manifest.to_dict(),
        "pipeline_ledger_path": _ledger_path(state, manifest),
        "pipeline_result": result,
        "retrieval_mode": str(state.get("retrieval_mode") or "snapshot"),
        "source_snapshot_dir": str(state.get("source_snapshot_dir") or ""),
        "source_snapshot_hash": snapshot_hash,
        "oracle_truth": {},
        "evaluator_truth_path": truth_path,
    }


def pipeline_retrieve_node(state: PentestState) -> dict:
    manifest = _manifest_from_state(state)
    ledger = _open_ledger(state, manifest)
    # B4: fail-fast when snapshot dir is missing or empty
    mode = str(state.get("retrieval_mode") or "snapshot")
    if mode == "snapshot":
        snap = str(state.get("source_snapshot_dir") or "")
        from src.pipeline.source_snapshot import validate_source_snapshot
        try:
            source_manifest = validate_source_snapshot(
                snap, full=True, expected_hash=str(state.get("source_snapshot_hash") or ""),
                official=bool(state.get("source_snapshot_hash")),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            ledger.record(phase="retrieve", stage="applicability", outcome="execution_failed",
                          failure_class="dataset_missing",
                          detail=f"source snapshot validation failed: {exc}")
            result = dict(state.get("pipeline_result", {}) or {})
            result.update({"source_record_count": 0, "retrieval_fail_reason": "dataset_missing"})
            return {
                "current_phase": "pipeline_retrieve",
                "pipeline_result": result,
                "retrieval_status": "dataset_missing",
                "source_snapshot_hash": str(state.get("source_snapshot_hash") or ""),
            }
        state = cast(PentestState, dict(state))
        state["source_snapshot_hash"] = str(source_manifest["snapshot_hash"])
    runner = _runner(state, manifest, ledger)
    observations = _observations_from_state(state)
    fingerprints = runner.evidence(observations)
    records: list[RawCveRecord] = []
    for fp in fingerprints:
        records.extend(runner.retrieve(fp))
    path = _write_json(_records_path(manifest), [r.to_dict() for r in records])
    result = dict(state.get("pipeline_result", {}) or {})
    result.update({
        "fingerprints": [fp.to_dict() for fp in fingerprints],
        "source_records_path": path,
        "source_record_count": len(records),
        "source_record_ids": sorted({r.cve_id for r in records}),
    })
    return {
        "current_phase": "pipeline_retrieve",
        "pipeline_result": result,
        "retrieval_status": "ok" if records else "empty",
    }


def _world_state_from_state(state: PentestState) -> WorldState:
    raw = state.get("world_state")
    if isinstance(raw, WorldState):
        return raw
    return WorldState.from_dict(dict(raw or {}))


def _contract_from_state(state: PentestState) -> RuntimeContract:
    """``C_A`` for the episode: the supplied contract, else one built from the task."""
    raw = state.get("contract")
    if isinstance(raw, RuntimeContract):
        return raw
    public = dict(state.get("public_task") or {})
    if public:
        manifest = state.get("pipeline_manifest") or {}
        limits = manifest.get("limits") if isinstance(manifest, dict) else None
        launch = {
            "container_port": state.get("target_port"),
            "limits": limits,
            "budget_tier": state.get("budget_tier"),
        }
        try:
            from src.pipeline.case_contract import build_contract
            return build_contract(public, launch)
        except (ValueError, TypeError, KeyError):
            pass
    host = str(state.get("target_ip") or "")
    raw_port = state.get("target_port")
    has_port = raw_port not in (None, "")
    port: int | None = None
    if has_port:
        try:
            port = int(raw_port)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            port = None
    return RuntimeContract(
        target_host=host,
        target_port=str(port) if port is not None else None,
        allowed_hostnames=(host,) if host else (),
        allowed_networks=("127.0.0.0/8",),
        allowed_ports=(port,) if port is not None else (),
    )


def _queue_to_list(queue: CandidateQueue) -> list[dict]:
    """Drain the priority queue into its serializable ``QueuedCandidate`` list."""
    items: list[dict] = []
    while len(queue):
        entry = queue.pop_best()
        if entry is None:
            break
        items.append(entry.to_dict())
    return items


def _queue_from_state(state: PentestState) -> CandidateQueue:
    """Rebuild ``Q`` from the serialized queue and the state's candidates."""
    by_id = {c.candidate_id: c for c in _candidate_objs(state.get("exploit_candidates", []))}
    queue = CandidateQueue()
    for item in (state.get("pipeline_result") or {}).get("queue") or []:
        if not isinstance(item, dict):
            continue
        candidate = by_id.get(str(item.get("candidate_id") or ""))
        if candidate is None:
            continue
        queue.push(candidate, str(item.get("verdict") or "unresolved"), item.get("g_missing") or ())
    return queue


def _memory_from_ledger(ledger: EventLedger) -> EpisodicMemory:
    """Rebuild ``M_E``/``H_act`` from the append-only execution ledger.

    The stored ``argv`` list is authoritative; ``rendered_params`` is only a
    fallback for events written before the list was recorded, because an operand
    may itself contain spaces (a JSON body, a WebSocket frame).
    """
    memory = EpisodicMemory()
    for event in ledger.events:
        payload = event.payload or {}
        proc_id = str(payload.get("proc_id") or "")
        if not proc_id:
            continue
        stored = payload.get("argv")
        if isinstance(stored, (list, tuple)) and stored:
            argv = [str(token) for token in stored]
        else:
            rendered = str(payload.get("rendered_params") or "")
            argv = rendered.split(" ") if rendered else []
        memory.record_procedure(argv, event.outcome or "dispatched", candidate_id=event.candidate_id)
    return memory


def pipeline_hypotheses_node(state: PentestState) -> dict:
    """Evidence-tiered hypothesis construction (Algorithm 1, steps 7-16; paper §3.2).

    Runs the existing retrieval path, assesses every candidate with the paper's
    tri-state applicability, and admits product-supported candidates into the
    ``CandidateQueue`` carrying their verdict and ``G(c)``. A candidate whose
    product identity is unsupported is not admissible at this stage: it pushes
    an ``IdentifyProduct`` subgoal instead. Missing version evidence keeps the
    candidate admissible as ``plausible`` and pushes an ``ObtainVersion`` subgoal.
    """
    prepared = pipeline_prepare_node(state)
    working = cast(PentestState, {**state, **prepared})
    retrieved = pipeline_retrieve_node(working)
    working.update(retrieved)

    manifest = _manifest_from_state(working)
    ledger = _open_ledger(working, manifest)
    world_state = _world_state_from_state(working)
    limits = ResourceLimits(**manifest.limits)

    records = _load_records((working.get("pipeline_result") or {}).get("source_records_path")
                            or _records_path(manifest))
    seeded = _candidate_objs(working.get("exploit_candidates", []))
    if records:
        candidates = collect_from_records(
            records,
            specs=_candidate_specs_from_state(working),
            candidates=seeded,
            max_cves=limits.max_cves_per_service,
        )
    else:
        candidates = seeded

    queue = CandidateQueue()
    subgoals = SubgoalQueue.from_list(working.get("subgoals") or [])
    admitted = 0
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id", "") or "")
        if contradicted(candidate, world_state):
            ledger.record(phase="hypotheses", stage="applicability", candidate_id=candidate_id,
                          cve_id=candidate.cve_id, outcome="not_applicable",
                          failure_class="identity_mismatch", detail="contradicted applicability")
            continue
        verdict = applicability(candidate, world_state)
        g_missing = prerequisites_missing(candidate, world_state)
        for subgoal in subgoals_from_preconditions(g_missing, service_key=candidate_id):
            subgoals.push(subgoal)
        if any(str(label).lower().startswith("product:") for label in g_missing):
            ledger.record(phase="hypotheses", stage="applicability", candidate_id=candidate_id,
                          cve_id=candidate.cve_id, outcome="not_executable",
                          detail="product identity unresolved")
            continue
        queue.push(candidate, verdict, g_missing)
        admitted += 1

    result = dict(working.get("pipeline_result", {}) or {})
    result.update({
        "queue": _queue_to_list(queue),
        "subgoals": subgoals.to_list(),
        "hypothesis_count": admitted,
    })
    updates: dict = {k: v for k, v in prepared.items()}
    updates.update(retrieved)
    updates.update({
        "current_phase": "pipeline_hypotheses",
        "pipeline_result": result,
        "subgoals": subgoals.to_list(),
        "hypothesis_count": admitted,
        "exploit_candidates": [c.to_dict() for c in candidates],
    })
    return updates


def pipeline_episode_node(state: PentestState) -> dict:
    """Run one budgeted orchestration episode (Algorithm 2; paper §3.3).

    The loop is the only path that dispatches a target-facing action. Success is
    admitted only from the evaluator's differential oracle; every other terminal
    state is a ``budgeted_failure``.
    """
    manifest = _manifest_from_state(state)
    ledger = _open_ledger(state, manifest)
    contract = _contract_from_state(state)
    world_state = _world_state_from_state(state)
    limits = ResourceLimits(**manifest.limits)
    serialized_budget = dict(state.get("budget_state") or {})
    budget = ResourceBudget.restore(limits, serialized_budget) if serialized_budget else ResourceBudget(limits)

    queue = _queue_from_state(state)
    subgoals = SubgoalQueue.from_list(state.get("subgoals") or [])
    memory = _memory_from_ledger(ledger)
    decisions = DecisionMemory.from_list(list(state.get("decision_memory") or []))

    runner = _runner(state, manifest, ledger)
    runner.budget = budget

    proofs: list[ProofArtifact] = []
    truth = _evaluator_truth(state, manifest)
    oracle = BenchmarkOracle()

    def _dispatch(compiled) -> DispatchOutcome:
        outcome = dispatch_compiled_procedure(runner, compiled, contract=contract, ledger=ledger)
        proof = getattr(outcome, "proof", None)
        if proof:
            try:
                proofs.append(proof if isinstance(proof, ProofArtifact) else ProofArtifact(**dict(proof)))
            except (TypeError, ValueError):
                pass
        return outcome

    def _oracle_verdict(compiled, outcome) -> bool:
        proof = getattr(outcome, "proof", None)
        if truth is None or not proof:
            return False
        try:
            artifact = proof if isinstance(proof, ProofArtifact) else ProofArtifact(**dict(proof))
        except (TypeError, ValueError):
            return False
        cves = [manifest.oracle_spec.get("cve_id")] if manifest.oracle_spec.get("cve_id") else list(truth.applicable_cves)
        for cve_id in [c for c in cves if c]:
            if oracle.evaluate_proof(cve_id, artifact, truth).task_proof:
                return True
        return False

    def _subgoal_resolved(subgoal: Subgoal) -> bool:
        """True when ``M_W`` now carries the evidence the subgoal asked for."""
        services = [svc for host in world_state.hosts.values() for svc in host.services]
        if subgoal.kind is SubgoalKind.IDENTIFY_PRODUCT:
            return any(str(svc.name or "").strip() and str(svc.name).lower() not in {"unknown", "http", "https"}
                       for svc in services)
        if subgoal.kind is SubgoalKind.OBTAIN_VERSION:
            return any(str(svc.version or "").strip()
                       or str(svc.version_evidence_class) in {"explicit", "inferred"} for svc in services)
        label = str(subgoal.detail or "")
        if label.lower().startswith("endpoint:"):
            path = label.split(":", 1)[1].strip()
            declared = {
                str(attrs.get("path") or key)
                for kind, key, attrs, _ in world_state.nodes("endpoint")
            }
            return bool(path) and path in declared
        return False

    def _resolve_subgoals(pending: SubgoalQueue, target_queue: CandidateQueue) -> bool:
        """``ResolveEvidenceSubgoals`` — handle the subgoals ``M_W`` can satisfy.

        Only satisfied subgoals leave ``G``; a subgoal that still lacks evidence
        stays pending, so the loop's ``G``-driven termination stays honest.
        """
        handled = 0
        outstanding = [item for item in pending.pending() if not _subgoal_resolved(item)]
        handled = len(pending.pending()) - len(outstanding)
        if handled:
            while len(pending):
                pending.pop()
            for item in outstanding:
                pending.push(item)

        candidates: list[ExploitCandidate] = []
        for fp in runner.evidence(_observations_from_state(state)):
            candidates.extend(collect_from_records(
                runner.retrieve(fp),
                specs=_candidate_specs_from_state(state),
                candidates=_candidate_objs(state.get("exploit_candidates", [])),
                max_cves=limits.max_cves_per_service,
            ))
        pushed = 0
        for candidate in candidates:
            if contradicted(candidate, world_state):
                continue
            g_missing = prerequisites_missing(candidate, world_state)
            if any(str(label).lower().startswith("product:") for label in g_missing):
                continue
            target_queue.push(candidate, applicability(candidate, world_state), g_missing)
            pushed += 1
        return bool(handled or pushed)

    critic_llm = _role_llm(state, "critic") if state.get("model_profile") else None

    episode: EpisodeResult = run_episode(
        contract=contract,
        world_state=world_state,
        queue=queue,
        subgoals=subgoals,
        budget=budget,
        compile_fn=lambda candidate: compile_procedure(
            candidate, contract=contract, world_state=world_state, state=state),
        dispatch=_dispatch,
        oracle_verdict=_oracle_verdict,
        registry=TEMPLATES,
        memory=memory,
        decisions=decisions,
        ledger=ledger,
        critique_fn=lambda proposal: critique(proposal, llm=critic_llm),
        resolve_subgoals=_resolve_subgoals,
        assess=lambda candidate: (applicability(candidate, world_state),
                                  prerequisites_missing(candidate, world_state)),
        artifact_root=str(manifest.run_dir),
    )

    proof_path = _write_json(_proofs_path(manifest), [p.to_dict() for p in proofs])
    result = dict(state.get("pipeline_result", {}) or {})
    result.update({
        "proofs_path": proof_path,
        "proof_count": len(proofs),
        "terminal_outcome": episode.outcome,
        "decision_units": episode.decision_units,
        "episode_iterations": episode.iterations,
        "queue": _queue_to_list(queue),
        "subgoals": subgoals.to_list(),
    })
    return {
        "current_phase": "pipeline_episode",
        "pipeline_manifest": manifest.to_dict(),
        "pipeline_ledger_path": _ledger_path(state, manifest),
        "pipeline_result": result,
        "terminal_outcome": episode.outcome,
        "decision_units": episode.decision_units,
        "episode_iterations": episode.iterations,
        "subgoals": subgoals.to_list(),
        "budget_state": budget.state_to_dict(),
        "world_state": world_state.to_dict(),
        "episodic_memory": memory.to_list(),
        "decision_memory": decisions.to_list(),
    }


def pipeline_oracle_node(state: PentestState) -> dict:
    manifest = _manifest_from_state(state)
    ledger = _open_ledger(state, manifest)
    truth = _evaluator_truth(state, manifest)
    proofs = _proof_objs((state.get("pipeline_result", {}) or {}).get("proofs_path") or _proofs_path(manifest))
    # B6: Distinguish missing truth from actual execution failure.
    result = OracleResult(outcome="no_truth", reason="no evaluator truth supplied for this run")
    if truth is not None:
        oracle = BenchmarkOracle()
        cves = [manifest.oracle_spec.get("cve_id", "")] if manifest.oracle_spec.get("cve_id") else truth.applicable_cves
        result = OracleResult(outcome="execution_failed", reason="no proof accepted")
        for cve_id in [c for c in cves if c]:
            for proof in proofs:
                candidate = oracle.evaluate_proof(cve_id, proof, truth)
                if candidate.task_proof:
                    result = candidate
                    break
                if candidate.vulnerability_confirmed and not result.vulnerability_confirmed:
                    result = candidate
            if result.task_proof:
                break
    ledger.record(
        phase="oracle", stage="task_proof",
        outcome=result.outcome,
        failure_class="" if result.task_proof or result.vulnerability_confirmed else "oracle_reject",
        detail=result.reason,
        proof_ref=result.proof_artifact.path if result.proof_artifact else "",
        payload={"evidence_used": result.evidence_used},
    )
    manifest.oracle_result = result.to_dict()
    _save_manifest(manifest)
    pipeline_result = dict(state.get("pipeline_result", {}) or {})
    pipeline_result.update({
        "outcome": result.outcome,
        "reason": result.reason,
        "ledger_path": _ledger_path(state, manifest),
        "run_dir": manifest.run_dir,
    })
    return {
        "current_phase": "done",
        "pipeline_manifest": manifest.to_dict(),
        "pipeline_result": pipeline_result,
        "execution_success": result.task_proof,
        "execution_summary": result.reason or result.outcome,
        "run_end_time": time.time(),
    }


def pipeline_finalize_node(state: PentestState) -> dict:
    """Normalize terminal infrastructure/model/budget failures.

    This node is intentionally separate from the oracle.  It does not inspect
    hidden truth and is used when the run cannot validly reach evaluation.
    """
    manifest = _manifest_from_state(state)
    ledger = _open_ledger(state, manifest)
    result = dict(state.get("pipeline_result", {}) or {})
    fail_reason = str(result.get("retrieval_fail_reason") or state.get("retrieval_status") or "")
    if fail_reason == "dataset_missing":
        terminal_status = "infrastructure_failure"
        failure_class = "dataset_missing"
        detail = "required retrieval snapshot/dataset is missing"
    else:
        terminal_status = str(result.get("termination_status") or "infrastructure_failure")
        failure_class = str(result.get("failure_class") or "infrastructure_failure")
        detail = str(result.get("reason") or "run finalized before oracle")
    ledger.record(
        phase="lifecycle",
        stage="termination",
        outcome=terminal_status,
        failure_class=failure_class,
        detail=detail,
    )
    result.update({
        "outcome": terminal_status,
        "termination_status": terminal_status,
        "terminal_causal_class": failure_class,
        "reason": detail,
        "ledger_path": _ledger_path(state, manifest),
        "run_dir": manifest.run_dir,
    })
    _save_manifest(manifest)
    return {
        "current_phase": "done",
        "pipeline_manifest": manifest.to_dict(),
        "pipeline_result": result,
        "execution_success": False,
        "execution_summary": detail,
        "run_end_time": time.time(),
    }


def _build_v7_graph(thread_id: str, *, source_snapshot_dir: str = "", source_snapshot_hash: str = ""):
    """Build the v7 paper-conformant runtime graph (paper §3.2-§3.3).

    Topology::

        recon → pipeline_hypotheses → pipeline_episode → pipeline_oracle → END
        pipeline_hypotheses → pipeline_finalize → END   (infrastructure failure)

    The episode node runs Algorithm 2: candidate selection, ``CompileAndGate``,
    budgeted dispatch, admission through ``Admit → Update``, and oracle-only
    success.  ``pipeline_finalize`` is the terminal node for runs that cannot
    validly reach evaluation (for example a missing source snapshot); it is not
    wired after the oracle because the oracle's verdict must not be overwritten.
    """
    checkpoint_path = os.path.join(_CHECKPOINT_DIR, f"{thread_id}.pkl")
    saver = _DiskBackedSaver(checkpoint_path)

    graph = StateGraph(PentestState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("recon", recon_node)
    graph.add_node("pipeline_hypotheses", pipeline_hypotheses_node)
    graph.add_node("pipeline_episode", pipeline_episode_node)
    graph.add_node("pipeline_oracle", pipeline_oracle_node)
    graph.add_node("pipeline_finalize", pipeline_finalize_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("recon")
    graph.add_conditional_edges("recon", _route_recon_v7, {
        "pipeline_hypotheses": "pipeline_hypotheses",
        "end": END,
    })
    graph.add_conditional_edges("pipeline_hypotheses", _route_hypotheses_v7, {
        "pipeline_episode": "pipeline_episode",
        "pipeline_finalize": "pipeline_finalize",
        "end": END,
    })
    graph.add_edge("pipeline_episode", "pipeline_oracle")
    graph.add_edge("pipeline_oracle", END)
    graph.add_edge("pipeline_finalize", END)

    # ── Compile with disk-backed MemorySaver ──────────────────────────────────
    compiled = graph.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    logger.info("v7 graph compiled. Thread: %s  Checkpoint: %s", thread_id, checkpoint_path)
    return compiled, config


def build_graph(thread_id: str = "default", *, source_snapshot_dir: str = "", source_snapshot_hash: str = ""):
    """Build the paper-conformant runtime graph (v7)."""
    return _build_v7_graph(thread_id, source_snapshot_dir=source_snapshot_dir,
                           source_snapshot_hash=source_snapshot_hash)
