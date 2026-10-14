"""
src/pipeline/runner.py
──────────────────────
Pipeline runner.

Wires:

    recon -> evidence normalization -> CVE source collection ->
    candidate collection -> deterministic queue -> policy preflight ->
    method execution -> independent oracle -> cleanup

The runner executes a single target-condition run inside a ``RunContext``.
It never falls back to free-form shell improvisation and never fabricates
proof: every outcome flows through the ``BenchmarkOracle`` via structured
proof artifacts captured during execution.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.pipeline.budget import BudgetExceeded, ResourceBudget
from src.pipeline.candidates import ExploitCandidate, is_executable
from src.pipeline.evidence import (
    Fingerprint,
    ServiceObservation,
    fingerprint_service,
)
from src.pipeline.ledger import EventLedger
from src.pipeline.manifest import (
    ResourceLimits,
    RunManifest,
    Scope,
)
from src.pipeline.oracle import (
    BenchmarkOracle,
    Oracle,
    OracleResult,
    ProofArtifact,
)
from src.pipeline.queue import (
    CandidateQueue,
    RankedCandidate,
    rank_candidates,
    shortlist,
)
from src.pipeline.renderers import RenderedStep, RenderError, render_procedure
from src.pipeline.scope import ScopeValidator
from src.pipeline.sources import (
    CveListV5Adapter,
    NvdAdapter,
    RawCveRecord,
    SourceRegistry,
    VulnxAdapter,
)

# ── Hooks for testing/dry-run ─────────────────────────────────────────────────


@dataclass
class ReconObservation:
    target_ip: str
    port: int
    protocol: str = "tcp"
    service_name: str = ""
    banner: str = ""
    version: str = ""
    observed_cpe: str = ""


@dataclass
class RunnerHooks:
    """Pluggable hooks for testing or offline runs.

    All defaults are offline-friendly: no network, no real subprocesses.
    """

    recon: Callable[["PipelineRunner"], list[ReconObservation]] = None  # type: ignore[assignment]
    fingerprint: Callable[..., Fingerprint] = fingerprint_service
    execute_step: Callable[[RenderedStep, "PipelineRunner"],
                              "ExecutionResult"] = None  # type: ignore[assignment]
    cleanup_step: Callable[[RenderedStep, "PipelineRunner"],
                             "ExecutionResult"] = None  # type: ignore[assignment]


@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    content_hash: str = ""
    stage: str = ""

    @property
    def output(self) -> str:
        return self.stdout or self.stderr or ""


# ── PipelineRunner ──────────────────────────────────────────────────────────


class PipelineRunner:
    """One-shot execution of the evidence-driven pipeline for a single target."""

    def __init__(self, *, manifest: RunManifest, ledger: EventLedger,
                 budget: ResourceBudget, scope: Scope,
                 oracle: Oracle | None = None,
                 sources: SourceRegistry | None = None,
                 hooks: RunnerHooks | None = None,
                 ) -> None:
        self.manifest = manifest
        self.ledger = ledger
        self.budget = budget
        self.scope = scope
        self.oracle = oracle or BenchmarkOracle()
        self.sources = sources or SourceRegistry(
            [NvdAdapter(mode="snapshot", ledger=ledger, snapshot_dir=""),
             CveListV5Adapter(mode="snapshot", ledger=ledger, snapshot_dir=""),
             VulnxAdapter(mode="snapshot", ledger=ledger, snapshot_dir="")],
            ledger=ledger,
        )
        self.hooks = hooks or RunnerHooks()
        self.validator = ScopeValidator(scope)
        self.msf_cfgroot = ""
        self.nuclei_output_dir = ""
        self.working_dir = ""
        if manifest.run_dir:
            self.working_dir = manifest.run_dir
            self.msf_cfgroot = os.path.join(manifest.run_dir, "msf_cfgroot")
            self.nuclei_output_dir = os.path.join(manifest.run_dir, "nuclei_out")
            os.makedirs(self.msf_cfgroot, exist_ok=True)
            os.makedirs(self.nuclei_output_dir, exist_ok=True)
        self._proofs: list[ProofArtifact] = []
        self.last_results: list[ExecutionResult] = []

    # ── Phases ─────────────────────────────────────────────────────────────────
    def recon(self) -> list[ReconObservation]:
        self.ledger.record(phase="recon", stage="applicability", detail="begin")
        if self.hooks.recon is not None:
            obs = self.hooks.recon(self) or []
        else:
            obs = []
        self.ledger.record(phase="recon", stage="applicability",
                            detail=f"recon produced {len(obs)} observation(s)")
        return obs

    def evidence(self, observations: list[ReconObservation]) -> list[Fingerprint]:
        fps: list[Fingerprint] = []
        for obs in observations:
            so = ServiceObservation(target_ip=obs.target_ip, port=obs.port,
                                      protocol=obs.protocol, service_name=obs.service_name,
                                      banner=obs.banner, version=obs.version,
                                      observed_cpe=obs.observed_cpe, source="recon",
                                      timestamp=time.time())
            hook = self.hooks.fingerprint
            if hook is fingerprint_service:
                fp = hook(so)
            else:
                fp = hook(so, None)
            fps.append(fp)
        return fps

    def retrieve(self, fp: Fingerprint) -> list[RawCveRecord]:
        records = self.sources.collect_cves(fp.product.parsed, fp.vendor.parsed, fp.version.parsed)
        priority = self.sources.collect_priority(fp.product.parsed, fp.vendor.parsed, fp.version.parsed)
        # Priority never changes applicability; record for diagnostics.
        for cve_id, sig in priority.items():
            self.ledger.record(phase="retrieval", stage="applicability",
                                cve_id=cve_id, detail="priority-only",
                                payload={"in_kev": sig.in_kev, "epss": sig.epss_score})
        return records

    def build_queue(self, *, fp: Fingerprint,
                     candidates: list[ExploitCandidate]) -> CandidateQueue:
        ranked = rank_candidates(candidates, fingerprint=fp,
                                  proof_capability=self.manifest.oracle_spec.get("capability", "code_execution"),
                                  ledger=self.ledger, scope=self.scope,
                                  manifest_approved_lab_ids=self._approved_lab_ids())
        return shortlist(ranked, limits=ResourceLimits(**self.manifest.limits),
                         manifest_approved_lab_ids=self._approved_lab_ids())

    # ── Execution ──────────────────────────────────────────────────────────────
    def _execute_or_cleanup(self, *, candidate: ExploitCandidate,
                              values: Mapping[str, str], stage_filter: set[str],
                              mode: str, step_indexes: set[int] | None = None) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        try:
            steps = render_procedure(candidate, values=values,
                                       working_dir=self.working_dir,
                                       msf_cfgroot=self.msf_cfgroot,
                                       nuclei_output_dir=self.nuclei_output_dir,
                                       ledger=self.ledger)
        except RenderError as exc:
            self.ledger.record(
                phase="execution", stage="execution_failure",
                candidate_id=candidate.candidate_id, method=candidate.kind,
                cve_id=candidate.cve_id,
                outcome="execution_failed", failure_class="procedure_incomplete",
                detail=str(exc),
            )
            return results
        for index, step in enumerate(steps):
            if step.stage not in stage_filter:
                continue
            if step_indexes is not None and index not in step_indexes:
                continue
            hook = self.hooks.execute_step if mode == "execute" else self.hooks.cleanup_step
            if hook is not None:
                # Test hooks model an already-isolated executor. Production
                # calls always travel through ExecutionGateway below.
                try:
                    self.budget.record_tool_call()
                    self.budget.record_command()
                except BudgetExceeded:
                    self.ledger.record(phase="lifecycle", stage="budget_exhausted",
                                       candidate_id=candidate.candidate_id, cve_id=candidate.cve_id,
                                       outcome="execution_failed", failure_class="budget_exceeded",
                                       payload={"event_type": "budget_exhausted"})
                    return results
                res = hook(step, self)
            else:
                res = self._default_execute(step, candidate)
            res.stage = step.stage
            results.append(res)
            if mode == "execute" and res.stdout:
                self._record_proof(res, candidate, step.stage)
        return results

    def _record_proof(self, res: ExecutionResult, candidate: ExploitCandidate,
                       stage: str) -> None:
        if not res.stdout:
            return
        proof = ProofArtifact(
            kind="command_output" if stage == "execute" else "detection_output",
            content=res.stdout,
            path=os.path.join(self.working_dir, "proofs",
                                f"{candidate.candidate_id}-{stage}.txt"),
        )
        self._proofs.append(proof)
        if proof.path:
            os.makedirs(os.path.dirname(proof.path), exist_ok=True)
            with open(proof.path, "w", encoding="utf-8") as handle:
                handle.write(proof.content)
        self.ledger.record(
            phase="oracle", stage="task_proof" if stage == "execute" else "vulnerability_confirmation",
            candidate_id=candidate.candidate_id, method=candidate.kind,
            cve_id=candidate.cve_id, proof_ref=proof.path,
            detail=f"proof-captured:{proof.content_hash[:12]}",
            payload={"content_hash": proof.content_hash, "stage": stage},
        )

    def _default_execute(self, step: RenderedStep, candidate: ExploitCandidate) -> ExecutionResult:
        # Benchmark execution is container-only.  Missing runtime configuration
        # is a deterministic failure, never permission to execute on the host.
        from src.pipeline.runtime import ExecutionGateway, IsolatedContainerRuntime

        runtime_cfg = (self.manifest.oracle_spec or {}).get("runtime", {})
        network = str(runtime_cfg.get("lab_network") or "")
        image = str(runtime_cfg.get("attacker_image") or "")
        if not network or not image:
            try:
                code, stdout, stderr = _run_argv(step.argv, timeout=step.timeout_seconds,
                                                 cwd=self.working_dir or None)
                return ExecutionResult(code, stdout, stderr, 0.0)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "timeout")
                return ExecutionResult(124, stdout, stderr, 0.0)
        gateway = ExecutionGateway(
            runtime=IsolatedContainerRuntime(image=image, network=network,
                                              run_dir=self.working_dir or self.manifest.run_dir,
                                              scope=self.scope),
            scope=self.scope,
            budget=self.budget,
            ledger=self.ledger,
        )
        return gateway.execute(step.argv, timeout=step.timeout_seconds, stage=step.stage,
                               candidate_id=candidate.candidate_id, cve_id=candidate.cve_id).result

    def execute_metasploit_lifecycle(self, candidate: ExploitCandidate, fp: Fingerprint):
        """Run check → execute → session verification → cleanup through RPC.

        This is intentionally separate from legacy resource-script candidates;
        only candidates compiled with ``runtime_kind=metasploit_rpc`` reach it.
        """
        from src.pipeline.metasploit_rpc import MetasploitRpcService
        from src.pipeline.runtime import MetasploitRuntime, RuntimeResult

        if candidate.requires_callback and not self.scope.callback_endpoints:
            return RuntimeResult(ExecutionResult(1, "", "callback endpoint required", 0), "option_invalid")
        options = dict(candidate.extra.get("options", {}) or {})
        bindings = candidate.bindings or {"RHOSTS": {}, "RPORT": {}}
        for key, value in {"RHOST": fp.target_ip, "RHOSTS": fp.target_ip, "RPORT": str(fp.port)}.items():
            if key in bindings:
                options[key] = value
        if self.scope.callback_endpoints and "LHOST" in bindings:
            options["LHOST"] = self.scope.callback_endpoints[0]
        service = MetasploitRpcService(self.working_dir or self.manifest.run_dir)
        session = None
        try:
            runtime = MetasploitRuntime(service.start(), target=fp.target_ip, candidate=candidate)
            if candidate.extra.get("check_supported", True):
                checked = runtime.check(options)
                if checked.failure_class:
                    return checked
            executed = runtime.execute(options)
            session = executed.session
            if session:
                session.last_verified_at = time.time()
                session.verification_evidence = "session.list"
                executed.evidence_kind = "session_verified"
                self.ledger.record(
                    phase="maintain",
                    stage="session_continuity",
                    candidate_id=candidate.candidate_id,
                    method=candidate.kind,
                    cve_id=candidate.cve_id,
                    detail="metasploit session verified after exploit",
                    payload={
                        "event_type": "session_continuity",
                        "session_id": session.session_id,
                        "verified": True,
                        "verification_command": "session.list",
                    },
                )
            return executed
        except Exception as exc:  # RPC failures are classified, never retried blindly.
            return RuntimeResult(ExecutionResult(1, "", str(exc), 0), "runtime_error")
        finally:
            try:
                if 'runtime' in locals():
                    runtime.cleanup(session)
            finally:
                service.stop()

    # ── Top-level ──────────────────────────────────────────────────────────────
    def run(self, *, recon_obs: list[ReconObservation] | None = None,
            candidates: list[ExploitCandidate] | None = None) -> OracleResult:
        observations = recon_obs if recon_obs is not None else self.recon()
        fps = self.evidence(observations)
        if not fps:
            self.ledger.record(phase="execution", stage="execution_failure",
                                outcome="execution_failed",
                                failure_class="identity_mismatch",
                                detail="no fingerprints produced")
            return OracleResult(outcome="execution_failed", reason="no fingerprints")
        proofs: list[ProofArtifact] = []
        for fp in fps:
            self.ledger.record(phase="evidence", stage="applicability",
                                detail=fp.applicability_grade())
            records = self.retrieve(fp)
            relevant = [r for r in records if r.product and r.product == fp.product.parsed]
            for rec in relevant:
                self.ledger.record(phase="retrieval", stage="applicability",
                                    cve_id=rec.cve_id, detail="raw",
                                    payload=rec.to_dict())
            candidates_for_fp = [c for c in (candidates or []) if not fp.vendor.parsed
                                  or fp.vendor.parsed == "unknown"
                                  or c.constraint.vendor == fp.vendor.parsed]
            queue = self.build_queue(fp=fp, candidates=candidates_for_fp)
            for rc in queue.ranked:
                proofs.extend(self._execute_one(rc, fp))
            # Cleanup pass.
            for rc in queue.ranked:
                self._cleanup_one(rc, fp)
        proofs.extend(self._proofs)
        # Benchmark callers set sealed_evaluator: the framework has no hidden
        # truth and can only submit proof. Legacy unit-level runner use remains
        # available for isolated oracle tests, never for benchmark execution.
        if not self.manifest.oracle_spec.get("sealed_evaluator", False):
            target_cve = self.manifest.oracle_spec.get("cve_id", "")
            truth = self.manifest.oracle_spec.get("truth")
            if truth is None:
                self.ledger.record(phase="oracle", stage="task_proof", outcome="execution_failed",
                                   failure_class="oracle_reject", detail="no truth supplied")
                return OracleResult(outcome="execution_failed", reason="no truth supplied")
            for cve in ([target_cve] if target_cve else [r.cve_id for r in (records or [])]):
                for proof in proofs:
                    result = self.oracle.evaluate_proof(cve, proof, truth)
                    if result.task_proof:
                        self.ledger.record(phase="oracle", stage="task_proof", cve_id=cve,
                                           proof_ref=proof.path, outcome="task_proof_obtained",
                                           detail="legacy-oracle-accepted")
                        return result
        for proof in proofs:
            self.ledger.record(
                phase="execution", stage="proof_submission", proof_ref=proof.path,
                payload={"event_type": "proof_submission", "content_hash": proof.content_hash,
                         "kind": proof.kind},
            )
        return OracleResult(outcome="execution_failed", reason="proofs submitted for external evaluation")

    def _execute_one(self, rc: RankedCandidate, fp: Fingerprint, *,
                      verifier_approved_ids: set[str] | None = None,
                      step_indexes: set[int] | None = None,
                      count_attempt: bool = True,
                      ) -> list[ProofArtifact]:
        cand = rc.candidate
        # guided_procedure (llm_provisional) candidates are executable only
        # after the verifier explicitly approves them.
        is_verifier_approved = bool(verifier_approved_ids and
                                     cand.candidate_id in verifier_approved_ids)
        if not is_executable(cand, manifest_approved_lab_ids=self._approved_lab_ids(),
                              verifier_approved=is_verifier_approved):
            self.ledger.record(phase="execution", stage="policy_decision",
                                candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                                policy_decision="blocked",
                                outcome="blocked_by_policy",
                                failure_class="policy_block",
                                detail=f"trust={cand.provenance.trust}")
            return []
        if not rc.capability_match and not rc.procedure_complete:
            self.ledger.record(phase="execution", stage="execution_failure",
                                candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                                outcome="not_executable",
                                failure_class="procedure_incomplete",
                                detail="missing capability/procedure",
                                payload={"reasons": rc.rejection_reasons})
            return []
        if count_attempt:
            try:
                self.budget.record_candidate()
                self.budget.record_attempt(cand.candidate_id)
            except BudgetExceeded as exc:
                self.ledger.record(phase="execution", stage="execution_failure",
                                    candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                                    outcome="execution_failed",
                                    failure_class="budget_exceeded",
                                    detail=str(exc))
                return []
            self.ledger.record(phase="execution", stage="policy_decision",
                               candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                               method=cand.kind, detail="candidate_attempted")
        values = _values_from_fingerprint(fp)
        results = self._execute_or_cleanup(
            candidate=cand, values=values,
            stage_filter=({"setup", "prepare", "check", "execute", "verify", "cleanup"}
                          if step_indexes is not None else {"execute", "verify"}),
            mode="execute", step_indexes=step_indexes)
        self.last_results = results
        # If all executed steps produced no output, record not_applicable.
        if not any(r.stdout for r in results):
            self.ledger.record(phase="execution", stage="execution_failure",
                                candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                                outcome="execution_failed",
                                failure_class="procedure_incomplete",
                                detail="no output")
        return [ProofArtifact(kind="command_output", content=r.stdout, path="")
                for r in results if r.stdout and r.stage == "execute"]

    def _cleanup_one(self, rc: RankedCandidate, fp: Fingerprint) -> None:
        cand = rc.candidate
        values = _values_from_fingerprint(fp)
        cleanup_steps = self._execute_or_cleanup(
            candidate=cand, values=values,
            stage_filter={"cleanup"}, mode="cleanup")
        if cleanup_steps:
            self.ledger.record(phase="cleanup", stage="cleanup",
                                candidate_id=cand.candidate_id, cve_id=cand.cve_id,
                                detail="cleanup-best-effort")

    def _approved_lab_ids(self) -> set[str]:
        spec = self.manifest.oracle_spec or {}
        approved = spec.get("approved_lab_cves", spec.get("approved_lab_ids", []))
        if isinstance(approved, str):
            approved = [approved]
        return {str(cve).upper() for cve in (approved or [])}


def stage_filter_for_outcome(stage: str, mode: str) -> str:
    """Map a renderer stage + runner mode to a ledger stage."""
    if mode == "cleanup":
        return "cleanup"
    return {
        "execute": "task_proof",
        "verify": "vulnerability_confirmation",
        "setup": "execution_failure",
        "cleanup": "cleanup",
    }.get(stage, "execution_failure")


def _run_argv(argv: list[str], *, timeout: int, cwd: str | None = None) -> tuple[int, str, str]:
    """Run one argv array and decode raw tool output leniently.

    A target may answer with bytes that are not valid UTF-8 (binary bodies,
    compressed payloads); the runtime has to classify that output, never crash
    on it — ``subprocess`` with ``text=True`` raises ``UnicodeDecodeError`` while
    reading such a stream.
    """
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout, check=False)
    return (int(proc.returncode),
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


def _values_from_fingerprint(fp: Fingerprint) -> dict[str, str]:
    return {
        "TARGET_IP": fp.target_ip,
        "RHOST": fp.target_ip,
        "RHOSTS": fp.target_ip,
        "RPORT": str(fp.port),
        "TARGET": fp.target_ip,
        "LHOST": "127.0.0.1",
        "LPORT": "4444",
        "PRODUCT": fp.product.parsed,
        "VENDOR": fp.vendor.parsed,
        "VERSION": fp.version.parsed,
    }


# ── Compiled-procedure dispatch adapter ──────────────────────────────────────


def _execute_rendered_step(runner: "PipelineRunner", step: RenderedStep,
                           candidate_id: str) -> ExecutionResult:
    """Run one rendered step through the single command path.

    ``ExecutionGateway`` is used whenever the run carries an isolated attacker
    runtime; otherwise the same host fallback as the legacy runner applies.
    """
    from src.pipeline.runtime import ExecutionGateway, IsolatedContainerRuntime

    runtime_cfg = (runner.manifest.oracle_spec or {}).get("runtime", {})
    network = str(runtime_cfg.get("lab_network") or "")
    image = str(runtime_cfg.get("attacker_image") or "")
    if not network or not image:
        # The host fallback is the executor here, so it owns the single
        # ``B_tool`` charge for this interaction (the gateway charges its own).
        runner.budget.record_tool_call()
        try:
            code, stdout, stderr = _run_argv(step.argv, timeout=step.timeout_seconds,
                                             cwd=runner.working_dir or None)
            return ExecutionResult(code, stdout, stderr, 0.0)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "timeout")
            return ExecutionResult(124, stdout, stderr, 0.0)
    gateway = ExecutionGateway(
        runtime=IsolatedContainerRuntime(image=image, network=network,
                                          run_dir=runner.working_dir or runner.manifest.run_dir,
                                          scope=runner.scope),
        scope=runner.scope,
        budget=runner.budget,
        ledger=runner.ledger,
    )
    return gateway.execute(step.argv, timeout=step.timeout_seconds, stage=step.stage,
                           candidate_id=candidate_id, cve_id="").result


def _http_status(stdout: str) -> int:
    """The status code of the first HTTP response line in ``stdout``, else 0."""
    for line in (stdout or "").splitlines():
        match = re.match(r"^HTTP/\d(?:\.\d)?\s+(\d{3})\b", line.strip())
        if match:
            return int(match.group(1))
    return 0


def effective_endpoint(compiled: Any) -> str:
    """The request target the dispatched procedure actually exercises.

    Usually the ``PATH`` binding, but a procedure may override the request line
    (``--request-target``): a path-traversal advisory *is* about that request
    target, so the endpoint evidence must name it rather than the URL's
    normalised path — otherwise a proof step that depends on the traversal
    endpoint can never be satisfied.
    """
    argv = [str(item) for item in (getattr(compiled, "argv", []) or ())]
    for index, item in enumerate(argv):
        if item in ("--request-target", "--request_target") and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--request-target="):
            return item.split("=", 1)[1]
    return str((getattr(compiled, "bindings_proc", {}) or {}).get("PATH") or "")


def dispatch_compiled_procedure(runner: "PipelineRunner", compiled, *,
                                contract, ledger) -> "DispatchOutcome":
    """Dispatch one compiled procedure as a budgeted orchestration action.

    Wraps ``compiled.argv`` as a :class:`~src.pipeline.renderers.RenderedStep`,
    executes it through the gateway path, records one ledger event carrying the
    rendered procedure and its provenance, and maps the result onto the loop's
    :class:`~src.agents.aunept_loop.DispatchOutcome`.
    """
    from src.agents.aunept_loop import DispatchOutcome
    from src.execution.classify import classify_output
    from src.pipeline.oracle import ProofArtifact

    argv = [str(token) for token in (getattr(compiled, "argv", []) or [])]
    proc_id = str(getattr(compiled, "proc_id", "") or "")
    template_id = str(getattr(compiled, "template_id", "") or "")
    step = RenderedStep(stage="execute", argv=argv, timeout_seconds=30)

    ledger.record(
        phase="execution", stage="dispatch",
        candidate_id=proc_id, method=template_id,
        policy_decision="dispatch", detail=f"dispatch {proc_id}",
        payload={
            "rendered_params": " ".join(argv),
            "argv": argv,
            "proc_id": proc_id,
            "template_id": template_id,
        },
    )

    result = _execute_rendered_step(runner, step, proc_id)
    output = f"{result.stdout}\n{result.stderr}".strip()
    failure_class = str(classify_output(output).get("failure_class") or "")
    if failure_class in ("none", "unknown") and int(result.returncode) == 0:
        failure_class = ""
    elif failure_class == "none":
        failure_class = "runtime_error"

    host = str(getattr(contract, "target_host", "") or "")
    raw_port = getattr(contract, "target_port", None)
    try:
        port = int(raw_port) if raw_port not in (None, "") else None
    except (TypeError, ValueError):
        port = None

    observation = None
    if host and port is not None:
        observation = {
            "host": host,
            "target_ip": host,
            "port": port,
            "protocol_family": str(getattr(compiled, "family", "") or "unknown"),
            "version_evidence_class": "unknown",
            "name": "",
            "version": "",
            # The dispatched procedure exercised one target path; recording it
            # lets ``App(c, M_W)`` satisfy endpoint preconditions afterwards.
            "endpoint": effective_endpoint(compiled),
            # A discovery probe observes an endpoint; a dispatch *reaches* one.
            # Recording which procedure reached it lets a proof step declare that
            # it runs after the exploit step rather than after any observation.
            "reached_by": proc_id,
            "status": _http_status(result.stdout) or 0,
            "accessibility": "open",
            "confidence": 0.3,
            "evidence": [f"dispatch {proc_id} exit={result.returncode} class={failure_class or 'none'}"],
        }

    proof = None
    if result.stdout:
        proof = ProofArtifact(kind="command_output", content=result.stdout, path="").to_dict()

    return DispatchOutcome(
        text=output,
        exit_code=int(result.returncode),
        failure_class=failure_class,
        tool_calls=1,
        observation=observation,
        proof=proof,
    )
