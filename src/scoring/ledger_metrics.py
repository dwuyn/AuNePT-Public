"""
src/scoring/ledger_metrics.py
──────────────────────────────
Ledger-based metrics for VeriPlanPT.

All metrics are computed solely from the structured EventLedger.
No heuristic re-parsing of LLM output or agent state.

Primary metric:
    per_run_osr()      — Overall Success Rate (task_proof_obtained)

Secondary metrics:
    ssr()              — Step-wise Success Rate (recon/vuln/exploit/maintain)
    service_accuracy() — Service/product identification accuracy
    correct_cve_at_k() — Correct-CVE@k (k=1,3,5)
    exploit_precision() — Applicability precision
    efficiency()       — Token, request, cost and timing stats
    reliability()      — Invalid command, repeated action, recovery rates
    hallucination_taxonomy() — 4-type deterministic hallucination counts

Structural metrics (paper §3.3 Thm 5, §3.4):
    unauthorised_generation_rate() — ICR_gen over dispatched steps
    http_requests_per_service()    — HTTP requests per service key (bound ≤ 4)
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from src.pipeline.ledger import Event, EventLedger

# ── Data containers ─────────────────────────────────────────────────────────


@dataclass
class SSRResult:
    """Step-wise Success Rate per phase."""
    recon: float = 0.0
    vuln: float = 0.0
    exploit: float = 0.0
    maintain: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "SSR_recon": round(self.recon, 4),
            "SSR_vuln": round(self.vuln, 4),
            "SSR_exploit": round(self.exploit, 4),
            "SSR_maintain": round(self.maintain, 4),
        }


@dataclass
class HallucinationCounts:
    """Deterministic hallucination taxonomy (4 types)."""
    nonexistent_command: int = 0     # command/option that does not exist
    fabricated_cve: int = 0          # CVE not present in retrieval snapshot
    wrong_applicability: int = 0     # applicability claim contradicts hidden truth
    false_success_claim: int = 0     # success claimed without oracle proof

    @property
    def total(self) -> int:
        return (self.nonexistent_command + self.fabricated_cve
                + self.wrong_applicability + self.false_success_claim)

    def to_dict(self) -> dict[str, int]:
        return {
            "hallucination_nonexistent_command": self.nonexistent_command,
            "hallucination_fabricated_cve": self.fabricated_cve,
            "hallucination_wrong_applicability": self.wrong_applicability,
            "hallucination_false_success_claim": self.false_success_claim,
            "hallucination_total": self.total,
        }


@dataclass
class LedgerMetrics:
    """Full metric bundle for one run."""
    run_id: str = ""
    osr: float = 0.0                  # 1.0 = task_proof_obtained, else 0.0
    vulnerability_confirmed: bool = False
    task_proof_obtained: bool = False
    ssr: SSRResult = field(default_factory=SSRResult)
    service_id_correct: bool | None = None  # None = not evaluable
    version_range_correct: bool | None = None
    correct_cve_at_1: bool | None = None
    correct_cve_at_3: bool | None = None
    correct_cve_at_5: bool | None = None
    exploit_applicability_precision: float | None = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_commands: int = 0
    total_usd: float = 0.0
    wall_seconds: float = 0.0
    success_per_million_tokens: float = 0.0
    invalid_command_rate: float = 0.0
    repeated_action_rate: float = 0.0
    recovery_rate: float = 0.0
    false_positive_on_control: bool = False
    scope_violations: int = 0
    hallucination: HallucinationCounts = field(default_factory=HallucinationCounts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "osr": self.osr,
            "vulnerability_confirmed": self.vulnerability_confirmed,
            "task_proof_obtained": self.task_proof_obtained,
            "service_id_correct": self.service_id_correct,
            "version_range_correct": self.version_range_correct,
            "correct_cve_at_1": self.correct_cve_at_1,
            "correct_cve_at_3": self.correct_cve_at_3,
            "correct_cve_at_5": self.correct_cve_at_5,
            "exploit_applicability_precision": self.exploit_applicability_precision,
            "total_input_tokens": self.total_input_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_commands": self.total_commands,
            "total_usd": round(self.total_usd, 6),
            "wall_seconds": round(self.wall_seconds, 2),
            "success_per_million_tokens": round(self.success_per_million_tokens, 6),
            "invalid_command_rate": round(self.invalid_command_rate, 4),
            "repeated_action_rate": round(self.repeated_action_rate, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "false_positive_on_control": self.false_positive_on_control,
            "scope_violations": self.scope_violations,
        }
        d.update(self.ssr.to_dict())
        d.update(self.hallucination.to_dict())
        return d


# ── Metric computation ────────────────────────────────────────────────────────


def compute_metrics(
    ledger: EventLedger,
    *,
    applicable_cves: Sequence[str] | None = None,
    is_patched_control: bool = False,
    snapshot_cve_ids: frozenset[str] | None = None,
) -> LedgerMetrics:
    """Compute all metrics from an EventLedger.

    Args:
        ledger: The run's event ledger (source of truth for all metrics).
        applicable_cves: Ground-truth CVE IDs (from hidden evaluator).  If None,
            CVE accuracy metrics are not computed.
        is_patched_control: True for the fixed-version control targets.
        snapshot_cve_ids: Set of CVE IDs present in the retrieval snapshot.
            Used for hallucination detection.
    """
    events = ledger.events
    metrics = LedgerMetrics(run_id=ledger.run_id)

    # ── OSR / proof ────────────────────────────────────────────────────────────
    for evt in events:
        if evt.outcome == "task_proof_obtained":
            metrics.task_proof_obtained = True
            metrics.osr = 1.0
        if evt.outcome == "vulnerability_confirmed":
            metrics.vulnerability_confirmed = True
    # False-positive check for patched controls.
    if is_patched_control and metrics.task_proof_obtained:
        metrics.false_positive_on_control = True
        metrics.osr = 0.0  # FP does not count as success

    # ── SSR ────────────────────────────────────────────────────────────────────
    {e.phase for e in events if e.phase}
    recon_events = [e for e in events if e.phase == "recon"]
    vuln_events = [e for e in events if e.phase in ("retrieve", "candidates", "queue")]
    exploit_events = [e for e in events if e.phase == "execution"]
    maintain_events = [e for e in events if e.phase == "maintain"]

    metrics.ssr.recon = 1.0 if any(e.stage == "applicability" and not e.failure_class for e in recon_events) else 0.0
    metrics.ssr.vuln = 1.0 if any(e.outcome in ("vulnerability_confirmed", "task_proof_obtained") for e in vuln_events + exploit_events) else 0.0
    metrics.ssr.exploit = 1.0 if metrics.task_proof_obtained else 0.0
    metrics.ssr.maintain = 1.0 if any(e.phase == "maintain" and not e.failure_class for e in maintain_events) else 0.0

    # ── CVE accuracy ───────────────────────────────────────────────────────────
    if applicable_cves is not None:
        ground_truth = set(str(c) for c in applicable_cves if c)
        proposed_cves_ordered: list[str] = []
        for e in events:
            if e.cve_id and e.cve_id not in proposed_cves_ordered:
                proposed_cves_ordered.append(e.cve_id)
        def _hit_at_k(k: int) -> bool:
            return bool(set(proposed_cves_ordered[:k]) & ground_truth)
        metrics.correct_cve_at_1 = _hit_at_k(1)
        metrics.correct_cve_at_3 = _hit_at_k(3)
        metrics.correct_cve_at_5 = _hit_at_k(5)

    # ── Applicability precision ───────────────────────────────────────────────
    exec_events = [e for e in events if e.phase == "execution" and e.candidate_id]
    if exec_events and applicable_cves:
        applicable_set = set(str(c) for c in applicable_cves if c)
        applicable_exec = sum(1 for e in exec_events if e.cve_id in applicable_set)
        metrics.exploit_applicability_precision = applicable_exec / len(exec_events)

    # ── Efficiency ────────────────────────────────────────────────────────────
    for e in events:
        payload = e.payload or {}
        metrics.total_llm_calls += int(payload.get("llm_calls", 0))
        metrics.total_tool_calls += int(payload.get("tool_calls", 0))
        metrics.total_commands += int(payload.get("executed_commands", 0))
        metrics.total_input_tokens += int(payload.get("total_input_tokens", 0))
        metrics.total_cached_tokens += int(payload.get("total_cached_input_tokens", 0))
        metrics.total_output_tokens += int(payload.get("total_output_tokens", 0))
        metrics.total_thinking_tokens += int(payload.get("total_thinking_tokens", 0))
        metrics.total_usd += float(payload.get("total_usd", 0.0))
    metrics.total_tokens = (
        metrics.total_input_tokens + metrics.total_cached_tokens
        + metrics.total_output_tokens + metrics.total_thinking_tokens
    )
    # Wall time from first/last event timestamps.
    timestamps = [e.started_at for e in events if e.started_at]
    if len(timestamps) >= 2:
        metrics.wall_seconds = max(timestamps) - min(timestamps)
    if metrics.total_tokens > 0:
        metrics.success_per_million_tokens = (1.0 if metrics.task_proof_obtained else 0.0) / (metrics.total_tokens / 1_000_000)

    # ── Reliability ───────────────────────────────────────────────────────────
    total_actions = len([e for e in events if e.phase in ("execution", "recon")])
    invalid = sum(1 for e in events if e.failure_class in ("command_invalid", "invalid_command"))
    repeated = sum(1 for e in events if e.failure_class == "repeated_action")
    recovered = sum(1 for e in events if e.detail and "recovery" in str(e.detail).lower())
    failures = sum(1 for e in events if e.outcome == "execution_failed")
    metrics.invalid_command_rate = invalid / max(total_actions, 1)
    metrics.repeated_action_rate = repeated / max(total_actions, 1)
    metrics.recovery_rate = recovered / max(failures, 1) if failures else 0.0

    # ── Scope violations ──────────────────────────────────────────────────────
    metrics.scope_violations = sum(1 for e in events if e.failure_class == "scope_violation")

    # ── Hallucination taxonomy ────────────────────────────────────────────────
    h = metrics.hallucination
    for e in events:
        fc = str(e.failure_class or "")
        detail = str(e.detail or "").lower()
        if fc in ("command_invalid", "invalid_command") or "nonexistent" in detail or "unknown option" in detail:
            h.nonexistent_command += 1
        if fc == "fabricated_cve" or "not in snapshot" in detail:
            h.fabricated_cve += 1
        if fc == "wrong_applicability" or "applicability_mismatch" in detail:
            h.wrong_applicability += 1
        if fc == "false_success" or ("success" in detail and "no oracle" in detail):
            h.false_success_claim += 1

    return metrics


def metrics_from_ledger_path(
    path: str,
    *,
    applicable_cves: Sequence[str] | None = None,
    is_patched_control: bool = False,
    snapshot_cve_ids: frozenset[str] | None = None,
) -> LedgerMetrics:
    """Load a ledger file and compute metrics."""
    ledger = EventLedger.load(path)
    return compute_metrics(
        ledger,
        applicable_cves=applicable_cves,
        is_patched_control=is_patched_control,
        snapshot_cve_ids=snapshot_cve_ids,
    )


# ── Structural metrics (paper §3.3 Thm 5, §3.4) ─────────────────────────────


_HTTP_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _events_of(events: Iterable[Event] | EventLedger) -> list[Event]:
    """Accept either an ``EventLedger`` or any iterable of ``Event``."""
    if isinstance(events, EventLedger):
        return list(events.events)
    return list(events)


def _is_dispatched(event: Event) -> bool:
    """True iff *event* describes a step that actually reached live dispatch.

    Gate-rejected and budget-exhausted steps were never dispatched and are
    excluded from every structural count.
    """
    if event.outcome in {"blocked", "blocked_by_policy", "budget_exhausted"}:
        return False
    if event.failure_class in {"policy_block", "budget_exceeded"}:
        return False
    if str(event.policy_decision or "") == "reject":
        return False
    payload = event.payload or {}
    if event.stage == "dispatch":
        return True
    return isinstance(payload.get("argv"), (list, tuple)) or payload.get("rendered_params") is not None


def _argv_of(event: Event) -> list[str]:
    """The dispatched argv, read from ``payload['argv']`` or ``rendered_params``."""
    payload = event.payload or {}
    argv = payload.get("argv")
    if isinstance(argv, (list, tuple)):
        return [str(item) for item in argv]
    rendered = payload.get("rendered_params")
    if isinstance(rendered, (list, tuple)):
        return [str(item) for item in rendered]
    if isinstance(rendered, str) and rendered.strip():
        try:
            return shlex.split(rendered)
        except ValueError:
            return rendered.split()
    return []


def _family_of(event: Event, registry: Mapping[str, Sequence[str]], template_id: str) -> str:
    """The detected protocol family for a dispatched step, from the payload or registry."""
    payload = event.payload or {}
    declared = str(payload.get("family") or payload.get("protocol_family") or "").strip().lower()
    if declared:
        return declared
    for family, template_ids in registry.items():
        if template_id in tuple(template_ids):
            return str(family)
    return ""


def _template_is_authorised(
    event: Event, registry: Mapping[str, Sequence[str]], template_id: str
) -> bool:
    if not template_id:
        # A dispatch carrying no template identifier cannot be tied to the
        # closed registry; the theorem requires one, so it is unauthorised.
        return False
    family = _family_of(event, registry, template_id)
    if not family:
        return False
    return template_id in tuple(registry.get(family, ()))


def unauthorised_generation_rate(
    events: Iterable[Event] | EventLedger,
    *,
    approved_tools: Iterable[str],
    registry: Mapping[str, Sequence[str]],
) -> float:
    """``ICR_gen`` — fraction of dispatched steps outside the approved generation space.

    A dispatched step is unauthorised when its ``argv[0]`` (basename) is outside
    *approved_tools* or when its ``template_id`` is not among the registered
    templates of its detected family.  Gate-rejected and budget-exhausted steps
    are not dispatched and never enter the denominator; with no dispatched step
    the rate is ``0.0``.
    """
    tools = frozenset(str(tool) for tool in approved_tools)
    dispatched = [event for event in _events_of(events) if _is_dispatched(event)]
    if not dispatched:
        return 0.0

    violations = 0
    for event in dispatched:
        payload = event.payload or {}
        argv = _argv_of(event)
        template_id = str(payload.get("template_id") or "")
        unauthorised = False
        if argv:
            tool = argv[0].rsplit("/", 1)[-1]
            if tool not in tools:
                unauthorised = True
        if not _template_is_authorised(event, registry, template_id):
            unauthorised = True
        if unauthorised:
            violations += 1
    return violations / len(dispatched)


def http_requests_per_service(
    events: Iterable[Event] | EventLedger,
) -> dict[str, int]:
    """Dispatched HTTP-family requests per ``host:port`` service key.

    The paper bounds contract-driven reconnaissance at ``|U_disc| + 1 <= 4``
    requests per HTTP-family service (one connect plus three discovery paths).
    Only HTTP/HTTPS URLs found in a dispatched step's argv are counted; other
    families and undispatched steps are ignored.
    """
    counts: dict[str, int] = {}
    for event in _events_of(events):
        if not _is_dispatched(event):
            continue
        for token in _argv_of(event):
            if "://" not in token:
                continue
            try:
                parts = urlsplit(token)
            except ValueError:
                continue
            scheme = parts.scheme.lower()
            if scheme not in _HTTP_SCHEMES:
                continue
            host = parts.hostname
            if not host:
                continue
            port = parts.port or _DEFAULT_PORTS[scheme]
            key = f"{host}:{port}"
            counts[key] = counts.get(key, 0) + 1
    return counts
