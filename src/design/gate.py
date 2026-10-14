"""Compile and gate (Figure 1, block F) and the evaluator plane.

Figure 1:

    F. Compile and gate
       Resolve_env(M_W, C_A)      -> resolve_env()
       Resolve_proc(rho(c))       -> resolve_proc()
       Policy checks:               -> compile_and_gate()
         registration, binding provenance, scope, replay,
         preconditions, budget
       approve | recover | reject   -> GateDecision

    Evaluator-private plane
       C_E (success oracle)         -> never exposed to the agent
       ExternalEvaluatorVerdict()   -> external_evaluator_verdict()
       {success, continue}
       verified success | budgeted failure -> terminal_outcome()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from src.design.contract import RuntimeContract
from src.design.subgoals import Subgoal, SubgoalKind, subgoals_from_preconditions
from src.execution.placeholders import resolve_placeholder_values
from src.memory.world_state import WorldState
from src.pipeline.budget import BudgetExceeded, ResourceBudget
from src.pipeline.candidates import SUPPORTED_KINDS, ExploitCandidate, evaluate_trust, is_executable
from src.pipeline.evaluator import EvaluatorResult
from src.pipeline.ledger import EventLedger
from src.pipeline.scope import ScopeValidator

#: precondition labels that the figure routes to the subgoal store instead of
#: rejecting the candidate outright.
RECOVERABLE_PRECONDITIONS = (
    "fingerprint_unknown_product",
    "no_fingerprint",
    "auth_required_but_not_met",
    "network_prereq",
    "version_unknown",
    "obtain_version",
)

#: v75 (§3.3) procedural invariants enforced by the pre-dispatch audit.
INVARIANTS: dict[str, str] = {
    "V1": "redundancy — exact replay of an already-dispatched procedure",
    "V2": "grounding — registered template and bindings resolved from M_W/C_A",
    "V3": "network boundary — destination inside the authorized scope",
    "V4": "benign action — contract-declared discovery/handshake step (critic policy)",
    "V5": "precondition — missing session/credential routed to the credential branch",
    "V6": "budget — phase budget not exhausted",
}

#: check name in the gate reasons -> invariant id (used when a check fires).
CHECK_INVARIANTS: dict[str, str] = {
    "registration": "V2",
    "binding_provenance": "V2",
    "scope": "V3",
    "replay": "V1",
    "preconditions": "V5",
    "budget": "V6",
}

#: invariants evaluated by the deterministic audit (paper §3.3 lists six).
GATE_INVARIANTS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")

#: closed discovery tuple for HTTP-family services (paper §3.2).
DISCOVERY_PATHS: tuple[str, ...] = ("/", "/robots.txt", "/.well-known/")

#: precondition labels a hard audit rejects instead of routing to ``recover``.
HARD_PRECONDITION_PREFIXES: tuple[str, ...] = ("payload_", "transport_", "procedure_", "serialization_")


class GateDecision(str, Enum):
    """Verdicts of the pre-dispatch gate (Figure 1, block F)."""

    APPROVE = "approve"
    RECOVER = "recover"
    REJECT = "reject"


class EvaluatorVerdict(str, Enum):
    """What the evaluator plane exposes to the agent (Figure 1)."""

    SUCCESS = "success"
    CONTINUE = "continue"


class TerminalOutcome(str, Enum):
    VERIFIED_SUCCESS = "verified_success"
    BUDGETED_FAILURE = "budgeted_failure"


@dataclass(frozen=True)
class GateVerdict:
    decision: GateDecision
    reasons: tuple[str, ...] = ()
    subgoals: tuple[Subgoal, ...] = ()
    invariants: tuple[str, ...] = ()
    #: ``info`` | ``warning`` | ``reject`` — contract-declared benign steps are
    #: approved with ``warning``, which resolves the cold-start deadlock.
    severity: str = "info"

    @property
    def approved(self) -> bool:
        return self.decision is GateDecision.APPROVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "subgoals": [item.to_dict() for item in self.subgoals],
            "invariants": list(self.invariants),
            "severity": self.severity,
        }


def resolve_env(
    world_state: WorldState | None,
    contract: RuntimeContract,
    exploit: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """``Resolve_env(M_W, C_A)`` — target-dependent bindings only from state/contract."""
    return resolve_placeholder_values(dict(exploit or {}), dict(state or {}), world_state or WorldState())


def resolve_proc(candidate: ExploitCandidate) -> dict[str, dict[str, Any]]:
    """``Resolve_proc(rho(c))`` — exploit-specific operands from the registered spec."""
    return dict(candidate.bindings or {})


def recoverable_preconditions(missing: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in missing or ():
        text = str(raw or "").strip()
        if text and any(text == item or text.startswith(item) for item in RECOVERABLE_PRECONDITIONS):
            out.append(text)
    return out


def subgoals_for(missing: Iterable[str], *, service_key: str = "") -> tuple[Subgoal, ...]:
    return tuple(subgoals_from_preconditions(list(missing or []), service_key=service_key))


def compile_and_gate(
    candidate: ExploitCandidate,
    *,
    contract: RuntimeContract,
    world_state: WorldState | None = None,
    state: Mapping[str, Any] | None = None,
    ledger: EventLedger | None = None,
    scope_validator: ScopeValidator | None = None,
    budget: ResourceBudget | None = None,
    exploit: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    verifier_approved: bool = False,
    approved_lab_ids: Iterable[str] | None = None,
    missing_preconditions: Iterable[str] | None = None,
    recover_enabled: bool = True,
) -> GateVerdict:
    """Run the six pre-dispatch policy checks and return the gate verdict.

    Order follows Figure 1: registration, binding provenance, scope, replay,
    preconditions, budget. Hard violations reject; recoverable preconditions
    become subgoals when ``recover_enabled``; otherwise the procedure is
    approved.
    """
    reasons: list[str] = []

    # 1. registration — the procedure must be a registered kind and trusted.
    if candidate.kind not in SUPPORTED_KINDS:
        return GateVerdict(GateDecision.REJECT, (f"registration: unsupported kind {candidate.kind!r}",),
                           invariants=(CHECK_INVARIANTS["registration"],))
    if not is_executable(candidate, manifest_approved_lab_ids=approved_lab_ids, verifier_approved=verifier_approved):
        trust = evaluate_trust(candidate, manifest_approved_lab_ids=approved_lab_ids)
        return GateVerdict(GateDecision.REJECT, (f"registration: trust={trust}",),
                           invariants=(CHECK_INVARIANTS["registration"],))
    reasons.append("registration=ok")

    # 2. binding provenance — target bindings must resolve to state or contract.
    values, unresolved = resolve_env(world_state, contract, exploit, state)
    if unresolved:
        return GateVerdict(GateDecision.REJECT, (f"binding_provenance: unresolved={','.join(sorted(unresolved))}",),
                           invariants=(CHECK_INVARIANTS["binding_provenance"],))
    reasons.append(f"bindings={len(values)}")

    # 3. destination scope.
    if scope_validator is not None and argv:
        decision = scope_validator.validate_args(list(argv))
        if not decision.allowed:
            reason = decision.reason or ",".join(decision.blocked_endpoints)
            return GateVerdict(GateDecision.REJECT, (f"scope: {reason}",),
                               invariants=(CHECK_INVARIANTS["scope"],))
        reasons.append("scope=ok")

    # 4. exact replay.
    if ledger is not None and argv:
        rendered = " ".join(str(item) for item in argv)
        if ledger.is_repeated_action(candidate.candidate_id, rendered):
            return GateVerdict(GateDecision.REJECT, ("replay: exact procedure already dispatched",),
                               invariants=(CHECK_INVARIANTS["replay"],))
        reasons.append("replay=ok")

    # 5. hard preconditions.
    missing = [str(item) for item in (missing_preconditions or ()) if str(item).strip()]
    recoverables = recoverable_preconditions(missing)
    hard = [item for item in missing if item not in recoverables]
    if hard:
        return GateVerdict(GateDecision.REJECT, (f"preconditions: {','.join(hard)}",),
                           invariants=(CHECK_INVARIANTS["preconditions"],))
    if recoverables:
        gates = subgoals_for(recoverables, service_key=str(getattr(candidate, "cve_id", "") or ""))
        if not recover_enabled:
            return GateVerdict(GateDecision.REJECT, (f"preconditions: {','.join(recoverables)}",),
                               invariants=(CHECK_INVARIANTS["preconditions"],))
        return GateVerdict(GateDecision.RECOVER, (f"preconditions: {','.join(recoverables)}",), gates,
                           invariants=(CHECK_INVARIANTS["preconditions"],))
    reasons.append("preconditions=ok")

    # 6. remaining budget.
    if budget is not None:
        try:
            budget.check_command()
            budget.check_attempt(candidate.candidate_id)
        except BudgetExceeded as exc:
            return GateVerdict(GateDecision.REJECT, (f"budget: {exc}",),
                               invariants=(CHECK_INVARIANTS["budget"],))
        reasons.append("budget=ok")

    return GateVerdict(GateDecision.APPROVE, tuple(reasons), invariants=GATE_INVARIANTS)


def _fired_invariants(findings: Sequence[str], missing: Sequence[str]) -> tuple[str, ...]:
    """Invariant ids that decided a critic-mapped gate verdict (v75 §3.3)."""
    ids: list[str] = []
    joined = " ".join(findings).lower()
    if "scope" in joined:
        ids.append("V3")
    if findings:
        ids.append("V2")
    for item in missing:
        ids.append("V5" if str(item).startswith("auth_required_but_not_met") else "V2")
    seen: set[str] = set()
    out: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def gate_decision_from_critic(
    *,
    approved: bool,
    missing_preconditions: Iterable[str] | None = None,
    challenges: Iterable[str] | None = None,
    service_key: str = "",
    recover_enabled: bool = False,
) -> GateVerdict:
    """Map a critic verdict onto the figure's gate vocabulary.

    ``recover_enabled=False`` mirrors the current run-time semantics (a critic
    that does not approve blocks the proposal). Setting it to ``True`` adopts
    the figure's ``recover`` semantics: recoverable preconditions are routed to
    the subgoal store instead of blocking dispatch.
    """
    missing = [str(item) for item in (missing_preconditions or ()) if str(item).strip()]
    findings = [str(item) for item in (challenges or ()) if str(item).strip()]
    if approved:
        return GateVerdict(GateDecision.APPROVE, ("critic=approved",))
    reasons: list[str] = [f"challenge: {item}" for item in findings]
    if missing:
        reasons.append(f"preconditions: {','.join(missing)}")
    if not reasons:
        reasons.append("critic=rejected")
    recoverables = recoverable_preconditions(missing)
    hard = [item for item in missing if item not in recoverables]
    fired = _fired_invariants(findings, missing)
    if recover_enabled and recoverables and not hard and not findings:
        return GateVerdict(GateDecision.RECOVER, (f"preconditions: {','.join(recoverables)}",),
                           subgoals_for(recoverables, service_key=service_key), invariants=fired)
    return GateVerdict(GateDecision.REJECT, tuple(reasons), invariants=fired)


#: absolute URLs inside a rendered argv, e.g. ``http://127.0.0.1:9090/path``.
_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://([^/\s]+)")
#: bare dotted-quad hosts appearing as standalone operands.
_HOST_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")


#: Options whose *value* is data the target receives, never a destination: a
#: URL inside a request body or header is payload, not somewhere this procedure
#: contacts (SSRF/JMX/deserialization exploits pass URLs exactly there).
_OPERAND_VALUE_OPTIONS = frozenset({
    "-d", "--data", "--data-binary", "--data-raw", "--data-urlencode", "--json",
    "-H", "--header", "-u", "--user", "-A", "--user-agent", "-e", "--referer",
    "-b", "--cookie", "-o", "--output", "-w", "--write-out", "-F", "--form",
    "-T", "--upload-file", "--post301", "--url-query", "--request-target",
})
#: Options whose value *is* a destination.
_DESTINATION_VALUE_OPTIONS = frozenset({"-x", "--proxy"})


def _collect_destinations(text: str, found: list[str]) -> None:
    """Append the hosts one argv token names, as either a URL or a bare host."""
    for match in _URL_RE.findall(str(text)):
        authority = match.rsplit("@", 1)[-1]
        host = authority.split(":", 1)[0].strip("[]")
        if host and host not in found:
            found.append(host)
    if not _URL_RE.search(str(text)):
        for host in _HOST_RE.findall(str(text)):
            if host not in found:
                found.append(host)


def argv_destinations(argv: Sequence[str]) -> tuple[str, ...]:
    """Hosts a rendered procedure would contact (paper §3.3, invariant V3).

    Only request targets count: option values that carry data (``--data-binary``,
    ``-H``, …) are skipped, so a URL embedded in a payload is not mistaken for a
    destination. ``--proxy`` is the exception — its value *is* a destination.
    """
    found: list[str] = []
    tokens = [str(token) for token in (argv or ())]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if name in _OPERAND_VALUE_OPTIONS:
            index += 1 if "=" in token and token.startswith("--") else 2
            continue
        if name in _DESTINATION_VALUE_OPTIONS:
            if "=" in token and token.startswith("--"):
                _collect_destinations(token.split("=", 1)[1], found)
            elif index + 1 < len(tokens):
                _collect_destinations(tokens[index + 1], found)
            index += 1 if "=" in token and token.startswith("--") else 2
            continue
        _collect_destinations(token, found)
        index += 1
    return tuple(found)


def _audit_grounding(
    compiled: Any,
    contract: RuntimeContract,
    world_state: WorldState | None,
    registry: Mapping[str, Sequence[str]],
) -> str:
    """V2 — the template is registered for the family and the target is in ``M_W``."""
    template_id = str(getattr(compiled, "template_id", "") or "")
    family = str(getattr(compiled, "family", "") or "")
    allowed = tuple(registry.get(family, ()))
    if not allowed or template_id not in allowed:
        return f"grounding: template {template_id!r} is not registered for family {family!r}"
    if world_state is None:
        return "grounding: no world state"
    host = str(contract.target_host or "")
    if not host or not world_state.provenance_of("host", host):
        return f"grounding: target {host!r} is absent from M_W"
    port = contract.target_port
    if port not in (None, "") and not world_state.provenance_of("port", str(port)):
        return f"grounding: port {port!r} is absent from M_W"
    return ""


def _audit_boundary(compiled: Any, contract: RuntimeContract) -> str:
    """V3 — every destination in the rendered argv is inside ``N_auth``."""
    for host in argv_destinations(list(getattr(compiled, "argv", ()) or ())):
        if not contract.allows_host(host):
            return f"boundary: destination {host!r} is outside the authorized scope"
    return ""


def _audit_benign(compiled: Any, contract: RuntimeContract, registry: Mapping[str, Sequence[str]]) -> bool:
    """V4 — a discovery or handshake step on a contract-declared family."""
    family = str(getattr(compiled, "family", "") or "")
    if family not in tuple(contract.protocol_families or ()):
        return False
    template_id = str(getattr(compiled, "template_id", "") or "")
    if family != "http":
        # non-HTTP families are reached by the handshake alone: the single
        # connect/banner step is benign by construction.
        return template_id in tuple(registry.get(family, ()))
    bindings = dict(getattr(compiled, "bindings_proc", {}) or {})
    return template_id == "http-request" and str(bindings.get("PATH") or "") in DISCOVERY_PATHS


def _audit_preconditions(missing: Sequence[str], *, recover_enabled: bool) -> GateVerdict | None:
    """V5 — recoverable prerequisites route to ``G``; hard ones reject."""
    hard = [item for item in missing if str(item).startswith(HARD_PRECONDITION_PREFIXES)]
    if hard:
        return GateVerdict(GateDecision.REJECT, (f"preconditions: {','.join(hard)}",), invariants=("V5",),
                           severity="reject")
    recoverables = recoverable_preconditions(missing)
    if recoverables and recover_enabled:
        gates = subgoals_for(recoverables)
        return GateVerdict(GateDecision.RECOVER, (f"preconditions: {','.join(recoverables)}",), gates,
                           invariants=("V5",), severity="warning")
    return None


def audit_procedure(
    compiled: Any,
    *,
    contract: RuntimeContract,
    world_state: WorldState | None = None,
    registry: Mapping[str, Sequence[str]] | None = None,
    h_act: Iterable[str] = (),
    budget: ResourceBudget | None = None,
    missing_preconditions: Iterable[str] = (),
    recover_enabled: bool = True,
    candidate_id: str = "",
) -> GateVerdict:
    """The deterministic pre-dispatch audit of the six invariants (paper §3.3).

    ``compiled`` is a compiled procedure (``template_id``, ``family``, ``argv``,
    ``proc_id``). Only ``approve`` reaches live dispatch; a contract-declared
    benign step is approved with severity ``warning``.
    """
    registry_map: Mapping[str, Sequence[str]] = dict(registry or {})
    # V1 — redundancy: the exact procedure was already dispatched.
    proc_id = str(getattr(compiled, "proc_id", "") or "")
    if proc_id and proc_id in {str(item) for item in h_act or ()}:
        return GateVerdict(GateDecision.REJECT, ("redundancy: exact procedure already dispatched",),
                           invariants=("V1",), severity="reject")

    # V2 — grounding: registered template for the family, target present in M_W.
    grounding = _audit_grounding(compiled, contract, world_state, registry_map)
    if grounding:
        return GateVerdict(GateDecision.REJECT, (grounding,), invariants=("V2",), severity="reject")

    # V3 — network boundary.
    boundary = _audit_boundary(compiled, contract)
    if boundary:
        return GateVerdict(GateDecision.REJECT, (boundary,), invariants=("V3",), severity="reject")

    # V4 — benign action (recorded; approval carries severity ``warning``).
    benign = _audit_benign(compiled, contract, registry_map)

    # V5 — preconditions.
    missing = [str(item) for item in missing_preconditions or () if str(item).strip()]
    precondition_verdict = _audit_preconditions(missing, recover_enabled=recover_enabled)
    if precondition_verdict is not None:
        invariants = precondition_verdict.invariants
        if benign:
            invariants = ("V4", *invariants)
        return GateVerdict(
            precondition_verdict.decision,
            precondition_verdict.reasons,
            precondition_verdict.subgoals,
            invariants=invariants,
            severity=precondition_verdict.severity,
        )

    # V6 — remaining budget: phase/candidate/attempt budgets. The tool budget
    # (B_tool) is consumed by ``BudgetedDispatch``, which reports
    # ``budget-exhausted`` and terminates the episode.
    if budget is not None:
        try:
            budget.check_candidate()
            budget.check_attempt(str(candidate_id or getattr(compiled, "candidate_id", "") or "procedure"))
        except BudgetExceeded as exc:
            return GateVerdict(GateDecision.REJECT, (f"budget: {exc}",), invariants=("V6",),
                               severity="reject")

    invariants = ("V4",) if benign else ("V1", "V2", "V3", "V4", "V5", "V6")
    return GateVerdict(
        GateDecision.APPROVE,
        ("audit: all six invariants satisfied",),
        invariants=invariants,
        severity="warning" if benign else "info",
    )


def combined_gate(
    compiled: Any,
    *,
    contract: RuntimeContract,
    world_state: WorldState | None = None,
    registry: Mapping[str, Sequence[str]] | None = None,
    critique_findings: Iterable[str] = (),
    **audit_kwargs: Any,
) -> GateVerdict:
    """Critique-first composition (paper §3.3).

    The model-based critique is collected before the deterministic audit and its
    findings fold into the same ``reject``/``recover`` outcomes: a finding can
    block a proposal but the critique can never authorize one.
    """
    verdict = audit_procedure(compiled, contract=contract, world_state=world_state, registry=registry,
                              **audit_kwargs)
    findings = tuple(str(item) for item in critique_findings or () if str(item).strip())
    if not findings:
        return verdict
    reasons = tuple(f"critique: {item}" for item in findings)
    if verdict.decision is GateDecision.REJECT:
        return GateVerdict(GateDecision.REJECT, (*verdict.reasons, *reasons), verdict.subgoals,
                           invariants=verdict.invariants, severity="reject")
    return GateVerdict(GateDecision.REJECT, (*verdict.reasons, *reasons), verdict.subgoals,
                       invariants=verdict.invariants, severity="reject")
