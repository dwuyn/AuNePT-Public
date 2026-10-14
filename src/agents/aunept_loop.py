"""Budgeted evidence-constrained orchestration loop (Algorithm 2, paper §3.3).

The loop is the only path that dispatches target-facing actions. It consumes the
decision budget one unit per iteration, drains the candidate queue, routes
recoverable outcomes, and admits success only from the evaluator's differential
oracle:

    while b_dec > 0:
        b_dec -= 1
        if Q empty: if G empty -> break, else ResolveEvidenceSubgoals
        c <- Q.PopBest()
        (v, P, r) <- CompileAndGate(c, s, C_A)
        v = recover -> record in M_D, push r into G, continue
        v = reject  -> record in M_D, continue
        (res, b) <- BudgetedDispatch(P, b)
        res = budget-exhausted -> record, break
        M_E.append(P, res); M_W <- Update(M_W, Admit(res, C_A))
        if ExternalEvaluatorVerdict() = success: return verified-success
        cls <- ClassifyOutcome(res, M_W); M_D.record(c, cls); RouteOutcome(cls, c, Q, G, s)
    return budgeted-failure
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.agents.critic import critique
from src.design.contract import RuntimeContract
from src.design.gate import GateDecision, combined_gate
from src.design.state import admit, update
from src.design.subgoals import SubgoalQueue
from src.execution.classify import (
    ROUTE_REPLAN,
    ROUTE_RETRY_ONCE,
    ROUTE_STOP,
    ROUTE_SUBGOAL,
    classify_outcome,
    route_outcome,
)
from src.memory.episodic import EpisodicMemory
from src.memory.world_state import WorldState
from src.pipeline.budget import BudgetExceeded, ResourceBudget
from src.pipeline.ledger import ALLOWED_OUTCOMES, FAILURE_CLASSES, EventLedger
from src.pipeline.queue import CandidateQueue

#: penalty added when ``payload_rejected`` returns a candidate for replanning.
REPLAN_PENALTY_STEP = 0.15

VERIFIED_SUCCESS = "verified_success"
BUDGETED_FAILURE = "budgeted_failure"


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of one budgeted dispatch, as the loop consumes it."""

    text: str = ""
    exit_code: int = 0
    failure_class: str = ""
    tool_calls: int = 1
    observation: Mapping[str, Any] | None = None
    proof: Mapping[str, Any] | None = None


@dataclass
class EpisodeResult:
    """Terminal result of one episode."""

    outcome: str = BUDGETED_FAILURE
    iterations: list[dict[str, Any]] = field(default_factory=list)
    decision_units: int = 0
    tool_calls: int = 0


def _record_decision(decisions: Any, *, phase: str, question: str, chosen: str, reasoning: str) -> None:
    """Append to ``M_D`` when a decision log was supplied."""
    if decisions is None or not hasattr(decisions, "record"):
        return
    from src.memory.decision import Decision

    step = 1
    if hasattr(decisions, "to_list"):
        try:
            step = len(decisions.to_list()) + 1
        except Exception:  # pragma: no cover - defensive
            step = 1
    decisions.record(Decision(step=step, phase=phase, question=question, chosen=chosen,
                              reasoning=reasoning))


def _log(ledger: EventLedger | None, **kwargs: Any) -> None:
    if ledger is None:
        return
    payload = dict(kwargs.get("payload") or {})
    outcome = str(kwargs.get("outcome") or "")
    if outcome and outcome not in ALLOWED_OUTCOMES:
        # The loop's routing vocabulary (approve/recover/classify/…) is telemetry,
        # not a terminal ledger outcome. Preserve it in the payload and keep the
        # append-only ledger inside its validated outcome contract.
        payload.setdefault("loop_outcome", outcome)
        kwargs["outcome"] = ""
    failure_class = str(kwargs.get("failure_class") or "")
    if failure_class and failure_class not in FAILURE_CLASSES:
        # Fine-grained design classes (network/auth/…) route the loop; the ledger
        # keeps its own taxonomy and records the raw class for audit.
        payload.setdefault("loop_failure_class", failure_class)
        kwargs["failure_class"] = "unknown"
    if payload:
        kwargs["payload"] = payload
    ledger.record(**kwargs)


def run_episode(
    *,
    contract: RuntimeContract,
    world_state: WorldState,
    queue: CandidateQueue,
    subgoals: SubgoalQueue,
    budget: ResourceBudget,
    compile_fn: Callable[[Any], Any],
    dispatch: Callable[[Any], DispatchOutcome],
    oracle_verdict: Callable[[Any, DispatchOutcome], bool],
    registry: Mapping[str, Sequence[str]] | None = None,
    memory: EpisodicMemory | None = None,
    decisions: Any | None = None,
    ledger: EventLedger | None = None,
    critique_fn: Callable[[Mapping[str, Any]], Sequence[str]] | None = None,
    critic_llm: Any | None = None,
    resolve_subgoals: Callable[[SubgoalQueue, CandidateQueue], bool] | None = None,
    assess: Callable[[Any], tuple[str, tuple[str, ...]]] | None = None,
    max_decisions: int | None = None,
    artifact_root: str = "",
) -> EpisodeResult:
    """Run Algorithm 2 until the oracle reports success or the budget runs out."""
    result = EpisodeResult()
    decision_limit = int(max_decisions if max_decisions is not None else contract.decision_budget)
    attempts: dict[str, int] = {}

    while result.decision_units < decision_limit:
        result.decision_units += 1

        # ── steps 5–12: candidate selection ──────────────────────────────
        if len(queue) == 0:
            if len(subgoals) == 0:
                break
            progressed = bool(resolve_subgoals(subgoals, queue)) if resolve_subgoals is not None else False
            _log(ledger, phase="execution", stage="subgoals", outcome="resolved" if progressed else "exhausted",
                 detail=f"pending={len(subgoals)}")
            if not progressed:
                break
            if len(queue) == 0:
                continue

        entry = queue.pop_best()
        if entry is None:
            continue
        candidate = entry.candidate
        candidate_id = str(getattr(candidate, "candidate_id", "") or "")

        # ``App(c, M_W)`` and ``G(c) = Prereq(c) \ Satisfied(M_W)`` are functions
        # of the *current* world state, so a queued verdict is re-assessed before
        # the gate: evidence admitted by earlier dispatches must be able to
        # satisfy a prerequisite the candidate was queued with.
        g_missing: tuple[str, ...] = tuple(entry.g_missing or ())
        if assess is not None:
            try:
                verdict, reassessed = assess(candidate)
            except ValueError:
                _log(ledger, phase="execution", stage="gate", candidate_id=candidate_id,
                     outcome="blocked", policy_decision="reject",
                     detail="contradicted by the current world state")
                result.iterations.append({"candidate_id": candidate_id, "decision": "reject",
                                          "detail": "contradicted by the current world state"})
                continue
            entry.verdict = verdict
            entry.g_missing = tuple(reassessed)
            entry.penalty = min(0.15 * len(entry.g_missing), 0.45)
            g_missing = entry.g_missing

        # ── step 13: CompileAndGate ──────────────────────────────────────
        try:
            compiled = compile_fn(candidate)
        except Exception as exc:  # registration/compilation failure is a policy block
            _record_decision(decisions, phase="policy", question=f"dispatch {candidate_id}",
                             chosen="reject", reasoning=f"compile failed: {exc}")
            _log(ledger, phase="execution", stage="compile", candidate_id=candidate_id, outcome="blocked",
                 failure_class="policy_block", policy_decision="reject", detail=str(exc))
            result.iterations.append({"candidate_id": candidate_id, "decision": "reject", "detail": str(exc)})
            continue

        proposal = {
            "candidate_id": candidate_id,
            "template_id": getattr(compiled, "template_id", ""),
            "family": getattr(compiled, "family", ""),
            "argv": list(getattr(compiled, "argv", []) or []),
            "rationale": getattr(candidate, "rationale", ""),
            "authorized_host": contract.target_host,
            "authorized_port": contract.target_port,
            "authorized_families": list(contract.protocol_families or ()),
            "scope": {
                "allowed_hosts": list(contract.allowed_hostnames or ()),
                "allowed_networks": list(contract.allowed_networks or ()),
                "allowed_ports": list(contract.allowed_ports or ()),
            },
            "grounding": "target asserted in M_W from the contract; template registered for the family",
            "missing_prerequisites": list(g_missing),
            # The run's own directory: a procedure may declare a payload artifact
            # (an ``@<path>`` data argument) that the run materialized there.
            "artifact_root": artifact_root,
        }
        findings: Sequence[str] = ()
        if critique_fn is not None:
            findings = critique_fn(proposal)
        elif critic_llm is not None:
            findings = critique(proposal, llm=critic_llm)
        _log(ledger, phase="execution", stage="critique", candidate_id=candidate_id,
             detail=("critic=unavailable" if (critique_fn is None and critic_llm is None)
                     else f"findings={len(findings)}" + (f": {'; '.join(str(f) for f in findings)[:200]}" if findings else "")),
             payload={"critique_mode": "model" if (critique_fn is not None or critic_llm is not None) else "unavailable",
                      "critique_findings": [str(item) for item in findings]})

        gate = combined_gate(
            compiled,
            contract=contract,
            world_state=world_state,
            registry=registry or {},
            h_act=memory.procedure_ids if memory is not None else (),
            budget=budget,
            missing_preconditions=g_missing,
            critique_findings=findings,
            candidate_id=candidate_id,
        )

        # ── steps 14–19: recover / reject ────────────────────────────────
        if gate.decision is GateDecision.RECOVER:
            for subgoal in gate.subgoals:
                subgoals.push(subgoal)
            _record_decision(decisions, phase="policy", question=f"dispatch {candidate_id}",
                             chosen="recover", reasoning="; ".join(gate.reasons))
            _log(ledger, phase="execution", stage="gate", candidate_id=candidate_id, outcome="recover",
                 policy_decision="recover", detail="; ".join(gate.reasons))
            result.iterations.append({"candidate_id": candidate_id, "decision": "recover",
                                      "invariants": list(gate.invariants)})
            continue
        if gate.decision is GateDecision.REJECT:
            _record_decision(decisions, phase="policy", question=f"dispatch {candidate_id}",
                             chosen="reject", reasoning="; ".join(gate.reasons))
            _log(ledger, phase="execution", stage="gate", candidate_id=candidate_id, outcome="blocked",
                 policy_decision="reject", detail="; ".join(gate.reasons))
            result.iterations.append({"candidate_id": candidate_id, "decision": "reject",
                                      "invariants": list(gate.invariants)})
            continue

        _log(ledger, phase="execution", stage="gate", candidate_id=candidate_id, outcome="approved",
             policy_decision="approve", detail=f"severity={gate.severity}")

        # ── step 20: BudgetedDispatch ────────────────────────────────────
        # The tool budget is charged exactly once per external interaction, by
        # whichever executes it: ExecutionGateway for an isolated runtime, the
        # host fallback inside the dispatch adapter otherwise.
        try:
            outcome = dispatch(compiled)
        except BudgetExceeded as exc:
            _record_decision(decisions, phase="execution", question=f"dispatch {candidate_id}",
                             chosen="budget-exhausted", reasoning=str(exc))
            _log(ledger, phase="execution", stage="dispatch", candidate_id=candidate_id,
                 outcome="budget_exhausted", failure_class="budget_exceeded", detail=str(exc))
            result.iterations.append({"candidate_id": candidate_id, "decision": "budget-exhausted"})
            result.outcome = BUDGETED_FAILURE
            return result

        budget.record_attempt(candidate_id)
        result.tool_calls += int(getattr(outcome, "tool_calls", 1) or 0)

        # ── steps 21–23: M_E append, Admit/Update ────────────────────────
        if memory is not None:
            memory.record_procedure(list(getattr(compiled, "argv", []) or []),
                                    str(getattr(outcome, "failure_class", "") or "success"),
                                    candidate_id=candidate_id)
        observation = getattr(outcome, "observation", None)
        if observation:
            update(world_state, admit(dict(observation), contract))

        # A transport error is retried exactly once, in place: the retry is a
        # recovery action, not a fresh candidate selection.
        failure_now = classify_outcome(outcome, world_state)
        retry_now = route_outcome(failure_now, attempts=attempts.get(candidate_id, 0)) == ROUTE_RETRY_ONCE
        if retry_now:
            attempts[candidate_id] = 1
            _log(ledger, phase="execution", stage="dispatch", candidate_id=candidate_id,
                 outcome="transport_error", failure_class=str(getattr(outcome, "failure_class", "") or ""),
                 detail="retrying once")
            try:
                outcome = dispatch(compiled)
            except BudgetExceeded as exc:
                _log(ledger, phase="execution", stage="dispatch", candidate_id=candidate_id,
                     outcome="budget_exhausted", failure_class="budget_exceeded", detail=str(exc))
                result.iterations.append({"candidate_id": candidate_id, "decision": "budget-exhausted"})
                result.outcome = BUDGETED_FAILURE
                return result
            budget.record_attempt(candidate_id)
            result.tool_calls += int(getattr(outcome, "tool_calls", 1) or 0)
            if memory is not None:
                memory.record_procedure(list(getattr(compiled, "argv", []) or []),
                                        str(getattr(outcome, "failure_class", "") or "success"),
                                        candidate_id=candidate_id)
            observation = getattr(outcome, "observation", None)
            if observation:
                update(world_state, admit(dict(observation), contract))

        # ── step 24: evaluator verdict ───────────────────────────────────
        if oracle_verdict(compiled, outcome):
            result.outcome = VERIFIED_SUCCESS
            _record_decision(decisions, phase="execution", question=f"verdict {candidate_id}",
                             chosen="verified-success", reasoning="evaluator admitted the proof")
            _log(ledger, phase="execution", stage="verdict", candidate_id=candidate_id, outcome="success",
                 detail="verified success")
            result.iterations.append({"candidate_id": candidate_id, "decision": "verified-success"})
            return result

        # ── steps 25–27: classify, record, route ─────────────────────────
        outcome_class = classify_outcome(outcome, world_state)
        attempts[candidate_id] = attempts.get(candidate_id, 0) + 1
        route = route_outcome(outcome_class, attempts=attempts[candidate_id])
        _record_decision(decisions, phase="execution", question=f"outcome {candidate_id}",
                         chosen=outcome_class or "none", reasoning=f"route={route}")
        _log(ledger, phase="execution", stage="classify", candidate_id=candidate_id,
             outcome=outcome_class or "none", failure_class=str(getattr(outcome, "failure_class", "") or ""),
             detail=f"route={route}")
        result.iterations.append({"candidate_id": candidate_id, "decision": outcome_class or "none",
                                  "route": route, "compiled": getattr(compiled, "proc_id", "")})

        if route == ROUTE_SUBGOAL:
            from src.design.subgoals import subgoals_from_preconditions

            for subgoal in subgoals_from_preconditions([outcome_class], service_key=candidate_id):
                subgoals.push(subgoal)
            queue.push(candidate, entry.verdict, entry.g_missing)
            continue
        if route == ROUTE_REPLAN:
            pushed = queue.push(candidate, entry.verdict, entry.g_missing, penalty_bonus=REPLAN_PENALTY_STEP)
            result.iterations[-1]["penalty"] = pushed.penalty
            continue
        if route == ROUTE_STOP:
            continue

    result.outcome = BUDGETED_FAILURE
    _log(ledger, phase="execution", stage="episode", outcome=BUDGETED_FAILURE,
         detail=f"decision_units={result.decision_units}")
    return result
