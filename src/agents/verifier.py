"""
src/agents/verifier.py
──────────────────────
Verifier / Critic agent — quality gates that sit between action nodes.

NOT the orchestrator. The orchestrator (graph.py) decides *what runs next*.
The Verifier decides *whether the output quality is sufficient to proceed*.

Three verifier nodes:
  1. recon_verifier    — checks evidence sufficiency after recon
  2. hypothesis_verifier — checks evidence chains after hypothesis generation
  3. execution_verifier  — checks success/failure/repetition after execution

Each verifier:
  - Reads from structured memory (world_state, episodic, decision)
  - Returns verdict: "pass" (go forward) or "block" (go back)
  - Logs its decision to verification_log
  - Can use a cheaper LLM model than the main agents
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config
from src.design.budgets import PhaseBudgets
from src.execution.tracker import TERMINAL_STATUSES
from src.memory.decision import DecisionMemory
from src.memory.episodic import EpisodicMemory
from src.memory.world_state import WorldState
from src.pipeline.budget import BudgetExceeded
from src.pipeline.oracle import OracleResult, TargetTruth
from src.state import PentestState, service_target_key, state_int
from src.utils.json_parser import extract_json
from src.utils.structured_logger import extract_token_usage, get_structured_logger

logger = logging.getLogger(__name__)
slog = get_structured_logger()

# ── Confidence thresholds ─────────────────────────────────────────────────────

RECON_CONFIDENCE_THRESHOLD = 0.5    # min confidence for any service
RECON_MIN_VERSIONED = 1             # need at least 1 service with version string
HYPOTHESIS_MIN_EVIDENCE = 2         # each hypothesis needs >= 2 evidence items
HYPOTHESIS_MIN_CONFIDENCE = 0.3     # minimum hypothesis confidence to pass
MAX_VERIFIER_BLOCKS = 3             # max times verifier can send back per phase


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE BUDGETS (v75 §3.3: K1/K2/K3 + B_max)
# ══════════════════════════════════════════════════════════════════════════════

# Verifier phase name -> the budget field that caps it.
_PHASE_BUDGET_FIELD: Dict[str, str] = {
    "recon": "K1",          # reconnaissance phase
    "hypothesis": "K2",     # hypothesis construction
    "exploit": "K3",        # exploit attempts per candidate
}

# Scope key for the global cost cap, shared by every phase.
_B_MAX_SCOPE = "B_max"


def enforce_phase_budgets(
    budgets: PhaseBudgets,
    *,
    phase: str,
    candidate_id: str = "",
) -> None:
    """Enforce a phase cap and the global ``B_max`` cap for one verifier action.

    Raises :class:`BudgetExceeded` when the phase cap is already reached or when
    the global cost cap ``budgets.B_max`` is exhausted. Each successful call
    consumes one unit of the phase counter and one unit of the global counter;
    counters are monotonic and never replenished.

    ``"recon"`` is capped by K1, ``"hypothesis"`` by K2, ``"exploit"`` by K3.
    ``candidate_id`` scopes the counter (K3 is always per candidate; K1/K2 are
    global to their phase unless a scope id is supplied).
    """
    field = _PHASE_BUDGET_FIELD.get(phase)
    if field is None:
        raise ValueError(f"Unknown verifier phase: {phase!r}")

    scope = f"{field}:{candidate_id}" if candidate_id else field
    cap = int(getattr(budgets, field))
    used = budgets.spent(scope)
    if used >= cap:
        raise BudgetExceeded(field, used, cap)

    total = budgets.spent(_B_MAX_SCOPE)
    if total >= int(budgets.B_max):
        raise BudgetExceeded(_B_MAX_SCOPE, total, int(budgets.B_max))

    budgets.record(scope)
    budgets.record(_B_MAX_SCOPE)


def admit_success(
    oracle_result: OracleResult,
    *,
    truth: "TargetTruth | None" = None,
) -> bool:
    """Admit success from oracle evidence only.

    True iff the oracle reports ``task_proof_obtained`` and the target is not a
    patched control. Model text is never inspected: the oracle verdict and the
    benchmark ground truth are the sole authorities.
    """
    if oracle_result.outcome != "task_proof_obtained":
        return False
    if truth is not None and truth.is_patched_control:
        return False
    return True



# ══════════════════════════════════════════════════════════════════════════════
#  RECON VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

_RECON_VERIFIER_SYSTEM = """You are a penetration testing quality checker.
Given the current world state (discovered services), check:

1. CONSISTENCY: Do the service findings contradict each other? (e.g., two different services on the same port, impossible version numbers)
2. CONFIDENCE: Are the service identifications reliable enough? Consider:
   - Version number present → reliable
   - Only service name, no version → needs deeper probing
   - Banner mismatch → suspicious
3. VERDICT: Is the reconnaissance sufficient to proceed to vulnerability analysis?

Reply in JSON only:
{"consistency": "ok|conflict", "issues": ["list of problems if any"], "verdict": "proceed|continue", "reason": "one line explanation"}
"""


def _check_recon_sufficiency(ws: WorldState) -> tuple[bool, str]:
    """
    Programmatic check (no LLM call needed for the obvious cases).
    Returns (is_sufficient, reason).
    """
    if not ws.hosts:
        return False, "No hosts discovered yet."

    all_services = []
    for host in ws.hosts.values():
        all_services.extend(host.services)

    if not all_services:
        return False, "No services discovered on any host."

    above_threshold = ws.get_services_above_confidence(RECON_CONFIDENCE_THRESHOLD)
    if not above_threshold:
        return False, (
            f"No service has confidence >= {RECON_CONFIDENCE_THRESHOLD}. "
            "Run deeper probes (nmap -sV -sC)."
        )

    versioned = ws.get_versioned_services()
    if len(versioned) < RECON_MIN_VERSIONED:
        return False, (
            f"Only {len(versioned)} service(s) have a version string "
            f"(need >= {RECON_MIN_VERSIONED}). Run version detection scripts."
        )

    return True, (
        f"OK: {len(versioned)} versioned service(s), "
        f"{len(above_threshold)} above confidence threshold."
    )


def _check_repetition(em: EpisodicMemory, phase: str) -> tuple[bool, str]:
    """Check if recent actions in this phase are mostly repeats."""
    recent = em.by_phase(phase)
    if len(recent) < 3:
        return False, ""
    last_3 = recent[-3:]
    repeat_count = sum(1 for e in last_3 if e.was_repeat)
    if repeat_count >= 2:
        return True, "Last 3 actions in this phase are mostly repeats — agent is stuck."
    return False, ""


def recon_verifier_node(state: PentestState) -> Dict[str, Any]:
    """
    Quality gate after recon. Checks:
    1. Are there enough high-confidence services?
    2. Is the agent stuck in a repeat loop?
    3. LLM consistency check (banner conflicts, impossible versions).
    """
    cfg = get_config()
    ws = WorldState.from_dict(state.get("world_state", {}))
    em = EpisodicMemory.from_list(state.get("episodic_memory", []))
    blocks = state_int(state, "recon_verifier_blocks", 0)
    vlog = list(state.get("verification_log", []))
    phase_timestamps = dict(state.get("phase_timestamps", {}))
    step_count = state_int(state, "recon_step_count", 0)
    max_steps = state_int(state, "recon_max_steps", 12)

    acc_tokens_in = state_int(state, "total_tokens_in", 0)
    acc_tokens_out = state_int(state, "total_tokens_out", 0)
    acc_requests = state_int(state, "total_llm_requests", 0)
    host_count = len(ws.hosts)
    service_count = sum(len(host.services) for host in ws.hosts.values())

    logger.info(
        "Recon verifier: starting quality gate for %d host(s) and %d discovered service(s)",
        host_count,
        service_count,
    )

    if step_count >= max_steps:
        sufficient, reason = _check_recon_sufficiency(ws)
        verdict = {
            "verdict": "proceed" if sufficient else "exhausted",
            "reason": (
                f"Recon step cap reached, but minimum evidence threshold is satisfied. {reason}"
                if sufficient else
                f"recon step cap reached before sufficient evidence. {reason}"
            ),
            "phase": "recon",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        payload: dict[str, Any] = {"verification_log": vlog}
        if not sufficient:
            payload["current_phase"] = "done"
        return payload

    # Force forward if we've blocked too many times
    if blocks >= MAX_VERIFIER_BLOCKS:
        verdict = {
            "verdict": "proceed",
            "reason": f"Forced forward after {MAX_VERIFIER_BLOCKS} blocks.",
            "phase": "recon",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.info("Recon verifier: forced proceed after %d blocks", blocks)
        return {"verification_log": vlog}

    # Check if recon is even done
    if not state.get("recon_complete"):
        verdict = {
            "verdict": "continue",
            "reason": "Recon not yet marked complete.",
            "phase": "recon",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        return {"verification_log": vlog}

    # Check sufficiency (programmatic)
    sufficient, reason = _check_recon_sufficiency(ws)

    # Check for repeat loops
    is_stuck, stuck_reason = _check_repetition(em, "recon")
    if is_stuck:
        verdict = {
            "verdict": "proceed",
            "reason": f"Proceeding despite insufficient evidence: {stuck_reason}",
            "phase": "recon",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.warning("Recon verifier: agent stuck, forcing proceed")
        return {"verification_log": vlog}

    if not sufficient:
        verdict = {
            "verdict": "block",
            "reason": reason,
            "phase": "recon",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.info("Recon verifier: BLOCK — %s", reason)
        return {
            "verification_log": vlog,
            "recon_verifier_blocks": blocks + 1,
            "recon_complete": False,
        }

    # ── LLM consistency check (only when programmatic check passes) ───────────
    llm_reason = reason
    llm_issues: list = []

    try:
        verifier_model = cfg.verifier["model"]
        logger.info(
            "Recon verifier: running LLM consistency check with model '%s'",
            verifier_model,
        )
        llm = cfg.get_llm(verifier_model)
        ws_summary = ws.to_summary()
        t0 = time.time()
        response = llm.invoke(
            [
                SystemMessage(content=_RECON_VERIFIER_SYSTEM),
                HumanMessage(content=(
                    f"World state summary:\n{ws_summary}\n\n"
                    "Check for consistency issues in the service findings above."
                )),
            ],
            stream=False,
        )
        tokens_in, tokens_out = extract_token_usage(response)
        acc_tokens_in += tokens_in
        acc_tokens_out += tokens_out
        acc_requests += 1
        slog.node_event(
            "recon_verifier", phase="recon", action="llm_consistency_check",
            tokens_in=tokens_in, tokens_out=tokens_out,
            duration_ms=(time.time() - t0) * 1000,
        )
        parsed = extract_json(response.content or "")
        if isinstance(parsed, dict):
            if parsed.get("consistency") == "conflict":
                llm_issues = parsed.get("issues", [])
                llm_reason = f"LLM detected conflicts: {'; '.join(llm_issues[:3])}"
                logger.warning("Recon verifier (LLM): conflicts detected — %s", llm_reason)
            else:
                llm_reason = parsed.get("reason", reason)
        logger.info(
            "Recon verifier: LLM consistency check completed in %.1fs",
            time.time() - t0,
        )
    except Exception as exc:
        logger.warning("Recon verifier LLM check failed (non-fatal): %s", exc)

    verdict_entry = {
        "verdict": "proceed",
        "reason": llm_reason,
        "phase": "recon",
        "llm_issues": llm_issues,
        "timestamp": time.time(),
    }
    vlog.append(verdict_entry)
    logger.info("Recon verifier: PASS — %s", llm_reason)
    phase_timestamps.setdefault("recon_complete_time", time.time())
    return {
        "verification_log": vlog,
        "total_tokens_in": acc_tokens_in,
        "total_tokens_out": acc_tokens_out,
        "total_tokens": acc_tokens_in + acc_tokens_out,
        "total_llm_requests": acc_requests,
        "phase_timestamps": phase_timestamps,
    }


def route_recon_verifier(state: PentestState) -> str:
    """After recon verifier: back to recon or forward to hypothesis."""
    vlog = state.get("verification_log", [])
    if not vlog:
        return "recon"
    last = vlog[-1]
    if last.get("phase") != "recon":
        return "recon"
    if last.get("verdict") == "exhausted":
        return "end"
    if last.get("verdict") == "proceed":
        return "hypothesis"
    return "recon"


# ══════════════════════════════════════════════════════════════════════════════
#  HYPOTHESIS VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

def hypothesis_verifier_node(state: PentestState) -> Dict[str, Any]:
    """
    Compatibility adapter for the legacy Phase 2 verifier.

    If the modular critic already produced a verdict in retrieval_bundle.critic_report,
    this adapter maps that result into the older proceed/need_more_recon semantics.
    Otherwise it falls back to the previous deterministic verifier behavior.
    """
    get_config()
    hypotheses = state.get("vuln_hypotheses", [])
    retrieval_bundle = state.get("retrieval_bundle", {}) or {}
    vlog = list(state.get("verification_log", []))
    blocks = state.get("hypothesis_verifier_blocks", 0)
    dm = DecisionMemory.from_list(state.get("decision_memory", []))
    latest = dm.get_latest()
    phase_timestamps = dict(state.get("phase_timestamps", {}))
    shortlist = list(retrieval_bundle.get("shortlist", []))
    critic_report = dict(retrieval_bundle.get("critic_report", {}) or {})

    if critic_report:
        critic_verdict = str(critic_report.get("verdict", ""))
        approved_ids = set(critic_report.get("approved_candidate_ids", []))
        filtered_hypotheses = [
            item for item in hypotheses
            if not approved_ids or item.get("candidate_id") in approved_ids
        ]
        if critic_verdict in {"pass", "best_effort_pass"}:
            if latest and latest.phase == "hypothesis":
                dm.mark_outcome(latest.step, "validated")
            verdict = {
                "verdict": "proceed",
                "reason": str(critic_report.get("reason", "Critic approved the shortlist.")),
                "phase": "hypothesis",
                "timestamp": time.time(),
            }
            vlog.append(verdict)
            phase_timestamps.setdefault("hypothesis_complete_time", time.time())
            return {
                "verification_log": vlog,
                "vuln_hypotheses": filtered_hypotheses,
                "decision_memory": dm.to_list(),
                "phase_timestamps": phase_timestamps,
            }
        if critic_verdict == "exhausted":
            if latest and latest.phase == "hypothesis":
                dm.mark_outcome(latest.step, "invalidated")
            verdict = {
                "verdict": "exhausted",
                "reason": str(critic_report.get("reason", "Phase 2 reached its limit.")),
                "phase": "hypothesis",
                "timestamp": time.time(),
            }
            vlog.append(verdict)
            return {
                "verification_log": vlog,
                "decision_memory": dm.to_list(),
                "current_phase": "done",
            }
        if latest and latest.phase == "hypothesis":
            dm.mark_outcome(latest.step, "invalidated")
        verdict = {
            "verdict": "need_more_recon",
            "reason": str(critic_report.get("reason", "Critic requested more recon.")),
            "phase": "hypothesis",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        return {
            "verification_log": vlog,
            "hypothesis_verifier_blocks": blocks + 1,
            "decision_memory": dm.to_list(),
        }

    if blocks >= MAX_VERIFIER_BLOCKS:
        verdict = {
            "verdict": "proceed",
            "reason": f"Forced forward after {MAX_VERIFIER_BLOCKS} blocks.",
            "phase": "hypothesis",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        return {"verification_log": vlog}

    if not hypotheses and not retrieval_bundle.get("shortlist"):
        if latest and latest.phase == "hypothesis":
            dm.mark_outcome(latest.step, "invalidated")
        verdict = {
            "verdict": "need_more_recon",
            "reason": "No vulnerability hypotheses generated. Need more recon data.",
            "phase": "hypothesis",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.info("Hypothesis verifier: BLOCK — no hypotheses")
        return {
            "verification_log": vlog,
            "hypothesis_verifier_blocks": blocks + 1,
            "decision_memory": dm.to_list(),
        }

    assessments = list(retrieval_bundle.get("assessments", []))
    poc_candidates = list(retrieval_bundle.get("poc_candidates", []))
    authoritative = {item.get("cve_id"): item for item in retrieval_bundle.get("authoritative_records", [])}
    poc_by_cve: dict[str, list[dict]] = {}
    for item in poc_candidates:
        poc_by_cve.setdefault(item.get("cve_id", ""), []).append(item)

    if shortlist:
        strong = []
        weak = []
        version_unknown = []
        for item in shortlist:
            cve_id = item.get("cve_id", "")
            candidate_id = item.get("candidate_id", "")
            assessment_item: dict[str, Any] = next(
                (entry for entry in assessments if entry.get("candidate_id") == candidate_id),
                {},
            )
            authority = authoritative.get(cve_id, {})
            source = str(authority.get("source", ""))
            has_authority = source in {"vendor", "kev", "cvemap", "nvd"}
            version_match = assessment_item.get("version_match", "unknown")
            hard_mismatch = "no" in {
                version_match,
                assessment_item.get("cpe_match", "unknown"),
                assessment_item.get("platform_match", "unknown"),
                assessment_item.get("network_match", "unknown"),
            }
            if item.get("source") == "google":
                sibling_sources = {entry.get("source") for entry in poc_by_cve.get(cve_id, []) if entry.get("candidate_id") != candidate_id}
                if sibling_sources.intersection({"github", "exploitdb"}):
                    weak.append(item)
                    continue
            if has_authority and assessment_item.get("verdict") == "confirmed" and version_match == "yes" and not hard_mismatch:
                strong.append(item)
            elif not hard_mismatch and has_authority and assessment_item.get("verdict") == "plausible":
                weak.append(item)
                if version_match != "yes":
                    version_unknown.append(item)
            else:
                weak.append(item)
                if version_match != "yes":
                    version_unknown.append(item)
    else:
        # Backward-compatible fallback while older unit tests still feed vuln_hypotheses only.
        strong = []
        weak = []
        version_unknown = []
        for h in hypotheses:
            evidence = h.get("evidence_chain", [])
            confidence = h.get("confidence", 0)
            if len(evidence) >= HYPOTHESIS_MIN_EVIDENCE and confidence >= HYPOTHESIS_MIN_CONFIDENCE:
                strong.append(h)
            else:
                weak.append(h)

    if not strong:
        if latest and latest.phase == "hypothesis":
            dm.mark_outcome(latest.step, "invalidated")
        reason = (
            "No strong authoritative candidate with confirmed version support survived retrieval. "
            "Need targeted recon to confirm exact version, CPE, or platform evidence."
            if shortlist and version_unknown else
            "No strong authoritative candidate survived retrieval. "
            "Need more recon to improve version, CPE, or platform evidence."
            if shortlist else
            (
                f"All {len(hypotheses)} hypotheses are weak "
                f"(need >= {HYPOTHESIS_MIN_EVIDENCE} evidence items and "
                f">= {HYPOTHESIS_MIN_CONFIDENCE} confidence). "
                "Going back for more recon."
            )
        )
        verdict = {
            "verdict": "need_more_recon",
            "reason": reason,
            "phase": "hypothesis",
            "weak_count": len(weak),
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.info("Hypothesis verifier: BLOCK — all weak")
        return {
            "verification_log": vlog,
            "hypothesis_verifier_blocks": blocks + 1,
            "decision_memory": dm.to_list(),
        }

    # Pass with only strong hypotheses
    if latest and latest.phase == "hypothesis":
        dm.mark_outcome(latest.step, "validated")
    verdict = {
        "verdict": "proceed",
        "reason": f"{len(strong)} strong hypothesis(es), {len(weak)} filtered out.",
        "phase": "hypothesis",
        "strong_count": len(strong),
        "filtered_count": len(weak),
        "timestamp": time.time(),
    }
    vlog.append(verdict)
    logger.info("Hypothesis verifier: PASS — %d strong, %d filtered", len(strong), len(weak))
    phase_timestamps.setdefault("hypothesis_complete_time", time.time())
    return {
        "verification_log": vlog,
        "vuln_hypotheses": strong if not shortlist else [
            item for item in hypotheses if item.get("candidate_id") in {entry.get("candidate_id") for entry in strong}
        ],
        "decision_memory": dm.to_list(),
        "phase_timestamps": phase_timestamps,
    }


def route_hypothesis_verifier(state: PentestState) -> str:
    """After hypothesis verifier: back to recon or forward to planning."""
    vlog = state.get("verification_log", [])
    if not vlog:
        return "planning"
    last = vlog[-1]
    if last.get("phase") != "hypothesis":
        return "planning"
    if last.get("verdict") == "exhausted":
        return "end"
    if last.get("verdict") == "need_more_recon":
        return "recon"
    return "planning"


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTION VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

_SUCCESS_MARKERS = [
    "uid=", "root@", "www-data@", "# ", "$ ",  # shell prompts
    "session opened", "meterpreter >",           # metasploit
    "flag{", "FLAG{", "HTB{",                    # CTF flags
    "pwned", "access granted",                   # generic success
]


def _collect_attempted_targets(state: PentestState) -> tuple[list[str], list[str]]:
    attempted_cves = list(dict.fromkeys(str(item) for item in (state.get("attempted_cves", []) or []) if str(item)))
    attempted_services = list(dict.fromkeys(str(item) for item in (state.get("attempted_services", []) or []) if str(item)))
    for exploit in list(state.get("exploit_plan", []) or []):
        cve_id = str(exploit.get("cve_id", "") or "").strip()
        if cve_id and cve_id not in attempted_cves:
            attempted_cves.append(cve_id)
        service_key = service_target_key(
            str(exploit.get("target_ip") or state.get("target_ip") or ""),
            exploit.get("target_port", 0),
            str(exploit.get("service", "") or ""),
        )
        if service_key not in attempted_services:
            attempted_services.append(service_key)
    return attempted_cves, attempted_services


def _next_service_index(state: PentestState, attempted_services: list[str]) -> int:
    target_services = list(state.get("target_services", []) or [])
    if not target_services:
        return int(state.get("current_service_index", 0) or 0)
    attempted = set(attempted_services)
    current_index = int(state.get("current_service_index", 0) or 0)
    for offset in range(1, len(target_services) + 1):
        index = (current_index + offset) % len(target_services)
        service = target_services[index]
        service_key = str(service.get("service_key") or service_target_key(
            str(service.get("target_ip", "")),
            service.get("port", 0),
            str(service.get("name", "")),
        ))
        if service_key not in attempted:
            return index
    return current_index


def _replan_execution_payload(
    state: PentestState,
    *,
    vlog: list[dict[str, Any]],
    dm: DecisionMemory,
    tracker: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    attempted_cves, attempted_services = _collect_attempted_targets(state)
    replan_count = state_int(state, "replan_count", 0) + 1
    next_index = _next_service_index(state, attempted_services)
    target_services = list(state.get("target_services", []) or [])
    next_service = target_services[next_index] if target_services else {}
    verdict = {
        "verdict": "replan",
        "reason": reason,
        "phase": "execution",
        "timestamp": time.time(),
    }
    vlog.append(verdict)
    latest = dm.get_latest()
    if latest and latest.phase == "planning":
        dm.mark_outcome(latest.step, "invalidated")
    return {
        "verification_log": vlog,
        "decision_memory": dm.to_list(),
        "replan_count": replan_count,
        "attempted_cves": attempted_cves,
        "attempted_services": attempted_services,
        "current_service_index": next_index,
        "app_name": str(next_service.get("name", "") or state.get("app_name", "")),
        "app_version": str(next_service.get("version", "") or state.get("app_version", "")),
        "keyword": str(next_service.get("name", "") or state.get("keyword", "")),
        "target_port": str(next_service.get("port", "") or state.get("target_port") or "") or None,
        "retrieval_bundle": {},
        "vuln_hypotheses": [],
        "cve_list": [],
        "exploit_plan": [],
        "execution_tracker": {},
        "selected_exploit": None,
        "current_proposal": None,
        "debate_history": [],
        "debate_round": 0,
        "planning_complete": False,
        "hypothesis_complete": False,
        "execution_success": False,
        "execution_summary": reason,
        "current_phase": "hypothesis",
        "phase2_route": "",
    }


def execution_verifier_node(state: PentestState) -> Dict[str, Any]:
    """
    Quality gate after each execution step. Checks:
    1. Success detection: look for shell/flag markers in recent output
    2. Repetition: same exploit attempt with same params?
    3. Exhaustion: have we tried all exploits in the plan?
    """
    get_config()
    em = EpisodicMemory.from_list(state.get("episodic_memory", []))
    vlog = list(state.get("verification_log", []))
    exec_step = state_int(state, "execution_step_count", 0)
    max_steps = state_int(state, "execution_max_steps", 30)
    dm = DecisionMemory.from_list(state.get("decision_memory", []))
    latest = dm.get_latest()
    phase_timestamps = dict(state.get("phase_timestamps", {}))
    tracker = dict(state.get("execution_tracker", {}) or {})
    replan_count = state_int(state, "replan_count", 0)
    replan_max = state_int(state, "replan_max", 0)
    can_replan = not state.get("timeout_exceeded") and replan_count < replan_max

    # Check if execution node already declared done
    if state.get("current_phase") == "done" or state.get("execution_success"):
        success = bool(state.get("execution_success"))
        if not success and can_replan:
            return _replan_execution_payload(
                state,
                vlog=vlog,
                dm=dm,
                tracker=tracker,
                reason=str(state.get("execution_summary") or "Execution exhausted the current plan; replanning."),
            )
        verdict = {
            "verdict": "success" if success else "exhausted",
            "reason": state.get("execution_summary", "Execution declared success."),
            "phase": "execution",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        if latest and latest.phase == "planning":
            dm.mark_outcome(latest.step, "validated" if success else "invalidated")
        payload: dict[str, Any] = {"verification_log": vlog, "decision_memory": dm.to_list()}
        if success:
            phase_timestamps.setdefault("execution_success_time", time.time())
            payload["phase_timestamps"] = phase_timestamps
        return payload

    if tracker:
        candidate_results = dict(tracker.get("candidate_results", {}))
        success_candidate = next(
            (
                candidate_id for candidate_id, result in candidate_results.items()
                if result.get("status") == "success" or result.get("verify_passed")
            ),
            "",
        )
        if success_candidate:
            verdict = {
                "verdict": "success",
                "reason": f"Candidate {success_candidate} verified successfully.",
                "phase": "execution",
                "timestamp": time.time(),
            }
            vlog.append(verdict)
            if latest and latest.phase == "planning":
                dm.mark_outcome(latest.step, "validated")
            phase_timestamps.setdefault("execution_success_time", time.time())
            return {
                "verification_log": vlog,
                "execution_success": True,
                "execution_summary": f"Execution succeeded with {success_candidate}.",
                "current_phase": "done",
                "decision_memory": dm.to_list(),
                "phase_timestamps": phase_timestamps,
            }

        order = list(tracker.get("candidate_order", []))
        index = int(tracker.get("current_candidate_index", 0) or 0)
        non_terminal = [
            candidate_id for candidate_id, result in candidate_results.items()
            if result.get("status") not in TERMINAL_STATUSES
        ]
        if not order or index >= len(order) or not non_terminal:
            last_failures = [
                f"{candidate_id}:{result.get('failure_class', 'unknown')}"
                for candidate_id, result in candidate_results.items()
                if result.get("status") in TERMINAL_STATUSES and result.get("status") != "success"
            ]
            reason = "Execution exhausted all candidates." + (f" Failures: {', '.join(last_failures[:3])}" if last_failures else "")
            if can_replan:
                return _replan_execution_payload(
                    state,
                    vlog=vlog,
                    dm=dm,
                    tracker=tracker,
                    reason=reason,
                )
            verdict = {
                "verdict": "exhausted",
                "reason": reason,
                "phase": "execution",
                "timestamp": time.time(),
            }
            vlog.append(verdict)
            if latest and latest.phase == "planning":
                dm.mark_outcome(latest.step, "invalidated")
            return {
                "verification_log": vlog,
                "current_phase": "done",
                "execution_summary": verdict["reason"],
                "decision_memory": dm.to_list(),
            }

        current_candidate = order[index] if 0 <= index < len(order) else ""
        current_result = candidate_results.get(current_candidate, {})
        verdict = {
            "verdict": "continue",
            "reason": (
                f"Continue execution with {current_candidate or 'next candidate'} "
                f"(status={current_result.get('status', 'pending')}, "
                f"failure={current_result.get('failure_class', 'none')})."
            ),
            "phase": "execution",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        return {
            "verification_log": vlog,
            "decision_memory": dm.to_list(),
            "spent_steps": em.total_steps(),
        }

    # Check for success markers in recent episodic outputs
    recent_exec = em.by_phase("execution")
    if recent_exec:
        last_output = recent_exec[-1].output_summary.lower()
        for marker in _SUCCESS_MARKERS:
            if marker.lower() in last_output:
                verdict = {
                    "verdict": "success",
                    "reason": f"Success marker detected: '{marker}'",
                    "phase": "execution",
                    "timestamp": time.time(),
                }
                vlog.append(verdict)
                logger.info("Execution verifier: SUCCESS — marker '%s' found", marker)
                if latest and latest.phase == "planning":
                    dm.mark_outcome(latest.step, "validated")
                phase_timestamps.setdefault("execution_success_time", time.time())
                return {
                    "verification_log": vlog,
                    "execution_success": True,
                    "execution_summary": f"Success detected via marker: {marker}",
                    "current_phase": "done",
                    "decision_memory": dm.to_list(),
                    "phase_timestamps": phase_timestamps,
                }

    # Check for repeat loop
    is_stuck, stuck_reason = _check_repetition(em, "execution")
    if is_stuck:
        reason = f"Execution stuck in repeat loop: {stuck_reason}"
        if can_replan:
            return _replan_execution_payload(
                state,
                vlog=vlog,
                dm=dm,
                tracker=tracker,
                reason=reason,
            )
        verdict = {
            "verdict": "exhausted",
            "reason": reason,
            "phase": "execution",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        logger.warning("Execution verifier: agent stuck, ending")
        if latest and latest.phase == "planning":
            dm.mark_outcome(latest.step, "invalidated")
        return {
            "verification_log": vlog,
            "current_phase": "done",
            "execution_summary": "Aborted: agent stuck in repeat loop.",
            "decision_memory": dm.to_list(),
        }

    # Check max steps
    if exec_step >= max_steps:
        reason = f"Hit max execution steps ({max_steps})."
        if can_replan:
            return _replan_execution_payload(
                state,
                vlog=vlog,
                dm=dm,
                tracker=tracker,
                reason=reason,
            )
        verdict = {
            "verdict": "exhausted",
            "reason": reason,
            "phase": "execution",
            "timestamp": time.time(),
        }
        vlog.append(verdict)
        if latest and latest.phase == "planning":
            dm.mark_outcome(latest.step, "invalidated")
        return {
            "verification_log": vlog,
            "current_phase": "done",
            "execution_summary": f"Exhausted after {max_steps} steps.",
            "decision_memory": dm.to_list(),
        }

    # Continue execution
    verdict = {
        "verdict": "continue",
        "reason": "No success or exhaustion yet. Continue.",
        "phase": "execution",
        "timestamp": time.time(),
    }
    vlog.append(verdict)
    return {
        "verification_log": vlog,
        "decision_memory": dm.to_list(),
        "spent_steps": em.total_steps(),
    }


def route_execution_verifier(state: PentestState) -> str:
    """After execution verifier: retry, done (success), exhausted, or replan."""
    vlog = state.get("verification_log", [])
    if not vlog:
        return "execution"
    last = vlog[-1]
    if last.get("phase") != "execution":
        return "execution"
    verdict = last.get("verdict", "")
    if verdict == "success":
        return "end"        # → END (oracle proof already adjudicates)
    if verdict == "replan":
        return "replan"
    if verdict == "exhausted":
        return "exhausted"  # → END directly
    return "execution"
