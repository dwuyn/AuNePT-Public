"""
src/pipeline/queue.py
─────────────────────
Deterministic applicability, ranking, and per-CVE shortlisting.

Hard-reject mismatches in product, version range, platform, auth, network,
endpoint, or callback scope. Rank remaining candidates by:

  1. Exact applicability evidence
  2. Ability to satisfy the benchmark proof
  3. Complete, validated procedure
  4. Artifact provenance and trust
  5. Lower expected side effects
  6. Lower expected time/tool cost
  7. Absence of prior classified failures

Retain at most five CVEs per service and two executable methods per CVE.
Unknown version permits a safe check / version-independent candidate but never
an exact-applicability rating.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.pipeline.candidates import (
    ExploitCandidate,
    is_executable,
)
from src.pipeline.evidence import Fingerprint, constraint_matches
from src.pipeline.ledger import EventLedger
from src.pipeline.manifest import ResourceLimits, Scope
from src.pipeline.scope import ScopeValidator
from src.retrieval.applicability import PENALTY_ALPHA, penalty, priority_key


@dataclass
class RankedCandidate:
    candidate: ExploitCandidate
    applicability: str                 # exact | partial | unknown | mismatch
    capability_match: bool
    procedure_complete: bool
    score: float
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class QueuedCandidate:
    """A candidate carrying its applicability verdict and missing prerequisites.

    ``priority`` is the ``priority_key`` tuple the binary heap orders by; the
    monotonic ``_order`` tiebreak on the heap guard keeps unorderable candidate
    objects from ever being compared.
    """

    candidate: ExploitCandidate
    verdict: str
    g_missing: tuple[str, ...] = ()
    penalty: float = 0.0
    priority: tuple[int, int, str] = (0, 0, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": getattr(self.candidate, "candidate_id", ""),
            "verdict": self.verdict,
            "g_missing": list(self.g_missing),
            "penalty": self.penalty,
            "priority": list(self.priority),
        }


@dataclass
class CandidateQueue:
    ranked: list[RankedCandidate] = field(default_factory=list)
    cv_per_service: dict[str, list[str]] = field(default_factory=dict)
    methods_per_cve: dict[str, list[str]] = field(default_factory=dict)
    #: binary heap of ``(priority_key, order, QueuedCandidate)``; paper §3.2
    #: priority: applicability ≻ smaller |G(c)| ≻ stable identifier.
    _heap: list[tuple[tuple[int, int, str], int, QueuedCandidate]] = field(
        default_factory=list, repr=False, compare=False,
    )
    _order: int = field(default=0, repr=False, compare=False)

    def push(
        self,
        candidate: ExploitCandidate,
        verdict: str,
        g_missing: Iterable[str] = (),
        *,
        penalty_bonus: float = 0.0,
    ) -> QueuedCandidate:
        """Push a candidate with its ``App(c, M_W)`` verdict and ``G(c)``.

        ``penalty_bonus`` is the increased penalty a ``payload_rejected``
        outcome adds before replanning: it demotes the candidate below
        better-supported ones while keeping it admissible.
        """
        g = tuple(g_missing or ())
        steps = int(round(float(penalty_bonus) / PENALTY_ALPHA)) if penalty_bonus else 0
        base = priority_key(candidate, verdict, g)
        priority = (base[0], base[1] + steps, base[2])
        item = QueuedCandidate(
            candidate=candidate,
            verdict=verdict,
            g_missing=g,
            penalty=min(penalty(g) + float(penalty_bonus), 1.0),
            priority=priority,
        )
        heapq.heappush(self._heap, (item.priority, self._order, item))
        self._order += 1
        return item

    def pop_best(self) -> QueuedCandidate | None:
        """Pop the highest-priority candidate, or ``None`` when empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)


def _capability_satisfies(candidate_cap: str, proof_cap: str) -> bool:
    if candidate_cap == proof_cap:
        return True
    # Detection can never satisfy RCE / file_write / auth_bypass / session.
    rank = {
        "detection": 0, "info_read": 1, "file_write": 2,
        "auth_bypass": 3, "code_execution": 4, "session": 5,
    }
    if candidate_cap == "detection" and rank.get(proof_cap, -1) > 0:
        return False
    return rank.get(candidate_cap, -1) >= rank.get(proof_cap, -1)


def _procedure_complete(candidate: ExploitCandidate) -> bool:
    if not candidate.procedure:
        return False
    stages = {s.stage for s in candidate.procedure}
    if "execute" not in stages:
        return False
    # All steps must have at least one argv token and a positive timeout.
    for step in candidate.procedure:
        if not step.argv:
            return False
        if step.timeout_seconds <= 0:
            return False
    return True


def _in_scope(candidate: ExploitCandidate, validator: ScopeValidator) -> tuple[bool, str]:
    for step in candidate.procedure:
        dec = validator.validate_args(list(step.argv), stage=step.stage)
        if not dec:
            if dec.unresolved_placeholders and not dec.blocked_endpoints:
                continue
            return False, f"scope:{step.stage}:{dec.reason}"
    return True, ""


def _side_effect_rank(side_effect: str) -> int:
    order = ["read_only", "info_read", "auth_bypass", "file_write",
             "session", "remote_exploit"]
    return order.index(side_effect) if side_effect in order else len(order)


def _expected_cost(candidate: ExploitCandidate) -> float:
    if "expected_runtime" in candidate.extra:
        try:
            return float(candidate.extra.get("expected_runtime") or 0.0)
        except (TypeError, ValueError):
            pass
    cost = 0.0
    for step in candidate.procedure:
        cost += float(step.timeout_seconds or 0)
    if candidate.kind in {"nmap_nse", "nuclei"}:
        cost *= 0.5
    return cost


def _difficulty_score(candidate: ExploitCandidate) -> float:
    score = 0.0
    try:
        score += 10.0 * float(candidate.extra.get("evidence_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    try:
        score += 10.0 * float(candidate.extra.get("procedure_readiness", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    if candidate.extra.get("prior_failure"):
        score -= 25.0
    return score


def rank_candidates(
    candidates: Iterable[ExploitCandidate],
    *,
    fingerprint: Fingerprint,
    proof_capability: str = "code_execution",
    ledger: EventLedger | None = None,
    scope: Scope | None = None,
    manifest_approved_lab_ids: Iterable[str] | None = None,
    resolver=None,
) -> list[RankedCandidate]:
    """Deterministic ranking with hard mismatch rejection."""
    validator = ScopeValidator(scope or Scope(), resolver=resolver)
    out: list[RankedCandidate] = []
    for cand in candidates:
        reason: list[str] = []
        # Trust gating: discovery_only and blocked candidates must never be
        # ranked for execution; they remain visible only for diagnostics.
        if cand.provenance.trust == "blocked":
            reason.append("trust:blocked")
        # Hard-reject mismatches.
        if cand.constraint.vendor and fingerprint.vendor.parsed != cand.constraint.vendor.lower():
            reason.append("vendor_mismatch")
        if cand.constraint.product and fingerprint.product.parsed != cand.constraint.product.lower():
            reason.append("product_mismatch")
        if cand.platform and (
            not fingerprint.platform_hints
            or cand.platform != fingerprint.platform_hints[0]
        ):
            reason.append("platform_mismatch")
        grade = constraint_matches(cand.constraint, fingerprint)
        if grade == "mismatch":
            reason.append("version_mismatch")
        if cand.auth_required == "yes" and fingerprint.auth_hint not in {"required", "unknown"}:
            reason.append("auth_prereq")

        # Procedure completeness and scope.
        proc_complete = _procedure_complete(cand)
        if not proc_complete:
            reason.append("procedure_incomplete")
        scope_ok, scope_msg = _in_scope(cand, validator)
        if not scope_ok:
            reason.append(scope_msg)

        capability_match = _capability_satisfies(cand.capability, proof_capability)
        if not capability_match:
            reason.append("capability_mismatch")

        # Final applicability label.
        applicability = grade
        if reason and applicability == "exact":
            applicability = "partial"

        # Score.
        score = 0.0
        if applicability == "exact":
            score += 100.0
        elif applicability == "partial":
            score += 50.0
        elif applicability == "unknown":
            score += 10.0
        if capability_match:
            score += 40.0
        if proc_complete:
            score += 20.0
        score += {"trusted": 15.0, "lab_approved": 10.0,
                   "discovery_only": -100.0, "blocked": -1000.0}.get(
            cand.provenance.trust, 0.0)
        score += _difficulty_score(cand)
        score -= 5.0 * _side_effect_rank(cand.side_effect_class)
        score -= _expected_cost(cand) * 0.01
        # Penalize kinds that cannot satisfy a non-detection proof.
        if not capability_match and proof_capability != "detection":
            score -= 200.0

        # Mismatch or hard-block ⇒ not executable.
        executable = is_executable(cand, manifest_approved_lab_ids=manifest_approved_lab_ids)
        if "vendor_mismatch" in reason or "product_mismatch" in reason \
                or "version_mismatch" in reason or "platform_mismatch" in reason \
                or "auth_prereq" in reason or "capability_mismatch" in reason \
                or "procedure_incomplete" in reason or scope_msg \
                or cand.provenance.trust == "blocked":
            executable = False

        shortlist_reasons = [r for r in reason if not r.startswith("scope:")]
        if scope_msg:
            shortlist_reasons.append("scope_violation")
        out.append(RankedCandidate(
            candidate=cand, applicability=applicability,
            capability_match=capability_match, procedure_complete=proc_complete,
            score=score, rejection_reasons=shortlist_reasons,
        ))
        if ledger is not None:
            outcome = ""
            failure_class = ""
            if not executable:
                if scope_msg:
                    outcome, failure_class = "blocked_by_policy", "scope_violation"
                elif any(r in reason for r in ("capability_mismatch", "procedure_incomplete")):
                    outcome, failure_class = "not_executable", "procedure_incomplete"
                elif reason or cand.provenance.trust == "blocked":
                    outcome, failure_class = "blocked_by_policy", "policy_block"
            ledger.record(
                phase="queue", stage="applicability",
                cve_id=cand.cve_id, candidate_id=cand.candidate_id,
                method=cand.kind,
                outcome=outcome,
                failure_class=failure_class,
                scope_decision="allowed" if scope_ok else "blocked",
                policy_decision="execute" if executable else "blocked",
                detail=applicability,
                payload={"score": score, "reasons": reason, "executable": executable,
                         "capability_match": capability_match},
            )
    # Sort descending by score, then by candidate_id for stability.
    out.sort(key=lambda rc: (-rc.score, rc.candidate.candidate_id))
    return out


def shortlist(
    ranked: list[RankedCandidate],
    *,
    limits: ResourceLimits,
    manifest_approved_lab_ids: Iterable[str] | None = None,
) -> CandidateQueue:
    """Apply max_cves_per_service and max_methods_per_cve shortlists."""
    by_service_cve: dict[str, dict[str, list[RankedCandidate]]] = {}
    # We don't have explicit service labels here; group by candidate source as
    # the v1 stand-in and store per-CVE method counts.
    for rc in ranked:
        if not is_executable(rc.candidate, manifest_approved_lab_ids=manifest_approved_lab_ids):
            continue
        if rc.rejection_reasons or not rc.capability_match or not rc.procedure_complete:
            continue
        if rc.applicability == "mismatch":
            continue
        by_service_cve.setdefault(rc.candidate.source, {}).setdefault(rc.candidate.cve_id, []).append(rc)
    selected: list[RankedCandidate] = []
    cv_per_service: dict[str, list[str]] = {}
    methods_per_cve: dict[str, list[str]] = {}
    for service, cves in by_service_cve.items():
        cv_per_service[service] = []
        # Order CVEs by best score.
        best_by_cve = sorted(cves.items(),
                              key=lambda kv: max((r.score for r in kv[1]), default=-1e9),
                              reverse=True)
        for cve, methods in best_by_cve:
            if len(cv_per_service[service]) >= limits.max_cves_per_service:
                break
            cv_per_service[service].append(cve)
            # Up to 2 methods per CVE.
            for rc in methods[:limits.max_methods_per_cve]:
                selected.append(rc)
                methods_per_cve.setdefault(cve, []).append(rc.candidate.kind)
    return CandidateQueue(ranked=selected, cv_per_service=cv_per_service,
                          methods_per_cve=methods_per_cve)
