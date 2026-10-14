"""
src/planning/policy.py
──────────────────────
Evidence-gated budget-aware action selection policy for VeriPlanPT.

The BudgetPolicy scores each candidate action by:

    score = w1 * P(success) + w2 * E(evidence_gain) - w3 * norm_cost - w4 * risk

Weights are loaded from policy.lock.json after 5-fold cross-validation on
the 40-case train split.  Before the file exists, equal weights are used.

Service rotation rules (hardcoded, not tunable):
  - Two consecutive failures that produce no new evidence on the same service
    force a rotation to the next service with remaining candidates.
  - No service may consume more than 50% of the remaining budget unless its
    confidence is >= 0.8 OR all other services have been exhausted.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from src.planning.difficulty import DifficultyVector

logger = logging.getLogger(__name__)

# Path to the locked weights file (written after CV on train split).
_POLICY_LOCK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "policy.lock.json",
)

# Allowed weight values in the grid search.
WEIGHT_GRID = {0.0, 0.25, 0.5, 0.75, 1.0}


@dataclass
class PolicyWeights:
    """Normalized action-scoring weights.  Must sum to 1.0."""
    w_success: float = 0.25      # P(success)
    w_evidence_gain: float = 0.25  # E(evidence gain)
    w_cost: float = 0.25         # normalized cost (penalty)
    w_risk: float = 0.25         # risk (penalty)

    def to_dict(self) -> dict[str, float]:
        return {
            "w_success": self.w_success,
            "w_evidence_gain": self.w_evidence_gain,
            "w_cost": self.w_cost,
            "w_risk": self.w_risk,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyWeights":
        return cls(
            w_success=float(d.get("w_success", 0.25)),
            w_evidence_gain=float(d.get("w_evidence_gain", 0.25)),
            w_cost=float(d.get("w_cost", 0.25)),
            w_risk=float(d.get("w_risk", 0.25)),
        )


@dataclass
class ScoredAction:
    """One candidate with its policy score and supporting signals."""
    candidate_id: str
    service_key: str
    cve_id: str
    kind: str
    p_success: float = 0.5
    expected_evidence_gain: float = 0.5
    normalized_cost: float = 0.5
    risk: float = 0.5
    difficulty_score: float = 0.5
    policy_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "service_key": self.service_key,
            "cve_id": self.cve_id,
            "kind": self.kind,
            "p_success": round(self.p_success, 4),
            "expected_evidence_gain": round(self.expected_evidence_gain, 4),
            "normalized_cost": round(self.normalized_cost, 4),
            "risk": round(self.risk, 4),
            "difficulty_score": round(self.difficulty_score, 4),
            "policy_score": round(self.policy_score, 4),
            "rank": self.rank,
        }


class BudgetPolicy:
    """Score and rank candidate actions under budget constraints.

    Load weights from policy.lock.json if it exists; otherwise use equal weights.
    The weights file is written (and thereafter immutable) after the CV sweep
    on the 40-case train split.
    """

    # Service rotation threshold: consecutive failures per service.
    ROTATION_THRESHOLD = 2
    # Max fraction of remaining budget a single service may consume.
    SERVICE_BUDGET_CAP = 0.5
    # Minimum confidence to exempt a service from the budget cap.
    CONFIDENCE_THRESHOLD = 0.8

    def __init__(self, weights: PolicyWeights | None = None) -> None:
        self._weights = weights or self._load_weights()
        self._consecutive_failures: dict[str, int] = {}

    @staticmethod
    def _load_weights() -> PolicyWeights:
        """Load weights from policy.lock.json if available; else use equal default."""
        if os.path.exists(_POLICY_LOCK_PATH):
            try:
                with open(_POLICY_LOCK_PATH, encoding="utf-8") as fh:
                    data = json.load(fh)
                w = PolicyWeights.from_dict(data.get("weights", {}))
                logger.info("BudgetPolicy: loaded locked weights from %s", _POLICY_LOCK_PATH)
                return w
            except Exception as exc:
                logger.warning("BudgetPolicy: could not load policy.lock.json: %s", exc)
        return PolicyWeights()

    def score_action(
        self,
        candidate_id: str,
        service_key: str,
        cve_id: str,
        kind: str,
        *,
        p_success: float,
        expected_evidence_gain: float,
        normalized_cost: float,
        risk: float,
        difficulty: DifficultyVector | None = None,
    ) -> ScoredAction:
        """Compute policy score for one candidate action."""
        w = self._weights
        # Adjust p_success down when difficulty is high.
        eff_p_success = p_success * (1.0 - 0.5 * (difficulty.difficulty_score if difficulty else 0.0))
        score = (
            w.w_success * eff_p_success
            + w.w_evidence_gain * expected_evidence_gain
            - w.w_cost * normalized_cost
            - w.w_risk * risk
        )
        return ScoredAction(
            candidate_id=candidate_id,
            service_key=service_key,
            cve_id=cve_id,
            kind=kind,
            p_success=p_success,
            expected_evidence_gain=expected_evidence_gain,
            normalized_cost=normalized_cost,
            risk=risk,
            difficulty_score=difficulty.difficulty_score if difficulty else 0.5,
            policy_score=score,
        )

    def rank_actions(self, actions: list[ScoredAction]) -> list[ScoredAction]:
        """Return actions sorted by policy_score descending."""
        ranked = sorted(actions, key=lambda a: a.policy_score, reverse=True)
        for i, a in enumerate(ranked):
            a.rank = i
        return ranked

    def should_rotate_service(self, service_key: str, produced_evidence: bool) -> bool:
        """Return True if this service has hit the rotation threshold.

        A service must be rotated after ROTATION_THRESHOLD consecutive failures
        that produced no new evidence.
        """
        if produced_evidence:
            self._consecutive_failures[service_key] = 0
            return False
        count = self._consecutive_failures.get(service_key, 0) + 1
        self._consecutive_failures[service_key] = count
        return count >= self.ROTATION_THRESHOLD

    def state_to_dict(self) -> dict[str, Any]:
        return {"consecutive_failures": dict(self._consecutive_failures)}

    def restore_state(self, state: dict[str, Any]) -> None:
        self._consecutive_failures = {
            str(key): int(value) for key, value in dict(state.get("consecutive_failures") or {}).items()
        }

    def is_service_budget_allowed(
        self,
        service_key: str,
        *,
        remaining_budget_tokens: int,
        max_budget_tokens: int,
        service_confidence: float,
        other_services_exhausted: bool,
    ) -> bool:
        """Return True if this service is allowed to consume more budget.

        A service is blocked if it would consume more than SERVICE_BUDGET_CAP
        of the remaining budget unless its confidence >= CONFIDENCE_THRESHOLD
        or all other services are exhausted.
        """
        if other_services_exhausted:
            return True
        if service_confidence >= self.CONFIDENCE_THRESHOLD:
            return True
        # Estimate whether this service is already dominating the budget.
        if max_budget_tokens <= 0:
            return True
        used_fraction = 1.0 - (remaining_budget_tokens / max_budget_tokens)
        # Allow if we are still below the cap.
        return used_fraction <= self.SERVICE_BUDGET_CAP


def load_policy() -> BudgetPolicy:
    """Return a BudgetPolicy with weights loaded from policy.lock.json (if available)."""
    return BudgetPolicy()


def lock_policy(weights: PolicyWeights, metadata: dict[str, Any] | None = None) -> None:
    """Write the final CV-selected weights to policy.lock.json.

    This must be called exactly once after the CV sweep on the train split,
    before the test split is opened.  The file is append-protected by its
    absence check.
    """
    if os.path.exists(_POLICY_LOCK_PATH):
        raise FileExistsError(
            f"policy.lock.json already exists at {_POLICY_LOCK_PATH}. "
            "Delete it manually only if you are re-running the train CV sweep."
        )
    payload: dict[str, Any] = {"weights": weights.to_dict()}
    if metadata:
        payload["metadata"] = metadata
    with open(_POLICY_LOCK_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("BudgetPolicy: weights locked at %s", _POLICY_LOCK_PATH)
