"""
src/planning/difficulty.py
──────────────────────────
Online difficulty estimator for the VeriPlanPT planning policy.

Computes a difficulty_score ∈ [0, 1] and a breakdown vector for a given
attack context.  The score is used by BudgetPolicy to adjust action utility
weights before candidate selection.

Inputs:
    attack_horizon      — remaining budget steps (tool_calls + commands remaining)
    evidence_confidence — mean confidence across active fingerprints
    context_load        — fraction of token budget already consumed [0, 1]
    historical_success  — per-CWE success rate from ledger history [0, 1]

A higher difficulty_score means the planner should prefer conservative,
high-confidence actions and avoid expensive exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DifficultyVector:
    """Decomposed difficulty signal for a single planning round."""
    horizon_pressure: float = 0.0    # 1 - (remaining_steps / max_steps) ∈ [0,1]
    evidence_gap: float = 0.0        # 1 - mean_confidence ∈ [0,1]
    context_load: float = 0.0        # token fraction used ∈ [0,1]
    historical_failure: float = 0.0  # 1 - historical_success ∈ [0,1]
    difficulty_score: float = 0.0    # weighted aggregate ∈ [0,1]

    def to_dict(self) -> dict[str, float]:
        return {
            "horizon_pressure": round(self.horizon_pressure, 4),
            "evidence_gap": round(self.evidence_gap, 4),
            "context_load": round(self.context_load, 4),
            "historical_failure": round(self.historical_failure, 4),
            "difficulty_score": round(self.difficulty_score, 4),
        }


class DifficultyEstimator:
    """Computes online difficulty estimate from budget and evidence signals.

    Weights are deliberately equal by default and must be tuned only on the
    40-case train split.  After the train pilot the weights are locked in
    policy.lock.json and must not be changed.
    """

    # Default equal weights — will be overridden by policy.lock.json after CV.
    DEFAULT_WEIGHTS = {
        "horizon_pressure": 0.25,
        "evidence_gap": 0.25,
        "context_load": 0.25,
        "historical_failure": 0.25,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        total = sum(w.values())
        if total > 0:
            self._w = {k: v / total for k, v in w.items()}
        else:
            self._w = dict(self.DEFAULT_WEIGHTS)

    def estimate(
        self,
        *,
        remaining_steps: int,
        max_steps: int,
        mean_evidence_confidence: float,
        token_fraction_used: float,
        historical_success_rate: float,
    ) -> DifficultyVector:
        """Return a DifficultyVector for the current planning context.

        Args:
            remaining_steps: tool_calls + executed_commands remaining in budget.
            max_steps: total allowed steps for this tier.
            mean_evidence_confidence: average confidence of active fingerprints [0,1].
            token_fraction_used: total_tokens / max_total_tokens [0,1].
            historical_success_rate: CWE-level empirical success rate [0,1].
        """
        horizon_pressure = 1.0 - max(0.0, min(1.0, remaining_steps / max(max_steps, 1)))
        evidence_gap = 1.0 - max(0.0, min(1.0, mean_evidence_confidence))
        context_load = max(0.0, min(1.0, token_fraction_used))
        historical_failure = 1.0 - max(0.0, min(1.0, historical_success_rate))

        score = (
            self._w["horizon_pressure"] * horizon_pressure
            + self._w["evidence_gap"] * evidence_gap
            + self._w["context_load"] * context_load
            + self._w["historical_failure"] * historical_failure
        )
        score = max(0.0, min(1.0, score))

        return DifficultyVector(
            horizon_pressure=horizon_pressure,
            evidence_gap=evidence_gap,
            context_load=context_load,
            historical_failure=historical_failure,
            difficulty_score=score,
        )

    @classmethod
    def from_budget_state(
        cls,
        budget_state: dict[str, Any],
        max_tool_calls: int,
        max_commands: int,
        max_tokens: int,
        mean_confidence: float = 0.5,
        historical_success: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> "tuple[DifficultyEstimator, DifficultyVector]":
        """Convenience factory: build estimator and estimate from a BudgetState dict."""
        estimator = cls(weights)
        remaining_tool = max(0, max_tool_calls - int(budget_state.get("tool_calls", 0)))
        remaining_cmd = max(0, max_commands - int(budget_state.get("executed_commands", 0)))
        remaining_steps = remaining_tool + remaining_cmd
        max_steps = max_tool_calls + max_commands
        total_tokens = int(budget_state.get("total_input_tokens", 0)) + \
                       int(budget_state.get("total_cached_input_tokens", 0)) + \
                       int(budget_state.get("total_output_tokens", 0)) + \
                       int(budget_state.get("total_thinking_tokens", 0))
        token_frac = total_tokens / max(max_tokens, 1)
        vector = estimator.estimate(
            remaining_steps=remaining_steps,
            max_steps=max_steps,
            mean_evidence_confidence=mean_confidence,
            token_fraction_used=token_frac,
            historical_success_rate=historical_success,
        )
        return estimator, vector
