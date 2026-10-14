"""
src/memory/decision.py
──────────────────────
Decision memory: records *why* each decision was made.

Enables:
  - Ablation studies (did memory verification improve decision quality?)
  - Contradiction detection by the Verifier
  - Audit trail for exploit selection reasoning

Each Decision records the question, chosen option, alternatives considered,
LLM reasoning, evidence references, and eventual outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Decision:
    """A single recorded decision in the pentest workflow."""
    step: int                                  # episodic step reference
    phase: str = ""                            # "recon" | "hypothesis" | "planning" | "execution"
    question: str = ""                         # "Which CVE to target?" | "Which exploit to try?"
    chosen: str = ""                           # "CVE-2021-41773"
    alternatives: list[str] = field(default_factory=list)
    reasoning: str = ""                        # LLM-generated justification
    evidence_refs: list[int] = field(default_factory=list)  # episodic step indices
    confidence: float = 0.0                    # 0.0-1.0
    outcome: str = "pending"                   # "pending" | "validated" | "invalidated"

    action: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    difficulty_vector: dict = field(default_factory=dict)
    expected_utility: float = 0.0
    budget_before: dict = field(default_factory=dict)
    budget_after: dict = field(default_factory=dict)
    verifier_verdict: str = "pending"


class DecisionMemory:
    """
    Append-only record of decisions and their outcomes.
    The Verifier uses this to detect contradictions and track decision quality.
    """

    def __init__(self) -> None:
        self._decisions: list[Decision] = []

    def record(self, decision: Decision) -> None:
        """Record a new decision."""
        self._decisions.append(decision)

    def get_by_phase(self, phase: str) -> list[Decision]:
        return [d for d in self._decisions if d.phase == phase]

    def mark_outcome(self, step: int, outcome: str) -> None:
        """Called by verifier/execution when a decision's outcome becomes known."""
        for d in self._decisions:
            if d.step == step:
                d.outcome = outcome
                return

    def get_latest(self) -> Optional[Decision]:
        return self._decisions[-1] if self._decisions else None

    def contradictions(self) -> list[tuple[Decision, Decision]]:
        """
        Find decision pairs where a later decision contradicts an earlier one.
        E.g., step 5 chose CVE-A, step 8 chose CVE-B but for the same service.
        """
        conflicts = []
        for i, d1 in enumerate(self._decisions):
            for d2 in self._decisions[i + 1:]:
                if (
                    d1.phase == d2.phase
                    and d1.question == d2.question
                    and d1.chosen != d2.chosen
                    and d1.outcome != "invalidated"
                ):
                    conflicts.append((d1, d2))
        return conflicts

    def validated_count(self) -> int:
        return sum(1 for d in self._decisions if d.outcome == "validated")

    def invalidated_count(self) -> int:
        return sum(1 for d in self._decisions if d.outcome == "invalidated")

    def get_pending(self) -> list[Decision]:
        return [d for d in self._decisions if d.verifier_verdict == "pending"]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_list(self) -> list[dict]:
        return [asdict(d) for d in self._decisions]

    @classmethod
    def from_list(cls, data: list[dict]) -> DecisionMemory:
        dm = cls()
        for d in data:
            dm._decisions.append(Decision(**d))
        return dm
