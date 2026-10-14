"""Phase budgets (Figure 1, block A: "finite budgets").

The design names four budgets (v75, §3.1/§3.3):

    K1    reconnaissance phase      bounded by the closed discovery tuple and
                                    the run's tool-call allowance
    K2    hypothesis construction   candidates retrieved per service
    K3    exploit attempts          attempts per candidate
    B_max global cost cap           total token allowance for the episode

They are derived from the run's :class:`ResourceLimits` so the design names and
the enforced manifest limits cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.manifest import ResourceLimits

__all__ = ["PhaseBudgets"]


@dataclass(frozen=True)
class PhaseBudgets:
    """v75 phase budgets mapped onto the enforced resource limits."""

    K1: int          # reconnaissance: tool calls available to the recon phase
    K2: int          # hypothesis construction: candidates per service
    K3: int          # exploit attempts per candidate
    B_max: int       # global cost cap (tokens)

    def __post_init__(self) -> None:
        # Monotonic per-scope usage counters ("finite budgets", v75 §3.3).
        # Not a dataclass field: it is runtime accounting, and therefore stays
        # out of equality, hashing and ``to_dict``.
        object.__setattr__(self, "_counters", {})

    def spent(self, scope: str) -> int:
        """Return how many enforcement units ``scope`` has already consumed."""
        return self._counters.get(scope, 0)

    def record(self, scope: str) -> int:
        """Consume one unit of ``scope`` and return the new, never-decreasing total."""
        value = self._counters.get(scope, 0) + 1
        self._counters[scope] = value
        return value

    @classmethod
    def from_limits(cls, limits: ResourceLimits) -> "PhaseBudgets":
        return cls(
            K1=int(limits.max_tool_calls),
            K2=int(limits.max_cves_per_service),
            K3=int(limits.max_attempts_per_candidate),
            B_max=int(limits.max_total_tokens),
        )

    def to_dict(self) -> dict[str, int]:
        return {"K1": self.K1, "K2": self.K2, "K3": self.K3, "B_max": self.B_max}
