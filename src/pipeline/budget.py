"""
src/pipeline/budget.py
──────────────────────
Resource-budget enforcement.

Implements three preregistered BudgetTiers as hard gates:

    Low    — 100k tokens,  20 LLM calls,  25 tool calls, 20 commands, 10 min
    Medium — 300k tokens,  40 LLM calls,  50 tool calls, 40 commands, 20 min
    High   — 750k tokens,  80 LLM calls, 100 tool calls, 80 commands, 30 min

Vertex reports cached input as a subset of input.  The token cap is therefore
input + output + thinking; cached input is retained only for billing analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.pipeline.manifest import ResourceLimits


class BudgetExceeded(Exception):
    """Raised when a hard resource limit would be exceeded."""

    def __init__(self, limit: str, used: int | float, maximum: int | float) -> None:
        self.limit = limit
        self.used = used
        self.maximum = maximum
        super().__init__(f"Budget '{limit}' exceeded: {used}/{maximum}")


class BudgetTier(str, Enum):
    """Preregistered experiment budget tiers.

    All three tiers are fixed and must not be changed after the train pilot.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def to_limits(self) -> ResourceLimits:
        """Return the ResourceLimits for this tier."""
        _map: dict[str, dict[str, int]] = {
            "low":    {"max_runtime_seconds": 600,  "max_tool_calls": 25,  "max_executed_commands": 20,
                       "max_cves_per_service": 3, "max_methods_per_cve": 2,
                       "max_executed_candidates": 2, "max_attempts_per_candidate": 2,
                       "max_total_tokens": 100_000, "max_llm_calls": 20},
            "medium": {"max_runtime_seconds": 1200, "max_tool_calls": 50,  "max_executed_commands": 40,
                       "max_cves_per_service": 5, "max_methods_per_cve": 2,
                       "max_executed_candidates": 3, "max_attempts_per_candidate": 3,
                       "max_total_tokens": 300_000, "max_llm_calls": 40},
            "high":   {"max_runtime_seconds": 1800, "max_tool_calls": 100, "max_executed_commands": 80,
                       "max_cves_per_service": 7, "max_methods_per_cve": 3,
                       "max_executed_candidates": 5, "max_attempts_per_candidate": 3,
                       "max_total_tokens": 750_000, "max_llm_calls": 80},
        }
        cfg = _map[self.value]
        return ResourceLimits(
            max_runtime_seconds=cfg["max_runtime_seconds"],
            max_tool_calls=cfg["max_tool_calls"],
            max_executed_commands=cfg["max_executed_commands"],
            max_cves_per_service=cfg["max_cves_per_service"],
            max_methods_per_cve=cfg["max_methods_per_cve"],
            max_executed_candidates=cfg["max_executed_candidates"],
            max_attempts_per_candidate=cfg["max_attempts_per_candidate"],
            max_total_tokens=cfg["max_total_tokens"],
            max_llm_calls=cfg["max_llm_calls"],
        )

    @classmethod
    def from_str(cls, value: str) -> "BudgetTier":
        """Parse tier from string.

        Typos are experimental-design errors and must fail before a paid run
        starts; they must not silently change the condition to medium.
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"unknown budget tier {value!r}; expected one of {[tier.value for tier in cls]}") from None


@dataclass
class BudgetState:
    started_at: float = 0.0
    tool_calls: int = 0
    executed_commands: int = 0
    cves_per_service: dict[str, int] = field(default_factory=dict)
    methods_per_cve: dict[str, int] = field(default_factory=dict)
    executed_candidates: int = 0
    attempts_per_candidate: dict[str, int] = field(default_factory=dict)
    # Token and LLM call tracking (added for VeriPlanPT)
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    llm_calls: int = 0
    total_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Canonical provider total; cached input is already included in input."""
        return self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens


class ResourceBudget:
    """Stateful budget tracker enforcing ``ResourceLimits`` as hard gates."""

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self.state = BudgetState(started_at=time.time())

    # ── Runtime ───────────────────────────────────────────────────────────────
    def remaining_seconds(self, now: float | None = None) -> float:
        if self.state.started_at <= 0:
            return float(self.limits.max_runtime_seconds)
        elapsed = (now if now is not None else time.time()) - self.state.started_at
        return max(0.0, float(self.limits.max_runtime_seconds) - elapsed)

    def check_runtime(self, now: float | None = None) -> None:
        remaining = self.remaining_seconds(now)
        if remaining <= 0:
            elapsed = (now if now is not None else time.time()) - self.state.started_at
            raise BudgetExceeded("max_runtime_seconds", round(elapsed, 1), self.limits.max_runtime_seconds)

    # ── Token budget ──────────────────────────────────────────────────────────
    def check_llm_call(self, estimated_tokens: int = 0) -> None:
        """Preflight one LLM request before spending it."""
        self.check_runtime()
        max_calls = self.limits.max_llm_calls
        if max_calls and self.state.llm_calls >= max_calls:
            raise BudgetExceeded("max_llm_calls", self.state.llm_calls, max_calls)
        max_tokens = self.limits.max_total_tokens
        if max_tokens and estimated_tokens and self.state.total_tokens + estimated_tokens > max_tokens:
            raise BudgetExceeded("max_total_tokens", self.state.total_tokens + estimated_tokens, max_tokens)

    def record_llm_usage(
        self,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        usd: float = 0.0,
    ) -> None:
        """Record token usage from one LLM call and check caps."""
        self.state.total_input_tokens += input_tokens
        self.state.total_cached_input_tokens += cached_input_tokens
        self.state.total_output_tokens += output_tokens
        self.state.total_thinking_tokens += thinking_tokens
        self.state.total_usd += usd
        self.state.llm_calls += 1
        max_tokens = self.limits.max_total_tokens
        if max_tokens and self.state.total_tokens > max_tokens:
            raise BudgetExceeded("max_total_tokens", self.state.total_tokens, max_tokens)
        max_calls = self.limits.max_llm_calls
        if max_calls and self.state.llm_calls > max_calls:
            raise BudgetExceeded("max_llm_calls", self.state.llm_calls, max_calls)

    # ── Tool calls / commands ─────────────────────────────────────────────────
    def check_tool_call(self) -> None:
        """Preflight one tool/external interaction without consuming it."""
        self.check_runtime()
        if self.state.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("max_tool_calls", self.state.tool_calls, self.limits.max_tool_calls)

    def record_tool_call(self) -> None:
        self.check_tool_call()
        self.state.tool_calls += 1

    def check_command(self) -> None:
        self.check_runtime()
        if self.state.executed_commands >= self.limits.max_executed_commands:
            raise BudgetExceeded("max_executed_commands", self.state.executed_commands, self.limits.max_executed_commands)

    def record_command(self) -> None:
        self.check_command()
        self.state.executed_commands += 1

    # ── CVEs per service ──────────────────────────────────────────────────────
    def check_cve_for_service(self, service_key: str) -> None:
        used = self.state.cves_per_service.get(service_key, 0)
        if used >= self.limits.max_cves_per_service:
            raise BudgetExceeded("max_cves_per_service", used, self.limits.max_cves_per_service)

    def record_cve_for_service(self, service_key: str) -> None:
        self.check_cve_for_service(service_key)
        self.state.cves_per_service[service_key] = self.state.cves_per_service.get(service_key, 0) + 1

    # ── Methods per CVE ───────────────────────────────────────────────────────
    def check_method_for_cve(self, cve_id: str) -> None:
        used = self.state.methods_per_cve.get(cve_id, 0)
        if used >= self.limits.max_methods_per_cve:
            raise BudgetExceeded("max_methods_per_cve", used, self.limits.max_methods_per_cve)

    def record_method_for_cve(self, cve_id: str) -> None:
        self.check_method_for_cve(cve_id)
        self.state.methods_per_cve[cve_id] = self.state.methods_per_cve.get(cve_id, 0) + 1

    # ── Executed candidates ────────────────────────────────────────────────────
    def check_candidate(self) -> None:
        if self.state.executed_candidates >= self.limits.max_executed_candidates:
            raise BudgetExceeded("max_executed_candidates", self.state.executed_candidates, self.limits.max_executed_candidates)

    def record_candidate(self) -> None:
        self.check_candidate()
        self.state.executed_candidates += 1

    # ── Attempts per candidate ─────────────────────────────────────────────────
    def check_attempt(self, candidate_id: str) -> None:
        used = self.state.attempts_per_candidate.get(candidate_id, 0)
        if used >= self.limits.max_attempts_per_candidate:
            raise BudgetExceeded("max_attempts_per_candidate", used, self.limits.max_attempts_per_candidate)

    def record_attempt(self, candidate_id: str) -> None:
        self.check_attempt(candidate_id)
        self.state.attempts_per_candidate[candidate_id] = self.state.attempts_per_candidate.get(candidate_id, 0) + 1

    def state_to_dict(self) -> dict[str, Any]:
        """Serialize BudgetState for storage in PentestState."""
        s = self.state
        return {
            "started_at": s.started_at,
            "tool_calls": s.tool_calls,
            "executed_commands": s.executed_commands,
            "cves_per_service": dict(s.cves_per_service),
            "methods_per_cve": dict(s.methods_per_cve),
            "executed_candidates": s.executed_candidates,
            "attempts_per_candidate": dict(s.attempts_per_candidate),
            "total_input_tokens": s.total_input_tokens,
            "total_cached_input_tokens": s.total_cached_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "total_thinking_tokens": s.total_thinking_tokens,
            "llm_calls": s.llm_calls,
            "total_usd": s.total_usd,
        }

    @classmethod
    def restore(cls, limits: ResourceLimits, state_dict: dict[str, Any]) -> "ResourceBudget":
        """Restore a ResourceBudget from a serialized state dict."""
        budget = cls(limits)
        s = budget.state
        s.started_at = float(state_dict.get("started_at", 0.0))
        s.tool_calls = int(state_dict.get("tool_calls", 0))
        s.executed_commands = int(state_dict.get("executed_commands", 0))
        s.cves_per_service = dict(state_dict.get("cves_per_service", {}))
        s.methods_per_cve = dict(state_dict.get("methods_per_cve", {}))
        s.executed_candidates = int(state_dict.get("executed_candidates", 0))
        s.attempts_per_candidate = dict(state_dict.get("attempts_per_candidate", {}))
        s.total_input_tokens = int(state_dict.get("total_input_tokens", 0))
        s.total_cached_input_tokens = int(state_dict.get("total_cached_input_tokens", 0))
        s.total_output_tokens = int(state_dict.get("total_output_tokens", 0))
        s.total_thinking_tokens = int(state_dict.get("total_thinking_tokens", 0))
        s.llm_calls = int(state_dict.get("llm_calls", 0))
        s.total_usd = float(state_dict.get("total_usd", 0.0))
        return budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.limits.to_dict(),
            "state": self.state_to_dict(),
        }
