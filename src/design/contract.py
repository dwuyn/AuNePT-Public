"""``C_A`` — the agent-visible runtime contract (Figure 1, block A).

The contract carries only what an authorized assessment needs: target
addressing, the authorized scope, the permitted service interfaces and the
finite budgets. It never carries the case's success oracle, the expected CVE
or the vulnerable version — those live in the evaluator-private contract
``C_E`` and are never exposed to the agent (see ``outcomes.py``).
"""
from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.pipeline.budget import ResourceBudget
from src.pipeline.manifest import ResourceLimits, Scope


def _network_objects(values: Iterable[str]) -> tuple[Any, ...]:
    networks = []
    for raw in values:
        try:
            networks.append(ipaddress.ip_network(str(raw), strict=False))
        except ValueError:
            continue
    return tuple(networks)


@dataclass(frozen=True)
class RuntimeContract:
    """Agent-visible runtime contract (``C_A``)."""

    target_host: str = ""
    target_port: str | None = None
    allowed_hostnames: tuple[str, ...] = ()
    allowed_networks: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = ()
    allowed_schemes: tuple[str, ...] = ()
    protocol_families: tuple[str, ...] = ()
    budget_tier: str = "medium"
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    # ── construction ──────────────────────────────────────────────────────
    @staticmethod
    def _coerce_scope(scope: Any) -> Scope:
        if isinstance(scope, Scope):
            return scope
        if isinstance(scope, Mapping):
            return Scope.from_dict(scope)
        return Scope()

    @staticmethod
    def _coerce_limits(limits: Any) -> ResourceLimits:
        if isinstance(limits, ResourceLimits):
            return limits
        if isinstance(limits, Mapping):
            return ResourceLimits.from_dict(limits)
        return ResourceLimits()

    @classmethod
    def from_scope(
        cls,
        scope: Scope | Mapping[str, Any],
        *,
        target_host: str = "",
        target_port: Any = None,
        limits: ResourceLimits | Mapping[str, Any] | None = None,
        budget_tier: str = "medium",
        protocol_families: Sequence[str] = (),
    ) -> "RuntimeContract":
        resolved = cls._coerce_scope(scope)
        port = None
        if target_port not in (None, ""):
            port = str(target_port)
        return cls(
            target_host=str(target_host or ""),
            target_port=port,
            allowed_hostnames=tuple(str(item) for item in resolved.allowed_hostnames),
            allowed_networks=tuple(str(item) for item in resolved.allowed_networks),
            allowed_ports=tuple(int(item) for item in resolved.allowed_ports),
            allowed_schemes=tuple(str(item) for item in resolved.allowed_schemes),
            protocol_families=tuple(str(item) for item in protocol_families),
            budget_tier=str(budget_tier or "medium"),
            limits=cls._coerce_limits(limits),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Any,
        *,
        target_host: str = "",
        target_port: Any = None,
        budget_tier: str = "medium",
        protocol_families: Sequence[str] = (),
    ) -> "RuntimeContract":
        scope = getattr(manifest, "scope", None) or Scope()
        limits = getattr(manifest, "limits", None)
        return cls.from_scope(
            scope,
            target_host=target_host or getattr(manifest, "target_id", "") or "",
            target_port=target_port,
            limits=limits,
            budget_tier=budget_tier,
            protocol_families=protocol_families,
        )

    # ── predicates ────────────────────────────────────────────────────────
    def allows_host(self, host: str) -> bool:
        """True iff ``host`` is inside the authorized scope (fail closed)."""
        value = str(host or "").strip()
        if not value:
            return False
        if value in self.allowed_hostnames:
            return True
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return any(address in network for network in _network_objects(self.allowed_networks))

    def allows_port(self, port: Any) -> bool:
        if port in (None, ""):
            return False
        if not self.allowed_ports:
            return False
        try:
            return int(port) in self.allowed_ports
        except (TypeError, ValueError):
            return False

    # ── finite budgets ────────────────────────────────────────────────────
    @property
    def decision_budget(self) -> int:
        """``B_dec`` — orchestration decisions (LLM calls) per episode."""
        return int(self.limits.max_llm_calls)

    @property
    def tool_budget(self) -> int:
        """``B_tool`` — agent-initiated external interactions per episode."""
        return int(self.limits.max_tool_calls)

    def new_budget(self) -> ResourceBudget:
        return ResourceBudget(self.limits)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limits"] = self.limits.to_dict()
        data["decision_budget"] = self.decision_budget
        data["tool_budget"] = self.tool_budget
        return data
