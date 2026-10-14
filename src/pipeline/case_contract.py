"""Build the agent-visible runtime contract ``C_A`` for a case.

``build_contract`` projects only public task metadata and the launch manifest
onto a :class:`~src.design.contract.RuntimeContract`. It never reads the
evaluator-private ``hidden/`` truth fields (expected CVE, vulnerable version,
success oracle); the contract carries addressing, scope and permitted service
interfaces only.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.design.contract import RuntimeContract
from src.pipeline.manifest import ResourceLimits, Scope
from src.pipeline.protocol_detect import PROTOCOL_FAMILIES

_FAMILY_ALIASES: dict[str, str] = {
    "tls": "tls", "ssl": "tls", "tls1.2": "tls", "tls1.3": "tls", "https": "tls",
    "ssh": "ssh", "ssh2": "ssh",
    "http": "http", "http1": "http", "http1.1": "http", "http2": "http", "h2": "http",
    "ws": "websocket", "wss": "websocket", "websocket": "websocket",
    "tcp": "raw-tcp", "raw": "raw-tcp", "rawtcp": "raw-tcp", "raw-tcp": "raw-tcp",
}


def normalize_protocol_family(value: Any) -> str:
    """Map a declared protocol label onto one of the five family names."""
    label = str(value or "").strip().lower()
    if label in PROTOCOL_FAMILIES:
        return label
    return _FAMILY_ALIASES.get(label, label)


def build_contract(public_task: Mapping[str, Any], launch: Mapping[str, Any]) -> RuntimeContract:
    """Assemble the agent-visible contract from public task and launch data.

    Raises :class:`ValueError` with ``"contract port is required"`` when neither
    ``launch["container_port"]`` nor ``public_task["target"]["exposed_ports"][0]``
    resolves a port.
    """
    public: Mapping[str, Any] = public_task or {}
    launch_map: Mapping[str, Any] = launch or {}
    target = public.get("target") if isinstance(public.get("target"), Mapping) else {}

    port = _resolve_port(launch_map, target)
    if port is None:
        raise ValueError("contract port is required")
    host = str(target.get("host") or "")

    families = _resolve_families(launch_map.get("relay_protocol"))
    scope = _resolve_scope(public, target, host=host, port=port)
    limits = _resolve_limits(launch_map, public)

    return RuntimeContract.from_scope(
        scope,
        target_host=host,
        target_port=port,
        limits=limits,
        budget_tier=str(launch_map.get("budget_tier") or public.get("budget_tier") or "medium"),
        protocol_families=families,
    )


def _resolve_port(launch: Mapping[str, Any], target: Mapping[str, Any]) -> int | None:
    raw = launch.get("container_port")
    if raw in (None, ""):
        exposed = target.get("exposed_ports")
        if isinstance(exposed, Sequence) and not isinstance(exposed, (str, bytes)) and exposed:
            raw = exposed[0]
    if raw in (None, ""):
        return None
    return int(raw)


def _resolve_families(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes)):
        raw_items: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        raw_items = value
    else:
        raw_items = [value]
    families: list[str] = []
    for item in raw_items:
        family = normalize_protocol_family(item)
        if family and family not in families:
            families.append(family)
    return tuple(families)


def _resolve_scope(
    public: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    host: str,
    port: int,
) -> Scope:
    declared = public.get("scope") if isinstance(public.get("scope"), Mapping) else {}
    hostnames = [str(item) for item in _as_sequence(declared.get("allowed_hosts"))]
    if host and host not in hostnames:
        hostnames.insert(0, host)
    networks = [str(item) for item in _as_sequence(declared.get("allowed_networks"))]
    for extra in _as_sequence(target.get("networks")):
        if str(extra) not in networks:
            networks.append(str(extra))
    ports = [int(item) for item in _as_sequence(declared.get("allowed_ports"))]
    if port not in ports:
        ports.insert(0, port)
    schemes = [str(item) for item in _as_sequence(declared.get("allowed_schemes"))]
    return Scope(
        allowed_hostnames=hostnames,
        allowed_networks=networks,
        allowed_ports=ports,
        allowed_schemes=schemes,
    )


def _resolve_limits(launch: Mapping[str, Any], public: Mapping[str, Any]) -> ResourceLimits:
    raw = launch.get("limits") or public.get("limits")
    if isinstance(raw, ResourceLimits):
        return raw
    if isinstance(raw, Mapping):
        return ResourceLimits.from_dict(raw)
    return ResourceLimits()


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)
