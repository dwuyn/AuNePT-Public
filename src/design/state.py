"""Evidence admission and structured state (Figure 1, block C).

``Admit(o, C_A) -> Update(M_W, Δ)``: observations become part of the world
state only through this function pair, and only when they belong to the
authorized contract. Generative output never writes to ``M_W`` directly.

``M_E`` (append-only execution ledger) and ``M_D`` (decision metadata) live in
``src/pipeline/ledger.py`` and ``src/memory/decision.py``; this module only
owns the admission step itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.design.contract import RuntimeContract
from src.memory.world_state import ServiceInfo, WorldState

PROVENANCE_CONTRACT = "contract"
PROVENANCE_OBSERVED = "observed"
PROVENANCE_DERIVED = "derived"

KIND_SERVICE = "service"
KIND_OBSERVATION = "observation"


@dataclass(frozen=True)
class Delta:
    """An admitted observation, ready to be written into ``M_W``."""

    kind: str = KIND_OBSERVATION
    provenance: str = PROVENANCE_OBSERVED
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provenance": self.provenance,
            "payload": dict(self.payload),
            "source": self.source,
        }


def admit(observation: Mapping[str, Any], contract: RuntimeContract) -> Delta:
    """``Admit(o, C_A)`` — validate an observation and tag its provenance.

    Raises ``PermissionError`` when the observation points outside the
    authorized contract; the caller records the rejection as evidence rather
    than silently dropping it.
    """
    payload = dict(observation or {})
    host = str(payload.get("host") or payload.get("ip") or payload.get("target_ip") or "").strip()
    port = payload.get("port")
    if host and not contract.allows_host(host):
        raise PermissionError(f"observation for host {host!r} is outside the authorized contract")
    if port is not None and not contract.allows_port(port):
        raise PermissionError(f"observation for port {port!r} is outside the authorized contract")
    kind = KIND_SERVICE if port is not None else KIND_OBSERVATION
    return Delta(
        kind=kind,
        provenance=PROVENANCE_OBSERVED,
        payload=payload,
        source=str(payload.get("source") or ""),
    )


def assert_port(contract: RuntimeContract, world_state: WorldState) -> None:
    """``AssertPort(C_A.host, C.port, C.relay)`` — seed ``M_W`` from the contract.

    Contract assertions enter ``M_W`` with provenance ``contract`` and are never
    downgraded by later observations (paper §3.2, Eq. 3).
    """
    host = str(contract.target_host or "").strip()
    if not host:
        raise ValueError("contract host is required")
    port = contract.target_port
    if port in (None, ""):
        raise ValueError("contract port is required")
    world_state.assert_node("host", host, {"ip": host}, PROVENANCE_CONTRACT)
    world_state.assert_node("port", str(port), {"port": int(port)}, PROVENANCE_CONTRACT)
    world_state.link(("host", host), "host-has-port", ("port", str(port)), PROVENANCE_CONTRACT)


def derive(
    payload: Mapping[str, Any],
    *,
    source: str,
    from_keys: Sequence[str] = (),
) -> Delta:
    """``derived`` evidence — a deterministic transformation of admitted evidence.

    Unlike :func:`admit`, a derived delta is not re-validated against ``C_A``:
    its inputs were already admitted, and the transformation itself is
    deterministic (for example ``InferServiceArchetype`` over a banner).
    """
    body = dict(payload or {})
    if from_keys:
        body["derived_from"] = [str(item) for item in from_keys]
    return Delta(
        kind=KIND_SERVICE if body.get("port") is not None else KIND_OBSERVATION,
        provenance=PROVENANCE_DERIVED,
        payload=body,
        source=str(source or ""),
    )


def update(world_state: WorldState, delta: Delta) -> WorldState:
    """``Update(M_W, Δ)`` — write an admitted delta into the world state.

    Writes both the typed service view and the property-graph assertion, so
    every persistent claim carries the provenance of the delta that produced it.
    """
    payload = delta.payload
    host = str(payload.get("host") or payload.get("ip") or "").strip()
    port = payload.get("port")
    if delta.kind != KIND_SERVICE or not host or port is None:
        return world_state

    service = ServiceInfo(
        port=int(port),
        protocol=str(payload.get("protocol") or "tcp"),
        name=str(payload.get("name") or ""),
        version=str(payload.get("version") or ""),
        banner=str(payload.get("banner") or ""),
        accessibility=str(payload.get("accessibility") or "open"),
        confidence=float(payload.get("confidence", 0.3) or 0.3),
        evidence=[str(item) for item in payload.get("evidence") or []],
        cpe=str(payload.get("cpe") or ""),
        provenance_sources=[delta.source] if delta.source else [],
        protocol_family=str(payload.get("protocol_family") or "unknown"),
        version_evidence_class=str(payload.get("version_evidence_class") or "unknown"),
    )
    # Graph assertions first: the admission's provenance is authoritative and
    # ``add_service`` only fills nodes that do not exist yet.
    world_state.assert_node("host", host, {"ip": host}, delta.provenance)
    world_state.assert_node("port", str(port), {"port": int(port)}, delta.provenance)
    world_state.link(("host", host), "host-has-port", ("port", str(port)), delta.provenance)
    world_state.assert_node(
        "service",
        f"{host}:{port}",
        {
            "host": host,
            "port": int(port),
            "name": service.name,
            "version": service.version,
            "protocol_family": service.protocol_family,
            "version_evidence_class": service.version_evidence_class,
        },
        delta.provenance,
    )
    world_state.link(("port", str(port)), "port-runs-service", ("service", f"{host}:{port}"), delta.provenance)
    # ``M_W`` is a property graph over hosts, ports, services *and endpoints*
    # (paper §3.2): a discovery or dispatch observation that names a path is an
    # endpoint assertion, and endpoint preconditions are evaluated against them.
    endpoint = str(payload.get("endpoint") or payload.get("path") or "").strip()
    if endpoint:
        world_state.assert_node(
            "endpoint",
            f"{host}:{port}{endpoint}",
            {"host": host, "port": int(port), "path": endpoint,
             "status": int(payload.get("status") or 0),
             "reached_by": str(payload.get("reached_by") or "")},
            delta.provenance,
        )
        world_state.link(
            ("port", str(port)), "port-exposes-endpoint",
            ("endpoint", f"{host}:{port}{endpoint}"), delta.provenance,
        )
    world_state.add_service(host, service)
    return world_state
