"""
src/memory/world_state.py
─────────────────────────
Structured world-state memory: a typed graph of host → service → version
with confidence scores and evidence chains for every claim.

Solves: "agent quên service, version, credential hint" (improve.txt line 53)
        "context loss is failure mode #1" (PentestGPT)

Every service claim has:
  - A confidence score (0.0-1.0) reflecting how sure we are
  - A list of evidence strings (raw tool outputs that support the claim)

The Verifier reads these to decide if we have enough evidence to proceed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

#: provenance precedence for merging assertions (paper §3.2): an assertion
#: admitted from a stronger source is never downgraded by a weaker one.
_PROVENANCE_RANK = {"derived": 1, "observed": 2, "contract": 3}


def _provenance_rank(provenance: str) -> int:
    return _PROVENANCE_RANK.get(str(provenance), 0)


@dataclass
class ServiceInfo:
    """A single service running on a port, with confidence and evidence."""
    port: int
    protocol: str = "tcp"                     # "tcp" | "udp"
    name: str = ""                            # "apache" | "openssh" | "mysql"
    version: str = ""                         # "2.4.49" | "" if unknown
    banner: str = ""                          # raw banner string
    accessibility: str = "open"               # "open" | "filtered" | "closed"
    confidence: float = 0.3                   # 0.0-1.0
    evidence: list[str] = field(default_factory=list)
    cpe: str = ""                             # CPE string if available
    # §3.2: protocol family (tls | ssh | http | websocket | raw-tcp) and the
    # version-evidence class (explicit | inferred | unknown). Only explicit or
    # inferred evidence may satisfy a version predicate.
    protocol_family: str = "unknown"
    version_evidence_class: str = "unknown"

    activated_at: float = 0.0
    deactivated_at: float = 0.0
    ttl_seconds: float = 3600.0
    provenance_sources: list[str] = field(default_factory=list)

    def bump_confidence(self, delta: float, new_evidence: str) -> None:
        """Increase confidence and append supporting evidence."""
        self.confidence = min(1.0, self.confidence + delta)
        if new_evidence:
            self.evidence.append(new_evidence)

    def is_active(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        if self.deactivated_at > 0:
            return False
        if self.activated_at > 0 and (now - self.activated_at) > self.ttl_seconds:
            return False
        return True


def _merge_typed_evidence(existing: "ServiceInfo", incoming: "ServiceInfo") -> None:
    """Copy the typed evidence markers, never letting a default erase a known value.

    ``protocol_family`` and ``version_evidence_class`` both default to
    ``"unknown"``; an incoming non-default value upgrades the stored one, but a
    default admission must not downgrade a family/evidence class already known.
    """
    if incoming.protocol_family and incoming.protocol_family != "unknown":
        existing.protocol_family = incoming.protocol_family
    if incoming.version_evidence_class and incoming.version_evidence_class != "unknown":
        existing.version_evidence_class = incoming.version_evidence_class


@dataclass
class HostInfo:
    """A single host with OS hints and list of discovered services."""
    ip: str
    os_hint: str = ""
    os_confidence: float = 0.3
    services: list[ServiceInfo] = field(default_factory=list)

    def get_service(self, port: int) -> Optional[ServiceInfo]:
        for svc in self.services:
            if svc.port == port:
                return svc
        return None

    def upsert_service(self, svc: ServiceInfo) -> None:
        """Insert or update a service by port number. Never let a generic label
        overwrite a specific product name, and never replace a non-empty banner
        with an empty one."""
        _GENERIC_NAMES = {
            "", "unknown", "tcpwrapped", "generic", "none", "n/a",
            "http", "https", "https-alt", "ssl/http", "ssl-http",
        }
        existing = self.get_service(svc.port)
        if existing is None:
            self.services.append(svc)
        else:
            name_is_generic = svc.name.strip().lower() in _GENERIC_NAMES
            existing_name_is_generic = existing.name.strip().lower() in _GENERIC_NAMES

            if svc.confidence > existing.confidence:
                # Only replace name if the incoming name is more specific
                if not name_is_generic or existing_name_is_generic:
                    existing.name = svc.name or existing.name
                existing.version = svc.version or existing.version
                # Only replace banner if incoming has real content
                if svc.banner and svc.banner.strip():
                    existing.banner = svc.banner
                existing.confidence = svc.confidence
                existing.cpe = svc.cpe or existing.cpe
                _merge_typed_evidence(existing, svc)
            elif not name_is_generic and existing_name_is_generic:
                # Upgrade name from generic to specific even at lower confidence
                existing.name = svc.name or existing.name
                if svc.version:
                    existing.version = svc.version
                if svc.banner and svc.banner.strip():
                    existing.banner = svc.banner
                _merge_typed_evidence(existing, svc)

            existing.evidence.extend(svc.evidence)
            existing.accessibility = svc.accessibility

            for src in svc.provenance_sources:
                if src not in existing.provenance_sources:
                    existing.provenance_sources.append(src)

            if svc.activated_at > 0:
                existing.activated_at = svc.activated_at
            if svc.deactivated_at > 0:
                existing.deactivated_at = svc.deactivated_at
            if svc.ttl_seconds != 3600.0:
                existing.ttl_seconds = svc.ttl_seconds


@dataclass
class Credential:
    """A credential found or guessed during the pentest."""
    username: str
    password: str = ""                         # or hash
    source: str = ""                           # "found in config" | "brute-forced" | "default"
    target_service: str = ""                   # "ssh" | "mysql" | "ftp"
    verified: bool = False


@dataclass
class Session:
    """An established session/foothold on the target."""
    session_type: str = ""                     # "shell" | "meterpreter" | "ssh"
    target_ip: str = ""
    target_port: int = 0
    privilege_level: str = ""                  # "user" | "root" | "www-data"
    established_at_step: int = 0               # episodic memory step reference
    is_alive: bool = True
    verification_command: str = ""
    proof: str = ""
    origin_exploit: str = ""
    last_verified_at: float = 0.0


@dataclass
class WorldState:
    """
    The full structured world-state.
    Replaces the flat port_services dict with confidence-tracked, evidence-backed data.

    ``M_W`` (paper §3.2) is a typed property graph over hosts, ports, services,
    endpoints, credentials/sessions and their relations. Assertions retain a
    provenance status in ``{contract, observed, derived}``; the typed ``hosts``
    view is kept in sync by :meth:`add_service` so existing readers keep working.
    """
    hosts: dict[str, HostInfo] = field(default_factory=dict)
    credentials: list[Credential] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)

    #: property-graph store: (kind, key) -> attributes, and its provenance map.
    nodes_store: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    provenance_store: dict[tuple[str, str], str] = field(default_factory=dict)
    edge_store: list[tuple[tuple[str, str], str, tuple[str, str], str]] = field(default_factory=list)

    #: deterministic node-kind order for read views.
    NODE_KINDS: tuple[str, ...] = ("host", "port", "service", "endpoint", "credential", "session")

    # ── Property-graph operations (M_W) ───────────────────────────────────

    def assert_node(
        self,
        kind: str,
        key: str,
        attrs: dict[str, Any] | None = None,
        provenance: str = "observed",
    ) -> None:
        """Insert or merge a node, keeping the strongest provenance status.

        Precedence: ``contract`` > ``observed`` > ``derived``. Merging never
        downgrades an assertion that was admitted from a stronger source.
        """
        token = (str(kind), str(key))
        merged = dict(self.nodes_store.get(token) or {})
        merged.update(dict(attrs or {}))
        self.nodes_store[token] = merged
        current = self.provenance_store.get(token)
        if current is None or _provenance_rank(provenance) > _provenance_rank(current):
            self.provenance_store[token] = str(provenance)

    def ensure_node(self, kind: str, key: str, attrs: dict[str, Any] | None = None) -> None:
        """Insert a node only when absent; never touches an existing provenance."""
        token = (str(kind), str(key))
        if token in self.nodes_store:
            return
        self.assert_node(kind, key, attrs, "observed")

    def link(
        self,
        src: tuple[str, str],
        rel: str,
        dst: tuple[str, str],
        provenance: str = "observed",
    ) -> None:
        """Add a typed relation between two nodes (idempotent)."""
        edge = ((str(src[0]), str(src[1])), str(rel), (str(dst[0]), str(dst[1])), str(provenance))
        if edge not in self.edge_store:
            self.edge_store.append(edge)

    def nodes(self, kind: str | None = None) -> list[tuple[str, str, dict[str, Any], str]]:
        """Read view: (kind, key, attrs, provenance), deterministically ordered."""
        order = {name: index for index, name in enumerate(self.NODE_KINDS)}
        items = [
            (node_kind, key, dict(attrs), self.provenance_store.get((node_kind, key), "observed"))
            for (node_kind, key), attrs in self.nodes_store.items()
            if kind is None or node_kind == kind
        ]
        items.sort(key=lambda item: (order.get(item[0], len(order)), item[1]))
        return items

    def edges(self) -> list[tuple[tuple[str, str], str, tuple[str, str], str]]:
        """Read view over the typed relations, in insertion order."""
        return list(self.edge_store)

    def provenance_of(self, kind: str, key: str) -> str:
        """Provenance status of an assertion (``""`` when the node is absent)."""
        return self.provenance_store.get((str(kind), str(key)), "")

    def service(self, host: str, port: int) -> ServiceInfo | None:
        """Typed accessor for one service, keyed by host and port."""
        entry = self.hosts.get(str(host))
        if entry is None:
            return None
        return entry.get_service(int(port))

    # ── Host / service operations ─────────────────────────────────────────

    def add_service(self, ip: str, service: ServiceInfo) -> None:
        """Add or update a service on a host (creates host if needed)."""
        if ip not in self.hosts:
            self.hosts[ip] = HostInfo(ip=ip)
        self.hosts[ip].upsert_service(service)
        # keep the property-graph view in sync without touching provenance that
        # an admission already recorded for these nodes.
        self.ensure_node("host", str(ip), {"ip": str(ip)})
        self.ensure_node("port", str(service.port), {"port": int(service.port)})
        self.ensure_node(
            "service",
            f"{ip}:{service.port}",
            {"host": str(ip), "port": int(service.port), "name": service.name},
        )
        self.link(("host", str(ip)), "host-has-port", ("port", str(service.port)), "observed")
        self.link(("port", str(service.port)), "port-runs-service", ("service", f"{ip}:{service.port}"), "observed")


    def update_service_confidence(
        self, ip: str, port: int, delta: float, evidence: str
    ) -> None:
        """Bump confidence for a specific service and record evidence."""
        host = self.hosts.get(ip)
        if host is None:
            return
        svc = host.get_service(port)
        if svc is not None:
            svc.bump_confidence(delta, evidence)

    def get_services_above_confidence(self, threshold: float) -> list[ServiceInfo]:
        """Return all services across all hosts with confidence >= threshold."""
        result: list[ServiceInfo] = []
        for host in self.hosts.values():
            result.extend(s for s in host.services if s.confidence >= threshold)
        return result

    def get_versioned_services(self) -> list[ServiceInfo]:
        """Return all services that have a non-empty version string."""
        result: list[ServiceInfo] = []
        for host in self.hosts.values():
            result.extend(s for s in host.services if s.version)
        return result

    def get_active_services(self, threshold: float = 0.0, now: float | None = None) -> list[ServiceInfo]:
        if now is None:
            now = time.time()
        result: list[ServiceInfo] = []
        for host in self.hosts.values():
            for svc in host.services:
                if svc.is_active(now) and svc.confidence >= threshold:
                    result.append(svc)
        return result

    def detect_conflicts(self) -> list[dict]:
        import re
        version_re = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
        conflicts = []
        for ip, host in self.hosts.items():
            port_svcs: dict[int, list[ServiceInfo]] = {}
            for svc in host.services:
                port_svcs.setdefault(svc.port, []).append(svc)

            for port, svcs in port_svcs.items():
                versions = set()
                sources = set()
                for svc in svcs:
                    if svc.version and svc.version != "unknown":
                        versions.add(svc.version)
                    for src in svc.provenance_sources:
                        sources.add(src)
                    for ev in svc.evidence:
                        for match in version_re.findall(str(ev)):
                            versions.add(match)
                if len(versions) > 1:
                    conflicts.append({
                        "ip": ip,
                        "port": port,
                        "conflicting_versions": sorted(list(versions)),
                        "sources": sorted(list(sources))
                    })
        return conflicts

    def to_context_dict(self, service_key: str | None = None) -> dict:
        now = time.time()
        result: dict[str, Any] = {"hosts": {}}
        if service_key:
            parts = service_key.split(":")
            if len(parts) in {3, 4}:
                target_ip, target_port_str = parts[:2]
                target_product = parts[-1]
                try:
                    target_port = int(target_port_str)
                except ValueError:
                    return result

                host = self.hosts.get(target_ip)
                if host:
                    for svc in host.services:
                        if svc.port == target_port and svc.name == target_product:
                            result["hosts"][target_ip] = {
                                "ip": host.ip,
                                "os_hint": host.os_hint,
                                "services": [asdict(svc)]
                            }
                            break
            return result

        for ip, host in self.hosts.items():
            active_svcs = [svc for svc in host.services if svc.is_active(now)]
            if active_svcs:
                result["hosts"][ip] = {
                    "ip": host.ip,
                    "os_hint": host.os_hint,
                    "services": [asdict(svc) for svc in active_svcs]
                }
        return result

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe dict for storage in PentestState."""
        return {
            "hosts": {
                ip: {
                    "ip": h.ip,
                    "os_hint": h.os_hint,
                    "os_confidence": h.os_confidence,
                    "services": [asdict(s) for s in h.services],
                }
                for ip, h in self.hosts.items()
            },
            "credentials": [asdict(c) for c in self.credentials],
            "sessions": [asdict(s) for s in self.sessions],
            "graph": {
                "nodes": [
                    {"kind": kind, "key": key, "attrs": dict(attrs), "provenance": provenance}
                    for kind, key, attrs, provenance in self.nodes()
                ],
                "edges": [
                    {"src": list(src), "rel": rel, "dst": list(dst), "provenance": provenance}
                    for src, rel, dst, provenance in self.edges()
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorldState:
        """Reconstruct WorldState from a serialized dict."""
        if not data:
            return cls()
        ws = cls()
        for ip, hd in data.get("hosts", {}).items():
            host = HostInfo(
                ip=hd["ip"],
                os_hint=hd.get("os_hint", ""),
                os_confidence=hd.get("os_confidence", 0.3),
                services=[ServiceInfo(**sd) for sd in hd.get("services", [])],
            )
            ws.hosts[ip] = host
        ws.credentials = [Credential(**cd) for cd in data.get("credentials", [])]
        ws.sessions = [Session(**sd) for sd in data.get("sessions", [])]

        graph = data.get("graph") or {}
        for node in graph.get("nodes") or []:
            ws.assert_node(
                str(node.get("kind") or ""),
                str(node.get("key") or ""),
                dict(node.get("attrs") or {}),
                str(node.get("provenance") or "observed"),
            )
        for edge in graph.get("edges") or []:
            src = edge.get("src") or ("", "")
            dst = edge.get("dst") or ("", "")
            ws.link((str(src[0]), str(src[1])), str(edge.get("rel") or ""), (str(dst[0]), str(dst[1])),
                    str(edge.get("provenance") or "observed"))
        if not graph:
            # Legacy payloads predate the graph: rebuild the typed view.
            for ip, host in ws.hosts.items():
                for service in host.services:
                    ws.add_service(ip, service)
        return ws

    @classmethod
    def from_port_services(
        cls, target_ip: str, port_services: dict, os_info: str | None = None
    ) -> WorldState:
        """
        Backward-compatible builder: convert the existing flat port_services dict
        into a WorldState with initial confidence=0.5 for all services.

        port_services format:
            {"80": {"name": "apache", "version": "2.4.49", "accessibility": "open"}}
        """
        ws = cls()
        host = HostInfo(
            ip=target_ip,
            os_hint=os_info or "",
            os_confidence=0.4 if os_info else 0.0,
        )
        for port_str, info in port_services.items():
            port = int(port_str)
            version = info.get("version", "")
            # Version present → higher initial confidence
            confidence = 0.6 if version else 0.4
            svc = ServiceInfo(
                port=port,
                name=info.get("name", ""),
                version=version,
                accessibility=info.get("accessibility", "open"),
                confidence=confidence,
                evidence=[f"Initial recon identified {info.get('name', '?')} {version} on port {port}"],
            )
            host.services.append(svc)
        ws.hosts[target_ip] = host
        return ws

    def to_summary(self) -> str:
        """Compact text summary for LLM context injection."""
        lines = []
        for ip, host in self.hosts.items():
            lines.append(f"Host {ip} (OS: {host.os_hint or '?'}, conf={host.os_confidence:.1f}):")
            for service in host.services:
                lines.append(
                    f"  :{service.port}/{service.protocol} {service.name} {service.version} "
                    f"[{service.accessibility}] conf={service.confidence:.2f}"
                )
        if self.credentials:
            lines.append("Credentials:")
            for c in self.credentials:
                lines.append(f"  {c.username}@{c.target_service} (verified={c.verified})")
        if self.sessions:
            lines.append("Sessions:")
            for session in self.sessions:
                lines.append(
                    f"  {session.session_type} → {session.target_ip}:{session.target_port} "
                    f"as {session.privilege_level}"
                )
        return "\n".join(lines)
