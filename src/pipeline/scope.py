"""
src/pipeline/scope.py
─────────────────────
Full endpoint/scope validation applied to *every* procedure stage
(setup / execute / verify / cleanup).

Commands are rendered as structured argument arrays, never free-form shell
strings. Each token is parsed for hostnames, IPv4/IPv6 literals, URLs, ports,
schemes, redirect targets, and callback endpoints. Hostnames are resolved
immediately and every resulting address must remain within the manifest scope.
Unresolved placeholders are rejected.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import urlparse

from src.pipeline.manifest import Scope

# ── Regexes ───────────────────────────────────────────────────────────────────
_IPV4_RE = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3})\b")
_IPV6_RE = re.compile(r"(?:\[([0-9a-fA-F:]+)\])|(?<![0-9a-fA-F:])([0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){2,7})(?![0-9a-fA-F:])")
_URL_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://[^\s\"'<>]+)")
_HOSTNAME_RE = re.compile(r"(?![\d.]+$)([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)+)")

# Unresolved placeholder patterns (renderer should have substituted these).
PLACEHOLDER_RE = re.compile(
    r"(\$\{[A-Za-z_]\w*\})"
    r"|(\{[A-Za-z_][\w]*\})"
    r"|(<[A-Za-z_][\w]*>)"
    r"|(\$\([A-Za-z_]\w*\))"
    r"|(__[A-Z0-9_]+__)"
)

Resolver = Callable[[str], list[str]]


def default_resolver(hostname: str) -> list[str]:
    """Resolve a hostname to a list of address strings via the system resolver."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return []
    out: list[str] = []
    for info in infos:
        addr = str(info[4][0])
        if addr not in out:
            out.append(addr)
    return out


@dataclass
class Endpoint:
    raw: str
    kind: str          # hostname|ipv4|ipv6|url|callback
    host: str = ""
    port: int | None = None
    scheme: str = ""

    def __str__(self) -> str:
        return self.raw


@dataclass
class ScopeDecision:
    allowed: bool
    blocked_endpoints: list[str] = field(default_factory=list)
    unresolved_placeholders: list[str] = field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class ScopeValidator:
    """Validates that every endpoint in rendered arguments stays within scope."""

    def __init__(self, scope: Scope, resolver: Resolver | None = None) -> None:
        self.scope = scope
        self.resolver = resolver or default_resolver
        self._networks = [
            ipaddress.ip_network(n, strict=False) for n in scope.allowed_networks
        ]
        self._callbacks = {self._normalize_ip(c) for c in scope.callback_endpoints}
        self._hostnames = {h.lower() for h in scope.allowed_hostnames}
        self._ports = set(scope.allowed_ports)
        self._schemes = {s.lower() for s in scope.allowed_schemes}

    @staticmethod
    def _normalize_ip(value: str) -> str:
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            return value.strip().lower()

    # ── Endpoint extraction ───────────────────────────────────────────────────
    def extract(self, tokens: Iterable[str]) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        consumed_spans: list[tuple[int, int]] = []
        text = " ".join(str(t) for t in tokens)

        # URLs first.
        for m in _URL_RE.finditer(text):
            endpoints.append(self._endpoint_from_url(m.group(1)))
            consumed_spans.append(m.span())
        # IPv6 (including bracketed).
        for m in _IPV6_RE.finditer(text):
            if self._in_spans(m.start(), consumed_spans):
                continue
            val = m.group(1) or m.group(2)
            endpoints.append(Endpoint(raw=val, kind="ipv6", host=val))
            consumed_spans.append(m.span())
        # IPv4.
        for m in _IPV4_RE.finditer(text):
            if self._in_spans(m.start(), consumed_spans):
                continue
            val = m.group(1)
            endpoints.append(Endpoint(raw=val, kind="ipv4", host=val))
            consumed_spans.append(m.span())
        # Bare hostnames.
        for m in _HOSTNAME_RE.finditer(text):
            if self._in_spans(m.start(), consumed_spans):
                continue
            val = m.group(1)
            endpoints.append(Endpoint(raw=val, kind="hostname", host=val))

        return self._dedupe(endpoints)

    @staticmethod
    def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
        return any(s <= pos < e for s, e in spans)

    @staticmethod
    def _dedupe(endpoints: list[Endpoint]) -> list[Endpoint]:
        seen: set[tuple[str, str]] = set()
        out: list[Endpoint] = []
        for ep in endpoints:
            key = (ep.kind, ep.host.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(ep)
        return out

    def _endpoint_from_url(self, url: str) -> Endpoint:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port
        scheme = parsed.scheme.lower()
        # Strip IPv6 brackets already handled by urlparse.
        "ipv6" if ":" in host and host.count(":") >= 2 else ("ipv4" if self._is_ipv4(host) else "hostname")
        return Endpoint(raw=url, kind="url", host=host, port=port, scheme=scheme)

    @staticmethod
    def _is_ipv4(value: str) -> bool:
        try:
            ipaddress.IPv4Address(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    # ── Placeholder detection ───────────────────────────────────────────────────
    def find_placeholders(self, tokens: Iterable[str]) -> list[str]:
        found: list[str] = []
        for tok in tokens:
            for m in PLACEHOLDER_RE.finditer(str(tok)):
                found.append(m.group(0))
        return list(dict.fromkeys(found))

    # ── Core validation ────────────────────────────────────────────────────────
    def validate_tokens(self, tokens: Iterable[str], *, stage: str = "") -> ScopeDecision:
        tokens = [str(t) for t in tokens]
        placeholders = self.find_placeholders(tokens)
        if placeholders:
            return ScopeDecision(
                allowed=False,
                unresolved_placeholders=placeholders,
                reason=f"Unresolved placeholder(s) in {stage or 'command'}: {placeholders}",
            )

        blocked: list[str] = []
        reason = ""
        for ep in self.extract(tokens):
            ok, why = self._check_endpoint(ep)
            if not ok:
                blocked.append(ep.raw)
                reason = why
        if blocked:
            return ScopeDecision(
                allowed=False,
                blocked_endpoints=blocked,
                reason=f"Endpoint(s) outside scope in {stage or 'command'}: {blocked} ({reason})",
            )
        return ScopeDecision(allowed=True)

    def validate_args(self, args: list[str], *, stage: str = "execute") -> ScopeDecision:
        return self.validate_tokens(args, stage=stage)

    def validate_url_redirect(self, url: str, fetcher: Callable[[str], str | None] | None) -> ScopeDecision:
        """If *fetcher* follows redirects, every hop's host must stay in scope."""
        if fetcher is None:
            # Without a fetcher we still validate the URL's own host.
            return self.validate_tokens([url], stage="redirect")
        visited: list[str] = []
        current = url
        for _ in range(5):
            if current in visited:
                break
            visited.append(current)
            dec = self.validate_tokens([current], stage="redirect")
            if not dec:
                return dec
            nxt = fetcher(current)
            if not nxt:
                break
            current = nxt
        return ScopeDecision(allowed=True)

    # ── Per-endpoint checks ────────────────────────────────────────────────────
    def _check_endpoint(self, ep: Endpoint) -> tuple[bool, str]:
        if not ep.host:
            return True, ""
        # Port / scheme checks for URLs.
        if ep.kind == "url":
            if self._schemes and ep.scheme not in self._schemes:
                return False, f"scheme '{ep.scheme}' not allowed"
            if ep.port is not None and self._ports and ep.port not in self._ports:
                return False, f"port {ep.port} not allowed"
        # Callback endpoints (egress targets such as the operator machine).
        if self._normalize_ip(ep.host) in self._callbacks:
            return True, ""
        # Explicit allowed hostname (must still resolve in-scope if networks are defined).
        if ep.host.lower() in self._hostnames:
            if not self._networks:
                return True, ""
            addrs = self._resolve(ep.host)
            if not addrs:
                return True, ""
            for addr in addrs:
                if not self._addr_in_networks(addr):
                    return False, f"allowed hostname '{ep.host}' resolved to out-of-scope {addr}"
            return True, ""
        # IP literals must be in-network.
        if self._is_ip(ep.host):
            return self._addr_in_networks(ep.host), f"address '{ep.host}' outside allowed networks"
        # Other hostnames: resolve and require all addresses in-network.
        addrs = self._resolve(ep.host)
        if not addrs:
            return False, f"hostname '{ep.host}' unresolved / foreign"
        for addr in addrs:
            if not self._addr_in_networks(addr):
                return False, f"hostname '{ep.host}' resolved to out-of-scope {addr}"
        return True, ""

    def _resolve(self, hostname: str) -> list[str]:
        if self._is_ip(hostname):
            return [hostname]
        return self.resolver(hostname)

    def _addr_in_networks(self, addr: str) -> bool:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        return any(ip in net for net in self._networks)

    # ── Convenience: validate all four stages of a procedure ───────────────────
    def validate_procedure(
        self,
        *,
        setup: list[list[str]] | None = None,
        execute: list[list[str]] | None = None,
        verify: list[list[str]] | None = None,
        cleanup: list[list[str]] | None = None,
    ) -> dict[str, ScopeDecision]:
        results: dict[str, ScopeDecision] = {}
        for stage, cmds in (("setup", setup), ("execute", execute), ("verify", verify), ("cleanup", cleanup)):
            if not cmds:
                continue
            for args in cmds:
                dec = self.validate_args(args, stage=stage)
                results.setdefault(stage, dec)
                if not dec:
                    return results
        return results
