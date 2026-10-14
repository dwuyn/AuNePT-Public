"""
src/agents/recon.py
────────────────────
Contract-driven reconnaissance — Algorithm 1, steps 1–6 (paper §3.2).

Reconnaissance is deterministic and model-free. It starts only from
``C_A``-authorized endpoints and performs exactly:

    1. ``AssertPort(C_A.host, C.port)``              — seed ``M_W``
    2. one connect/handshake banner grab             — bounded by ``K1``
    3. protocol-family classification                — TLS/SSH/HTTP/WS/raw
    4. HTTP-family endpoint probing over the closed discovery tuple
       ``("/", "/robots.txt", "/.well-known/")``     — one short GET each
    5. non-HTTP families: the handshake only
    6. observation admission                         — ``Admit`` -> ``Update``

No fuzzing, wordlists, path brute force, full-port scans or LLM calls happen
here. Derived identity (``InferServiceArchetype`` over an uninformative banner)
enters ``M_W`` through ``Derive``.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import re
import socket
import time
from typing import Any, Dict, Mapping, Sequence

from src.design.budgets import PhaseBudgets
from src.design.contract import RuntimeContract
from src.design.state import admit, assert_port, derive, update
from src.memory.world_state import WorldState
from src.pipeline.case_contract import normalize_protocol_family
from src.pipeline.protocol_detect import PROTOCOL_FAMILIES, detect_protocol
from src.state import PentestState, state_int

logger = logging.getLogger(__name__)

#: The closed HTTP discovery tuple ``U_disc`` (paper §3.2).
DISCOVERY_PATHS: tuple[str, ...] = ("/", "/robots.txt", "/.well-known/")

#: Phase-budget scope consumed by every recon interaction (K1 = reconnaissance).
K1_SCOPE = "K1"

_BANNER_LIMIT = 4096
_BODY_LIMIT = 4096
_DEFAULT_TIMEOUT = 3.0

#: Identity names too generic to count as a classified archetype.
_GENERIC_IDENTITY = {"", "http", "https", "tcp", "raw-tcp", "unknown", "generic", "n/a"}

_VERSION_RE = re.compile(r"(\d+\.\d+[\d.]*)")
_PRODUCT_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.+\-]*)/([0-9][0-9A-Za-z.\-]*)")
_SSH_PRODUCT_RE = re.compile(r"SSH-\d+\.\d+-(\S+)")

#: Deterministic service-signature table used by ``InferServiceArchetype``.
_ARCHETYPE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("jolokia", "jolokia"),
    ("werkzeug", "werkzeug"),
    ("tornado", "tornado"),
    ("nginx", "nginx"),
    ("apache", "apache"),
    ("openssh", "openssh"),
    ("basehttp", "python-http"),
    ("jetty", "jetty"),
    ("tomcat", "tomcat"),
    ("express", "express"),
)


class TransportError(RuntimeError):
    """Raised when a connect or handshake fails at the transport layer."""


def grab_banner(host: str, port: int, *, timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    """Grab one connect/handshake banner.

    Opens a single TCP connection and performs one read of whatever bytes are
    already available; returns ``b""`` when the port is silent, and raises
    :class:`TransportError` when the connection itself fails. No payload is
    sent and no retry ladder is performed.
    """
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout)) as sock:
            sock.settimeout(float(timeout))
            try:
                return sock.recv(_BANNER_LIMIT)
            except (socket.timeout, TimeoutError):
                return b""
    except OSError as exc:
        raise TransportError(f"connect to {host}:{port} failed: {exc}") from exc


class HttpSession:
    """Issues one bounded GET per call against a single ``host:port``."""

    def __init__(self, host: str, port: int, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._host = str(host)
        self._port = int(port)
        self._timeout = float(timeout)

    def get(self, path: str) -> dict[str, Any]:
        """Perform a single short GET and return status, headers and body."""
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request(
                "GET", path,
                headers={"Connection": "close", "User-Agent": "aunept-recon"},
            )
            response = conn.getresponse()
            body = response.read(_BODY_LIMIT)
            headers = {str(key).lower(): str(value) for key, value in response.getheaders()}
            return {
                "path": str(path),
                "status": int(response.status),
                "headers": headers,
                "body": body,
            }
        finally:
            conn.close()


def bounded_discovery(
    host: str,
    port: int,
    family: str,
    *,
    budget: PhaseBudgets,
    session: HttpSession,
) -> list[dict[str, Any]]:
    """Probe the endpoints authorized for ``family``, bounded by ``K1``.

    HTTP-family services get exactly the closed discovery tuple ``U_disc`` as
    single short GETs; every other family gets nothing beyond the handshake
    already taken. Each interaction consumes one unit of ``K1`` and the ladder
    stops as soon as the budget is exhausted.
    """
    if normalize_protocol_family(family) != "http":
        return []

    observations: list[dict[str, Any]] = []
    for path in DISCOVERY_PATHS:
        if not _k1_available(budget):
            break
        _consume_k1(budget)
        try:
            response = session.get(path)
        except (OSError, http.client.HTTPException) as exc:
            observations.append(_http_observation(host, port, path, error=str(exc)))
            break
        observations.append(_http_observation(host, port, path, response=response))
    return observations


def infer_service_archetype(
    family: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deterministic service-signature matching over admitted observations.

    Normalizes an uninformative banner through static signatures over the
    returned status codes, headers and body snippets (paper §3.2). Returns an
    empty mapping when no signature matches.
    """
    corpus: list[str] = [str(family)]
    for observation in observations:
        corpus.append(str(observation.get("banner") or ""))
        corpus.append(str(observation.get("name") or ""))
        for item in observation.get("evidence") or []:
            corpus.append(str(item))
    haystack = " ".join(corpus).lower()
    for signature, archetype in _ARCHETYPE_SIGNATURES:
        if signature in haystack:
            return {
                "name": archetype,
                "version_evidence_class": "inferred",
                "evidence": [f"InferServiceArchetype:{signature}"],
            }
    return {}


def recon_node(state: PentestState) -> Dict[str, Any]:
    """LangGraph node: deterministic, contract-driven reconnaissance.

    Asserts the contract port, grabs a single handshake, classifies the
    protocol family, probes only the authorized endpoints (bounded by ``K1``),
    and admits every observation through ``Admit`` -> ``Update``. Sets
    ``recon_complete=True`` when the ladder finishes or ``K1`` is exhausted.
    """
    contract = _resolve_contract(state)
    budgets = _resolve_budgets(state, contract)
    ws = WorldState.from_dict(state.get("world_state", {}) or {})
    assert_port(contract, ws)

    host = str(contract.target_host)
    port = int(contract.target_port)
    timeout = float(state.get("recon_banner_timeout") or _DEFAULT_TIMEOUT)

    phase_timestamps = dict(state.get("phase_timestamps", {}))
    phase_timestamps.setdefault("recon_start", time.time())

    # ── steps 1–3: assert port, one handshake, classify the family ────────────
    banner = b""
    transport_error = ""
    if _k1_available(budgets):
        _consume_k1(budgets)
        try:
            banner = grab_banner(host, port, timeout=timeout)
        except TransportError as exc:
            transport_error = str(exc)

    family = _classify_family(banner, port, contract)
    name, version, evidence_class = _scan_identity(banner.decode("latin-1", "replace"))
    handshake = {
        "host": host,
        "target_ip": host,
        "port": port,
        "protocol_family": family,
        "name": name,
        "version": version,
        "version_evidence_class": evidence_class,
        "banner": banner.decode("latin-1", "replace")[:512],
        "accessibility": "closed" if transport_error else "open",
        "confidence": 0.6 if version else 0.45,
        "evidence": [
            f"handshake transport_error: {transport_error}" if transport_error
            else f"handshake banner: {banner[:120]!r}"
        ],
    }
    ws = update(ws, admit(handshake, contract))
    observations: list[dict[str, Any]] = [handshake]

    # ── steps 4–5: bounded discovery (HTTP-family only) ───────────────────────
    if not transport_error and normalize_protocol_family(family) == "http":
        session = HttpSession(host, port, timeout=timeout)
        for observation in bounded_discovery(host, port, family, budget=budgets, session=session):
            ws = update(ws, admit(observation, contract))
            observations.append(observation)

    # ── step 6: derived identity from an uninformative banner ─────────────────
    derived: list[dict[str, Any]] = []
    if _is_generic_identity(name):
        archetype = infer_service_archetype(family, observations)
        if archetype:
            delta = derive(
                {
                    "host": host,
                    "port": port,
                    "protocol_family": family,
                    "name": archetype["name"],
                    "version_evidence_class": archetype["version_evidence_class"],
                    "confidence": 0.4,
                    "evidence": archetype["evidence"],
                },
                source="InferServiceArchetype",
                from_keys=[f"{host}:{port}"],
            )
            ws = update(ws, delta)
            derived.append(delta.to_dict())

    phase_timestamps["recon_complete_time"] = time.time()
    logger.info(
        "Recon complete: host=%s port=%s family=%s k1_spent=%s",
        host, port, family, budgets.spent(K1_SCOPE),
    )
    return {
        "messages": list(state.get("messages", [])),
        "current_phase": "recon",
        "recon_complete": True,
        "recon_step_count": state_int(state, "recon_step_count", 0) + 1,
        "world_state": ws.to_dict(),
        "recon_observations": observations,
        "recon_derived": derived,
        "phase_timestamps": phase_timestamps,
    }


def route_recon(state: PentestState) -> str:
    """Conditional edge: advance once recon has completed."""
    if state.get("current_phase") == "done":
        return "planning"
    if state.get("recon_complete"):
        return "planning"
    return "recon"


# ── internals ─────────────────────────────────────────────────────────────────


def _k1_available(budget: PhaseBudgets) -> bool:
    return int(budget.spent(K1_SCOPE)) < int(budget.K1)


def _consume_k1(budget: PhaseBudgets) -> None:
    budget.record(K1_SCOPE)


def _classify_family(banner: bytes, port: int, contract: RuntimeContract) -> str:
    """Classify the family from the banner, falling back to the declared one.

    A port that returns no signature is ``raw-tcp`` unless ``C_A`` declares a
    permitted service interface for it, in which case the declared family guides
    the (still bounded) probe.
    """
    family, _detail = detect_protocol(banner, port)
    if family != "raw-tcp":
        return family
    for declared in contract.protocol_families:
        candidate = normalize_protocol_family(declared)
        if candidate in PROTOCOL_FAMILIES and candidate != "raw-tcp":
            return candidate
    return "raw-tcp"


def _scan_identity(text: str) -> tuple[str, str, str]:
    """Extract ``(name, version, version_evidence_class)`` from a banner."""
    body = (text or "").strip()
    if not body:
        return "", "", "unknown"
    product = _PRODUCT_RE.search(body)
    if product:
        return product.group(1).lower(), product.group(2), "explicit"
    ssh = _SSH_PRODUCT_RE.search(body)
    if ssh:
        token = ssh.group(1)
        name = token.split("_", 1)[0].lower() if "_" in token else token.lower()
        version = _VERSION_RE.search(token)
        return name, (version.group(1) if version else ""), ("explicit" if version else "unknown")
    head = body.split()[0]
    return head.lower(), "", "unknown"


def _http_observation(
    host: str,
    port: int,
    path: str,
    *,
    response: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Build a ``Delta``-ready observation for one HTTP discovery request."""
    if response is None:
        return {
            "host": host, "target_ip": host, "port": int(port),
            "protocol_family": "http",
            "name": "", "version": "", "version_evidence_class": "unknown",
            "banner": "", "accessibility": "closed",
            "endpoint": path, "status": 0,
            "evidence": [f"GET {path} failed: {error}"],
        }
    headers = dict(response.get("headers") or {})
    server = str(headers.get("server") or "")
    name, version = _split_product(server)
    body = response.get("body") or b""
    digest = hashlib.sha256(bytes(body)).hexdigest()[:16]
    banner = f"HTTP {response['status']}"
    if server:
        banner = f"{banner} Server: {server}"
    return {
        "host": host, "target_ip": host, "port": int(port),
        "protocol_family": "http",
        "name": name or "http",
        "version": version,
        "version_evidence_class": "explicit" if version else "unknown",
        "banner": banner[:512],
        "accessibility": "open",
        "confidence": 0.6 if version else 0.45,
        "endpoint": path,
        "status": int(response["status"]),
        "evidence": [
            f"GET {path} -> {response['status']} {server}".strip(),
            f"body-sha256:{digest}",
            f"body: {bytes(body)[:200]!r}",
        ],
    }


def _split_product(text: str) -> tuple[str, str]:
    match = _PRODUCT_RE.search(text or "")
    if match:
        return match.group(1).lower(), match.group(2)
    token = (text or "").split(" ")[0].strip()
    return token.lower(), ""


def _is_generic_identity(name: str) -> bool:
    return str(name or "").strip().lower() in _GENERIC_IDENTITY


def _resolve_contract(state: PentestState) -> RuntimeContract:
    """Return ``C_A`` from the state, rebuilding a minimal one when absent."""
    raw = state.get("contract")
    if isinstance(raw, RuntimeContract):
        return raw
    host = str(state.get("target_ip") or "")
    raw_port = state.get("target_port")
    port = str(raw_port) if raw_port not in (None, "") else None
    allowed_ports = (int(raw_port),) if raw_port not in (None, "") else ()
    return RuntimeContract(
        target_host=host,
        target_port=port,
        allowed_hostnames=(host,) if host else (),
        allowed_networks=("127.0.0.0/8",),
        allowed_ports=allowed_ports,
    )


def _resolve_budgets(state: PentestState, contract: RuntimeContract) -> PhaseBudgets:
    """Return the phase budgets the state carries, defaulting to ``C_A``'s."""
    raw = state.get("phase_budgets")
    if isinstance(raw, PhaseBudgets):
        return raw
    limits = contract.limits
    if isinstance(raw, Mapping):
        return PhaseBudgets(
            K1=int(raw.get("K1", limits.max_tool_calls)),
            K2=int(raw.get("K2", limits.max_cves_per_service)),
            K3=int(raw.get("K3", limits.max_attempts_per_candidate)),
            B_max=int(raw.get("B_max", limits.max_total_tokens)),
        )
    return PhaseBudgets.from_limits(limits)
