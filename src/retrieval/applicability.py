"""
Applicability assessment and shortlist construction.

Verdict vocabulary follows the design (v75, §3.2 ``Evidence-Tiered Hypothesis
Retrieval``) and is tri-state for *compatibility*:

    confirmed  product identity resolved AND observed version satisfies the
               affected range (``version_match == "yes"``) AND score >= 0.72
    plausible  product identity resolved, but the version is unknown or only
               partially constrained
    unknown    product identity itself is unresolved (no product key matched
               the advisory text)
    rejected   an explicit predicate mismatch (``no``) or score < 0.50 —
               such candidates are discarded rather than attempted
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from packaging import version

from src.memory.world_state import WorldState
from src.retrieval.models import (
    ApplicabilityAssessment,
    AuthoritativeRecord,
    PocCandidate,
    ProcedureSnippet,
    ProductFingerprint,
)

_TRUST = {
    "vendor": 1.0,
    "kev": 0.95,
    "nvd": 0.85,
    "cvemap": 0.85,
    "exploitdb": 0.80,
    "github": 0.65,
    "google": 0.35,
}


def _tri_to_num(value: str) -> float:
    return {"yes": 1.0, "unknown": 0.5, "no": 0.0}.get(value, 0.5)


def _match_version(target_version: str, ranges: list[dict[str, Any]]) -> str:
    if not target_version:
        return "unknown"
    if not ranges:
        return "unknown"
    try:
        current = version.parse(target_version)
    except Exception:
        return "unknown"
    any_unknown = False
    for item in ranges:
        exact = str(item.get("version") or "").strip()
        min_ver = str(item.get("min_version") or "").strip()
        max_ver = str(item.get("max_version") or "").strip()
        if exact:
            try:
                return "yes" if version.parse(exact) == current else "no"
            except Exception:
                any_unknown = True
                continue
        try:
            if min_ver and current < version.parse(min_ver):
                continue
            if max_ver and current > version.parse(max_ver):
                continue
            return "yes"
        except Exception:
            any_unknown = True
    return "unknown" if any_unknown else "no"


def _record_text(record: AuthoritativeRecord) -> str:
    return " ".join([
        record.title.lower(),
        record.description.lower(),
        " ".join(record.references).lower(),
        " ".join(f"{item.get('vendor','')} {item.get('product','')}" for item in record.affected_ranges).lower(),
    ])


def _product_identity_resolved(fp: ProductFingerprint, record: AuthoritativeRecord) -> bool:
    """v75 tri-state: the product identity is resolved iff a product key from the
    fingerprint matches the advisory text; vendor-only or empty fingerprints are
    ``unknown`` rather than ``plausible``."""
    product = (fp.product or "").lower()
    if not product:
        return False
    return product in _record_text(record)


def _match_cpe(fp: ProductFingerprint, record: AuthoritativeRecord) -> str:
    text = _record_text(record)
    product = (fp.product or fp.raw_service).lower()
    vendor = (fp.vendor or "").lower()
    if product and product in text:
        return "yes" if not vendor or vendor in text else "unknown"
    if vendor and vendor in text:
        return "unknown"
    return "unknown" if not product else "no"


def _match_platform(fp: ProductFingerprint, record: AuthoritativeRecord) -> str:
    if not fp.platform_hints or not record.platform_hints:
        return "unknown"
    if set(fp.platform_hints).intersection(record.platform_hints):
        return "yes"
    return "no"


_PLATFORM_ALIASES: dict[str, str] = {
    "win": "windows", "win32": "windows", "win64": "windows",
    "w32": "windows", "w64": "windows", "winnt": "windows",
    "gnu/linux": "linux",
    "osx": "macos", "darwin": "macos", "mac": "macos",
}

_FOREIGN_IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")


def _check_snippet_target_assumptions(snippet: ProcedureSnippet, fp: ProductFingerprint) -> str:
    """Cross-check snippet target_assumptions against fingerprint platform hints."""
    if not snippet.target_assumptions:
        return "unknown"

    fp_platforms = {h.lower().strip() for h in fp.platform_hints}
    expanded_fp = set(fp_platforms)
    for hint in fp_platforms:
        if hint in _PLATFORM_ALIASES:
            expanded_fp.add(_PLATFORM_ALIASES[hint])

    snippet_platforms: set[str] = set()
    for assumption in snippet.target_assumptions:
        normalized = assumption.lower().strip()
        snippet_platforms.add(normalized)
        if normalized in _PLATFORM_ALIASES:
            snippet_platforms.add(_PLATFORM_ALIASES[normalized])

    if snippet_platforms and expanded_fp:
        if snippet_platforms & expanded_fp:
            return "yes"
        return "no"

    return "unknown"


def _check_foreign_target_ips(snippet: ProcedureSnippet, fp: ProductFingerprint) -> tuple[str, list[str]]:
    """Check for hardcoded foreign target IPs in snippet commands.

    Returns (signal, offending_ips) where signal is 'clean', 'warning', or
    'no_good_commands'.
    """
    if not snippet.commands:
        return "clean", []

    allowed = {ip for ip in (fp.target_ip, "127.0.0.1", "0.0.0.0") if ip}
    offending_ips: set[str] = set()
    clean_commands = 0

    for cmd in snippet.commands:
        ips_in_cmd = set(_FOREIGN_IP_RE.findall(cmd))
        foreign = ips_in_cmd - allowed
        if foreign:
            offending_ips.update(foreign)
        else:
            clean_commands += 1

    if not offending_ips:
        return "clean", []
    if clean_commands > 0:
        return "warning", sorted(offending_ips)
    return "no_good_commands", sorted(offending_ips)


def _match_auth(ws: WorldState, fp: ProductFingerprint, record: AuthoritativeRecord) -> str:
    if record.auth_hint != "required":
        return "yes"
    if ws.credentials:
        return "yes"
    if fp.auth_hint == "required":
        return "unknown"
    return "unknown"


def _match_network(ws: WorldState, fp: ProductFingerprint) -> str:
    host = ws.hosts.get(fp.target_ip)
    if not host:
        return "unknown"
    service = host.get_service(fp.port)
    if not service:
        return "unknown"
    return "yes" if service.accessibility == "open" else "no" if service.accessibility == "closed" else "unknown"


def _procedure_ready(snippet: ProcedureSnippet) -> bool:
    return bool(
        snippet.commands
        or snippet.dependencies
        or snippet.setup_commands
        or snippet.verify_commands
        or snippet.usage_notes
    )


def _estimated_cost(candidate: PocCandidate, snippet: ProcedureSnippet) -> float:
    cost = 1.0
    if candidate.source == "github":
        cost += 0.25
    if candidate.source == "google":
        cost += 0.35
    if len(snippet.placeholders) >= 4:
        cost += 0.15
    if len(snippet.dependencies) >= 3:
        cost += 0.15
    if any(token in (candidate.language or "").lower() for token in ("ruby", "perl", "java")):
        cost += 0.1
    return round(cost, 3)


def flatten_assessment_map(assessments: list[ApplicabilityAssessment]) -> dict[str, ApplicabilityAssessment]:
    return {item.candidate_id: item for item in assessments}


def assess_candidates(
    ws: WorldState,
    fingerprints: list[ProductFingerprint],
    records: list[AuthoritativeRecord],
    candidates: list[PocCandidate],
    snippets: list[ProcedureSnippet],
) -> list[ApplicabilityAssessment]:
    snippet_map = {item.candidate_id: item for item in snippets}
    by_cve: dict[str, list[PocCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_cve[candidate.cve_id].append(candidate)

    assessments: list[ApplicabilityAssessment] = []
    for record in records:
        matching_fps = [
            fp for fp in fingerprints
            if (fp.product and fp.product.lower() in (record.title + " " + record.description).lower())
            or fp.raw_service.lower() in (record.title + " " + record.description).lower()
        ] or fingerprints[:1]
        fp = matching_fps[0] if matching_fps else None
        if fp is None:
            continue
        for candidate in by_cve.get(record.cve_id, []):
            snippet = snippet_map.get(candidate.candidate_id, ProcedureSnippet(candidate_id=candidate.candidate_id))
            version_match = _match_version(fp.version, record.affected_ranges)
            cpe_match = _match_cpe(fp, record)
            platform_match = _match_platform(fp, record)
            snippet_assumption_match = _check_snippet_target_assumptions(snippet, fp)
            foreign_ip_signal, foreign_ips = _check_foreign_target_ips(snippet, fp)
            if snippet_assumption_match == "no" and platform_match != "no":
                platform_match = "no"
            auth_match = _match_auth(ws, fp, record)
            network_match = _match_network(ws, fp)
            procedure_ready = _procedure_ready(snippet)
            trust_score = _TRUST.get(record.source, _TRUST.get(candidate.source, 0.5))
            estimated_cost = _estimated_cost(candidate, snippet)
            applicability = (
                _tri_to_num(version_match) * 0.35
                + _tri_to_num(cpe_match) * 0.25
                + _tri_to_num(platform_match) * 0.15
                + _tri_to_num(auth_match) * 0.10
                + _tri_to_num(network_match) * 0.15
            )
            readiness = 1.0 if procedure_ready else 0.35
            score = (
                applicability * 0.40
                + trust_score * 0.25
                + readiness * 0.20
                + min(max(candidate.raw_confidence, 0.0), 1.0) * 0.10
                - (estimated_cost / 5.0) * 0.05
            )
            hard_mismatch = "no" in {version_match, cpe_match, platform_match, network_match}
            reasons = [
                f"version={version_match}",
                f"cpe={cpe_match}",
                f"platform={platform_match}",
                f"auth={auth_match}",
                f"network={network_match}",
                f"procedure_ready={procedure_ready}",
            ]
            if snippet_assumption_match != "unknown":
                reasons.append(f"snippet_platform={snippet_assumption_match}")
            if foreign_ip_signal == "warning":
                reasons.append(f"foreign_ip_hygiene=warning:{','.join(foreign_ips)}")
            elif foreign_ip_signal == "no_good_commands":
                reasons.append(f"foreign_ip_hygiene=all_foreign:{','.join(foreign_ips)}")
            if hard_mismatch:
                verdict = "rejected"
            elif not _product_identity_resolved(fp, record):
                # v75 tri-state: unresolved product identity -> unknown.
                verdict = "unknown"
                reasons.append("product_identity_unresolved")
            elif score >= 0.72:
                verdict = "confirmed"
            elif score >= 0.50:
                verdict = "plausible"
            else:
                verdict = "rejected"
            if verdict == "confirmed" and version_match != "yes":
                verdict = "plausible"
                reasons.append("version_confirmation_required")
            assessments.append(ApplicabilityAssessment(
                cve_id=record.cve_id,
                candidate_id=candidate.candidate_id,
                version_match=version_match,
                cpe_match=cpe_match,
                platform_match=platform_match,
                auth_match=auth_match,
                network_match=network_match,
                procedure_ready=procedure_ready,
                trust_score=round(trust_score, 3),
                estimated_cost=estimated_cost,
                score=round(max(score, 0.0), 3),
                verdict=verdict,
                reasons=reasons,
            ))
    return assessments


def build_shortlist(
    fingerprints: list[ProductFingerprint],
    records: list[AuthoritativeRecord],
    candidates: list[PocCandidate],
    snippets: list[ProcedureSnippet],
    assessments: list[ApplicabilityAssessment],
    top_cves: int = 5,
) -> list[dict[str, Any]]:
    {(fp.target_ip, fp.port): fp for fp in fingerprints}
    record_map = {record.cve_id: record for record in records}
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    snippet_map = {snippet.candidate_id: snippet for snippet in snippets}
    shortlist: list[dict[str, Any]] = []

    assessments = sorted(
        assessments,
        key=lambda item: (item.verdict == "confirmed", item.score, item.trust_score),
        reverse=True,
    )
    used_cves: set[str] = set()
    for assessment in assessments:
        record = record_map.get(assessment.cve_id)
        candidate = candidate_map.get(assessment.candidate_id)
        snippet = snippet_map.get(assessment.candidate_id)
        if record is None or candidate is None:
            continue
        text = (record.title + " " + record.description).lower()
        fp = next(
            (
                item for item in fingerprints
                if item.product.lower() in text or item.raw_service.lower() in text
            ),
            fingerprints[0] if fingerprints else None,
        )
        if fp is None:
            continue
        if assessment.cve_id in used_cves:
            continue
        used_cves.add(assessment.cve_id)
        shortlist.append({
            "cve_id": assessment.cve_id,
            "candidate_id": assessment.candidate_id,
            "source": candidate.source,
            "title": record.title,
            "score": assessment.score,
            "verdict": assessment.verdict,
            "trust_score": assessment.trust_score,
            "estimated_cost": assessment.estimated_cost,
            "service": fp.raw_service,
            "vendor": fp.vendor,
            "product": fp.product,
            "version": fp.version,
            "port": fp.port,
            "target_ip": fp.target_ip,
            "path": candidate.path,
            "locator": candidate.locator,
            "references": record.references[:5],
            "commands": snippet.commands[:5] if snippet else [],
            "dependencies": snippet.dependencies[:5] if snippet else [],
            "placeholders": snippet.placeholders[:8] if snippet else [],
            "required_placeholders": snippet.required_placeholders[:8] if snippet else [],
            "working_directory": snippet.working_directory if snippet else "",
            "setup_commands": snippet.setup_commands[:5] if snippet else [],
            "verify_commands": snippet.verify_commands[:3] if snippet else [],
            "success_indicators": snippet.success_indicators[:8] if snippet else [],
            "failure_indicators": snippet.failure_indicators[:8] if snippet else [],
            "reasons": assessment.reasons,
        })
        if len({item["cve_id"] for item in shortlist}) >= top_cves:
            break
    return shortlist


# ── Evidence-tiered hypotheses (paper §3.2, Algorithm 1 steps 7–16) ───────────
#
# Retrieval is product-keyed: ``App(c, M_W)`` is a predicate tri-state —
# ``confirmed`` requires evidence for every mandatory predicate, ``plausible``
# requires support for every predicate but insufficient evidence to confirm
# (a candidate whose version evidence is ``unknown`` stays here, never
# promoted, never discarded), and ``unresolved`` means a mandatory non-product
# predicate has no current support. Contradicted candidates are discarded and
# never receive a verdict (``contradicted()`` must be checked first).
#
# Missing prerequisites ``G(c) = Prereq(c) \ Satisfied(M_W)`` stay attached to
# the candidate and contribute the bounded penalty ``min(α|G|, P_cap)`` with
# ``α = 0.15`` and ``P_cap = 0.45``; candidates are ordered by applicability
# (``confirmed ≻ plausible ≻ unresolved``), then smaller ``|G(c)|``, then a
# stable identifier.

#: paper constants (§3.2), verbatim.
PENALTY_ALPHA = 0.15
PENALTY_CAP = 0.45

#: applicability ordering, worst..best (higher = better); negated in the key so
#: a min-heap pops ``confirmed ≻ plausible ≻ unresolved`` first.
APPLICABILITY_RANK: dict[str, int] = {"unresolved": 0, "plausible": 1, "confirmed": 2}

_SUPPORTED = "supported"
_UNSUPPORTED = "unsupported"
_CONTRADICTED = "contradicted"
_ABSENT = "absent"

_VERSION_EVIDENCE_CLASSES = ("explicit", "inferred")


def _candidate_id(candidate: Any) -> str:
    return str(getattr(candidate, "candidate_id", "") or "")


def _candidate_product(candidate: Any) -> str:
    constraint = getattr(candidate, "constraint", None)
    product = getattr(candidate, "product", "") or getattr(constraint, "product", "")
    return str(product or "").strip()


def _candidate_vendor(candidate: Any) -> str:
    constraint = getattr(candidate, "constraint", None)
    vendor = getattr(candidate, "vendor", "") or getattr(constraint, "vendor", "")
    return str(vendor or "").strip()


def _binding_target(candidate: Any) -> tuple[str, int]:
    bindings = getattr(candidate, "bindings", None) or {}
    env = bindings.get("env", bindings) if isinstance(bindings, dict) else {}
    host = str((env or {}).get("TARGET_IP") or (env or {}).get("HOST") or "").strip()
    raw_port = (env or {}).get("RPORT") or (env or {}).get("PORT")
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = 0
    return host, port


def _candidate_target(candidate: Any, world_state: WorldState) -> tuple[str, ServiceInfo | None]:
    """Locate the ``M_W`` service a candidate is keyed against.

    Explicit target bindings win; otherwise the first service whose identity
    matches the candidate product; otherwise a lone service is unambiguous.
    """
    host, port = _binding_target(candidate)
    if host and port:
        service = world_state.service(host, port)
        if service is not None:
            return host, service
    services = [(ip, svc) for ip, entry in world_state.hosts.items() for svc in entry.services]
    product = _candidate_product(candidate).lower()
    if product:
        for ip, svc in services:
            haystack = f"{svc.name} {svc.cpe} {svc.banner}".lower()
            if product in haystack:
                return ip, svc
    if len(services) == 1:
        return services[0]
    return "", None


def _parse_version_predicate(predicate: str) -> tuple[str, str]:
    text = str(predicate or "").strip()
    for operator in (">=", "<=", "==", ">", "<"):
        if text.startswith(operator):
            return operator, text[len(operator):].strip()
    return "==", text


def _version_bounds(candidate: Any) -> dict[str, Any] | None:
    """Return the version predicate as bounds, or ``None`` when version-independent."""
    extra = getattr(candidate, "extra", None) or {}
    predicate = str(getattr(candidate, "version_predicate", "") or extra.get("version_predicate") or "").strip()
    if predicate:
        operator, value = _parse_version_predicate(predicate)
        if not value:
            return None
        if operator == ">=":
            return {"lo": value, "lo_inc": True}
        if operator == ">":
            return {"lo": value, "lo_inc": False}
        if operator == "<=":
            return {"hi": value, "hi_inc": True}
        if operator == "<":
            return {"hi": value, "hi_inc": False}
        return {"exact": value}
    constraint = getattr(candidate, "constraint", None)
    if constraint is None or getattr(constraint, "is_unknown_version", False):
        return None
    lo = str(getattr(constraint, "version_start", "") or "").strip()
    hi = str(getattr(constraint, "version_end", "") or "").strip()
    if not lo and not hi:
        return None
    return {
        "lo": lo,
        "hi": hi,
        "lo_inc": bool(getattr(constraint, "version_start_inclusive", True)),
        "hi_inc": bool(getattr(constraint, "version_end_inclusive", True)),
    }


def _version_in_bounds(current: version.Version, bounds: dict[str, Any]) -> bool:
    exact = str(bounds.get("exact") or "").strip()
    if exact:
        return current == version.parse(exact)
    lo = str(bounds.get("lo") or "").strip()
    hi = str(bounds.get("hi") or "").strip()
    if lo:
        bound = version.parse(lo)
        if current < bound or (current == bound and not bounds.get("lo_inc", True)):
            return False
    if hi:
        bound = version.parse(hi)
        if current > bound or (current == bound and not bounds.get("hi_inc", True)):
            return False
    return True


def _product_state(candidate: Any, service: ServiceInfo | None) -> str:
    product = _candidate_product(candidate).lower()
    if not product:
        return _UNSUPPORTED
    if service is None:
        return _UNSUPPORTED
    haystack = f"{service.name} {service.cpe} {service.banner}".lower().strip()
    if not haystack:
        return _UNSUPPORTED
    if product in haystack:
        return _SUPPORTED
    # An unrecognized product identity is *unsupported*, not contradicted:
    # retrieval is product-keyed and routes the candidate to IdentifyProduct.
    return _UNSUPPORTED


def _version_state(candidate: Any, service: ServiceInfo | None) -> str:
    bounds = _version_bounds(candidate)
    if bounds is None:
        return _ABSENT
    if service is None:
        return _UNSUPPORTED
    evidence_class = str(getattr(service, "version_evidence_class", "unknown") or "unknown")
    observed = str(getattr(service, "version", "") or "").strip()
    if evidence_class not in _VERSION_EVIDENCE_CLASSES or not observed:
        return _UNSUPPORTED
    try:
        current = version.parse(observed)
        return _SUPPORTED if _version_in_bounds(current, bounds) else _CONTRADICTED
    except Exception:
        return _UNSUPPORTED


def _service_platform_hints(
    host: str, service: ServiceInfo | None, world_state: WorldState
) -> list[str]:
    if service is None:
        return []
    key = f"{host}:{service.port}"
    for kind, node_key, attrs, _ in world_state.nodes("service"):
        if kind == "service" and node_key == key:
            hints = attrs.get("platform_hints") or attrs.get("platform") or []
            return [str(item) for item in hints] if isinstance(hints, (list, tuple)) else [str(hints)]
    return []


def _platform_state(candidate: Any, service: ServiceInfo | None, host: str, world_state: WorldState) -> str:
    platform = str(getattr(candidate, "platform", "") or "").strip()
    if not platform:
        return _ABSENT
    hints = _service_platform_hints(host, service, world_state)
    if not hints:
        return _ABSENT
    return _SUPPORTED if platform.lower() in {hint.lower() for hint in hints} else _CONTRADICTED


def _network_state(candidate: Any, service: ServiceInfo | None) -> str:
    prereqs = list(getattr(candidate, "network_prereqs", ()) or ())
    if not prereqs:
        return _ABSENT
    if service is None:
        return _UNSUPPORTED
    accessibility = str(getattr(service, "accessibility", "") or "")
    if accessibility == "open":
        return _SUPPORTED
    if accessibility == "closed":
        return _CONTRADICTED
    return _UNSUPPORTED


def _auth_state(candidate: Any, world_state: WorldState) -> str:
    required = str(getattr(candidate, "auth_required", "unknown") or "unknown").lower()
    if required != "yes":
        return _ABSENT
    return _SUPPORTED if (world_state.credentials or world_state.sessions) else _UNSUPPORTED


#: ``<port>:<path>`` — a prerequisite bound to one service's endpoint.
_ENDPOINT_PORT_QUALIFIED = re.compile(r"^(?P<port>[0-9]+):(?P<path>.*)$")
#: ``dispatched:<port>:<path>`` / ``dispatched:<path>`` — a prerequisite that
#: only a *dispatched procedure* satisfies, never a discovery probe of the same
#: path: it expresses "this step runs after the step that reaches the endpoint".
_ENDPOINT_DISPATCHED = re.compile(r"^dispatched:(?:(?P<port>[0-9]+):)?(?P<path>.*)$")


def _endpoint_state(candidate: Any, world_state: WorldState) -> str:
    """``endpoint`` predicate over the endpoint nodes of ``M_W``.

    An endpoint node is keyed ``host:port<path>``, so a prerequisite may be
    written bare (``/robots.txt``: any service that exposes the path satisfies it)
    or port-qualified (``9090/robots.txt`` as ``<port>:<path>``: only that
    service's endpoint does). A port-qualified prerequisite is what keeps a
    *proof* step behind the *exploit* step that reaches the endpoint: a
    discovery probe of the same path on another port must not satisfy it.
    """
    prereqs = [str(item) for item in getattr(candidate, "endpoint_prereqs", ()) or () if str(item).strip()]
    if not prereqs:
        return _ABSENT
    nodes = [
        (str(attrs.get("path") or node_key), str(attrs.get("port")),
         bool(str(attrs.get("reached_by") or "")))
        for kind, node_key, attrs, _ in world_state.nodes("endpoint")
        if kind == "endpoint"
    ]
    declared = {path for path, _, _ in nodes}
    qualified = {(port, path) for path, port, _ in nodes}
    reached = {path for path, _, by_dispatch in nodes if by_dispatch}
    reached_qualified = {(port, path) for path, port, by_dispatch in nodes if by_dispatch}
    for item in prereqs:
        dispatched = _ENDPOINT_DISPATCHED.match(item)
        if dispatched:
            port, path = dispatched.group("port"), dispatched.group("path")
            if port is not None:
                if (port, path) not in reached_qualified:
                    return _UNSUPPORTED
            elif path not in reached:
                return _UNSUPPORTED
            continue
        match = _ENDPOINT_PORT_QUALIFIED.match(item)
        if match:
            if (match.group("port"), match.group("path")) not in qualified:
                return _UNSUPPORTED
        elif item not in declared:
            return _UNSUPPORTED
    return _SUPPORTED


def _predicate_states(candidate: Any, world_state: WorldState) -> dict[str, str]:
    host, service = _candidate_target(candidate, world_state)
    return {
        "product": _product_state(candidate, service),
        "version": _version_state(candidate, service),
        "platform": _platform_state(candidate, service, host, world_state),
        "network": _network_state(candidate, service),
        "auth": _auth_state(candidate, world_state),
        "endpoint": _endpoint_state(candidate, world_state),
    }


def contradicted(candidate: Any, world_state: WorldState) -> bool:
    """``True`` when any mandatory predicate is explicitly contradicted.

    Contradicted candidates are discarded rather than ranked; they must never
    reach :func:`applicability`.
    """
    return any(state == _CONTRADICTED for state in _predicate_states(candidate, world_state).values())


def applicability(candidate: Any, world_state: WorldState) -> str:
    """``App(c, M_W) ∈ {confirmed, plausible, unresolved}`` (paper §3.2).

    ``confirmed`` when every mandatory predicate is supported; ``plausible``
    when the product is supported and only the version predicate lacks support
    (unknown version evidence never promotes and never discards); ``unresolved``
    when the product identity or any mandatory non-product predicate is
    unsupported. Raises ``ValueError`` for contradicted candidates, which are
    discarded and never carry a verdict.
    """
    states = _predicate_states(candidate, world_state)
    if any(state == _CONTRADICTED for state in states.values()):
        raise ValueError(
            f"contradicted candidate {_candidate_id(candidate)!r} has no applicability verdict"
        )
    if states["product"] != _SUPPORTED:
        return "unresolved"
    if any(
        state != _SUPPORTED and state != _ABSENT
        for key, state in states.items()
        if key not in ("product", "version")
    ):
        return "unresolved"
    if states["version"] == _UNSUPPORTED:
        # unknown/inferred-insufficient version evidence: plausible, never promoted.
        return "plausible"
    return "confirmed"


def _prerequisite_label(key: str, candidate: Any, service: ServiceInfo | None) -> str:
    if key == "product":
        return f"product:{_candidate_product(candidate) or 'unknown'}"
    if key == "version":
        bounds = _version_bounds(candidate) or {}
        return f"version:{_candidate_product(candidate) or 'unknown'} {bounds}"
    if key == "platform":
        return f"platform:{getattr(candidate, 'platform', '')}"
    if key == "network":
        host, _ = _binding_target(candidate)
        port = getattr(service, "port", "") if service is not None else ""
        return f"network:{host}:{port}"
    if key == "auth":
        return "auth:session"
    return f"endpoint:{','.join(str(item) for item in getattr(candidate, 'endpoint_prereqs', ()) or ())}"


def prerequisites_missing(candidate: Any, world_state: WorldState) -> tuple[str, ...]:
    """``G(c) = Prereq(c) \\ Satisfied(M_W)`` — labels of unsatisfied prerequisites.

    Labels are stable and feed :func:`src.design.subgoals.subgoals_from_preconditions`
    (``version`` → ``ObtainVersion``, ``product`` → ``IdentifyProduct``, …).
    """
    states = _predicate_states(candidate, world_state)
    _, service = _candidate_target(candidate, world_state)
    return tuple(
        _prerequisite_label(key, candidate, service)
        for key, state in states.items()
        if state == _UNSUPPORTED
    )


def penalty(g_missing: Any) -> float:
    """Bounded penalty ``min(α|G|, P_cap)`` with ``α = 0.15``, ``P_cap = 0.45``."""
    return min(PENALTY_ALPHA * len(tuple(g_missing or ())), PENALTY_CAP)


def priority_key(candidate: Any, verdict: str, g_missing: Any) -> tuple[int, int, str]:
    """``(-rank, |G(c)|, candidate_id)`` — smaller sorts first in a min-heap."""
    rank = APPLICABILITY_RANK.get(str(verdict), -1)
    return (-rank, len(tuple(g_missing or ())), _candidate_id(candidate))
