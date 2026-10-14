"""
src/pipeline/evidence.py
────────────────────────
Service fingerprinting with strict observation/inference separation.

Defects fixed vs the legacy ``src/retrieval/fingerprint.py``:

  * Protocol strings (``HTTP/1.1``, ``TLSv1.2``), CVE ids, ports, dates, and
    status codes can never become application versions.
  * Generic service aliases (``http proxy``, ``ssh``) can no longer fabricate
    a vendor/product identity; an unknown identity stays unknown.
  * Observed CPE fields and inferred CPE fields are stored separately.
    An inferred CPE never overwrites an observed CPE.
  * Unknown version cannot receive exact-applicability ratings downstream
    (the consumer enforces this via ``Fingerprint.applicability_grade``).
  * Inclusive/exclusive affected-version boundaries are normalised once at
    fingerprint emission so downstream applicability code never has to.

Evidence precedence (highest to lowest):

  1. Explicit scanner CPE or structured product/version evidence.
  2. Recognized product-specific banner or header parser.
  3. A second independent probe confirming the identity.
  4. Generic service name, retained only as a service hypothesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from packaging import version as _packaging_version

from src.pipeline.ledger import EventLedger

# ── Constants ─────────────────────────────────────────────────────────────────

# Strings that are NEVER valid application versions. Any extraction that
# produces one of these must drop the candidate and fall back to "unknown".
_NON_VERSION_TOKENS = frozenset({
    "http", "https", "ssl", "tls", "tcp", "udp", "ssh", "smtp", "ftp",
    "smb", "rdp", "imap", "pop3", "dns", "ldap", "http/1.0", "http/1.1",
    "http/2", "http/2.0", "tlsv1.0", "tlsv1.1", "tlsv1.2", "tlsv1.3",
})

# Regex for a real dotted-number or numeric version token. Used to *accept*
# versions; everything else is rejected by default.
_VERSION_RE = re.compile(
    r"\b(\d+(?:\.\d+){1,3}(?:[\-+]?(?:p|rc|alpha|beta|preview|dev)\d*)?)\b",
    re.IGNORECASE,
)
# Strict CVE/date/status rejection patterns.
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)
_STATUS_CODE_RE = re.compile(r"^\d{3}$")
_DATE_RE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
)
_PORT_RE = re.compile(r"^\d{1,5}$")

# ── Models ────────────────────────────────────────────────────────────────────


@dataclass
class IdentityField:
    raw: str
    parsed: str
    source: str
    timestamp: float
    observed: bool         # observed from probe vs inferred from a rule
    confidence: str        # "high" | "medium" | "low" | "unknown"
    reason: str


@dataclass
class VersionConstraint:
    vendor: str = ""
    product: str = ""
    version_start: str = ""
    version_end: str = ""
    version_start_inclusive: bool = True
    version_end_inclusive: bool = True
    is_unknown_version: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "product": self.product,
            "version_start": self.version_start,
            "version_end": self.version_end,
            "version_start_inclusive": self.version_start_inclusive,
            "version_end_inclusive": self.version_end_inclusive,
            "is_unknown_version": self.is_unknown_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionConstraint":
        return cls(**{k: data.get(k, getattr(cls(), k)) for k in (
            "vendor", "product", "version_start", "version_end",
            "version_start_inclusive", "version_end_inclusive",
            "is_unknown_version",
        )})


_RANGE_CLAUSE_RE = re.compile(
    r"^\s*(?P<op><=|>=|<|>|==|=)?\s*(?P<version>[0-9][0-9A-Za-z.\-+_]*?)\s*$"
)
_RANGE_WILDCARD_RE = re.compile(r"(?:\.[xX*]|\.x)+$")


def _range_clause(text: str) -> tuple[str, str] | None:
    """``(operator, version)`` for one advisory clause, or ``None`` when unusable."""
    match = _RANGE_CLAUSE_RE.match(str(text or ""))
    if match is None:
        return None
    version = _RANGE_WILDCARD_RE.sub("", match.group("version").strip().rstrip("."))
    if not version or not version[0].isdigit():
        return None
    try:
        _packaging_version.parse(version)
    except Exception:  # noqa: BLE001 - an unparseable token is not a version
        return None
    return match.group("op") or "", version


def normalize_advisory_range(
    start: str,
    end: str,
    *,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> tuple[str, str, bool, bool]:
    """Return ``(low, high, low_inclusive, high_inclusive)`` for an advisory range.

    CNA records encode the affected range inconsistently: the *version* field may
    carry an inline operator (``"<= 3.31.4"``) or a whole clause list
    (``">= 4.0.0, < 4.0.5"``), the *lessThan* field is often empty, wildcards
    appear as ``6.2.x``, and some vendors label builds (``"OTP-27.0-rc1"``). Only
    a whitespace-free version token is comparable; a label-like or empty field
    leaves that side unbounded, so a candidate is never constrained by a bound
    the advisory did not actually state.
    """
    low, high = "", ""
    low_inclusive, high_inclusive = True, True
    for text, inclusive, bare_side in ((start, start_inclusive, "low"), (end, end_inclusive, "high")):
        for clause in str(text or "").split(","):
            parsed = _range_clause(clause)
            if parsed is None:
                continue
            operator, version = parsed
            if operator in {"<=", "<"}:
                high, high_inclusive = version, operator == "<="
            elif operator in {">=", ">"}:
                low, low_inclusive = version, operator == ">="
            elif bare_side == "low":
                low, low_inclusive = version, inclusive
            else:
                high, high_inclusive = version, inclusive
    return low, high, low_inclusive, high_inclusive


@dataclass
class Fingerprint:
    target_ip: str
    port: int
    protocol: str

    # Identity fields, each with explicit observation/inference metadata.
    vendor: IdentityField = field(default_factory=lambda: IdentityField("", "", "", 0.0, False, "unknown", ""))
    product: IdentityField = field(default_factory=lambda: IdentityField("", "", "", 0.0, False, "unknown", ""))
    version: IdentityField = field(default_factory=lambda: IdentityField("", "", "", 0.0, False, "unknown", ""))
    observed_cpe: str = ""
    inferred_cpe_candidates: list[str] = field(default_factory=list)
    platform_hints: list[str] = field(default_factory=list)
    auth_hint: str = "unknown"

    evidence: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    def applicability_grade(self) -> str:
        """Return ``"exact"``, ``"partial"``, or ``"unknown"`` for ranking.

        The handoff forbids exact applicability when the version is unknown.
        """
        if self.version.observed and self.version.confidence in {"high", "medium"} \
                and self.version.parsed and not self.version.parsed == "unknown":
            return "exact"
        if self.version.parsed and self.version.parsed != "unknown":
            return "partial"
        return "unknown"

    @property
    def service_key(self) -> str:
        """Stable key for this fingerprint: target_ip:port:protocol:product.

        Used by B2 multi-fingerprint planner to persist the active service
        across graph nodes without serializing the full Fingerprint object.
        """
        product = self.product.parsed or "unknown"
        return f"{self.target_ip}:{self.port}:{self.protocol.lower()}:{product}"

    def cpe_primary(self) -> str:
        return self.observed_cpe or (self.inferred_cpe_candidates[0] if self.inferred_cpe_candidates else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ip": self.target_ip,
            "port": self.port,
            "protocol": self.protocol,
            "vendor": self.vendor.__dict__,
            "product": self.product.__dict__,
            "version": self.version.__dict__,
            "observed_cpe": self.observed_cpe,
            "inferred_cpe_candidates": list(self.inferred_cpe_candidates),
            "platform_hints": list(self.platform_hints),
            "auth_hint": self.auth_hint,
            "evidence": list(self.evidence),
            "evidence_sources": list(self.evidence_sources),
            "applicability_grade": self.applicability_grade(),
            "service_key": self.service_key,
        }


# ── Banner / version parsing ──────────────────────────────────────────────────

# Product-specific banner parsers. Returning ``None`` means this parser is not
# authoritative for the banner; the normalizer then falls back to the alias
# table and finally leaves the identity as "unknown".
_BANNER_PARSERS: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (re.compile(r"(?:server\s*:\s*)?apache[/\s](?P<version>[0-9][0-9.\-+a-z]*)", re.IGNORECASE),
     {"vendor": "apache", "product": "httpd"}),
    (re.compile(r"apache[/\s]?httpd[/\s]?(?P<version>[0-9][0-9.\-+a-z]*)", re.IGNORECASE),
     {"vendor": "apache", "product": "httpd"}),
    (re.compile(r"nginx[/\s](?P<version>[0-9][0-9.\-+a-z]*)", re.IGNORECASE),
     {"vendor": "nginx", "product": "nginx"}),
    (re.compile(r"openssh[_\s](?P<version>[0-9][0-9.p]+)", re.IGNORECASE),
     {"vendor": "openbsd", "product": "openssh"}),
    (re.compile(r"vsftpd\s*(?P<version>[0-9][0-9.\-+]*)", re.IGNORECASE),
     {"vendor": "vsftpd", "product": "vsftpd"}),
    (re.compile(r"proftpd\s*(?P<version>[0-9][0-9.\-+]*)", re.IGNORECASE),
     {"vendor": "proftpd", "product": "proftpd"}),
    (re.compile(r"mysql\s*(?P<version>[0-9][0-9.\-+a-z]*)", re.IGNORECASE),
     {"vendor": "oracle", "product": "mysql"}),
    (re.compile(r"mariadb\s*(?P<version>[0-9][0-9.\-+a-z]*)", re.IGNORECASE),
     {"vendor": "mariadb", "product": "mariadb"}),
    (re.compile(r"postgresql\s*(?P<version>[0-9][0-9.\-+]*)", re.IGNORECASE),
     {"vendor": "postgresql", "product": "postgresql"}),
    (re.compile(r"microsoft-iis[/\s]?(?P<version>[0-9.]+)", re.IGNORECASE),
     {"vendor": "microsoft", "product": "iis"}),
    (re.compile(r"apache-coyote[/\s]?(?P<version>[0-9.]+)", re.IGNORECASE),
     {"vendor": "apache", "product": "tomcat"}),
]


def _identity_field(
    parsed: str,
    *,
    raw: str = "",
    source: str = "",
    observed: bool = False,
    confidence: str = "unknown",
    timestamp: float = 0.0,
    reason: str = "",
) -> IdentityField:
    parsed = (parsed or "").strip().lower()
    if not parsed or parsed == "unknown":
        confidence = "unknown"
    return IdentityField(
        raw=raw or parsed,
        parsed=parsed or "unknown",
        source=source,
        timestamp=timestamp,
        observed=observed,
        confidence=confidence,
        reason=reason,
    )


def _parse_version(raw: str, *, source: str, timestamp: float) -> IdentityField:
    """
    Extract a real application version from ``raw``.

    Returns an ``IdentityField`` with ``parsed == "unknown"`` whenever the
    candidate string is empty, a CVE id, a date, a status code, a port, or
    a known protocol label.
    """
    raw = (raw or "").strip()
    if not raw:
        return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                confidence="unknown", reason="empty version")
    if _CVE_RE.match(raw):
        return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                confidence="unknown", reason="CVE identifier rejected as version")
    if _STATUS_CODE_RE.match(raw):
        return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                confidence="unknown", reason="HTTP status code rejected as version")
    if _DATE_RE.match(raw):
        return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                confidence="unknown", reason="date rejected as version")
    if _PORT_RE.match(raw):
        return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                confidence="unknown", reason="port rejected as version")
    # The first dotted-number token wins if it's a real version.
    match = _VERSION_RE.search(raw)
    if match:
        candidate = match.group(1).lower()
        if candidate.lower() in _NON_VERSION_TOKENS or "." not in candidate:
            return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                                    confidence="unknown", reason="non-version token rejected")
        return _identity_field(candidate, raw=raw, source=source, timestamp=timestamp,
                                observed=True, confidence="high",
                                reason="matched dotted-number version token")
    return _identity_field("unknown", raw=raw, source=source, timestamp=timestamp,
                            confidence="unknown", reason="no valid version token found")


def parse_banner_identity(banner: str, *, source: str, timestamp: float) -> tuple[str, str, IdentityField | None]:
    """Return ``(vendor, product, version_field_or_none)`` for a banner.

    Returns ``("unknown", "unknown", None)`` when no product-specific banner
    parser matches and no alias rule applies.
    """
    if not banner:
        return ("unknown", "unknown", None)
    for pattern, identity in _BANNER_PARSERS:
        m = pattern.search(banner)
        if not m:
            continue
        version = m.groupdict().get("version") or ""
        vfield = _parse_version(version, source=source, timestamp=timestamp) if version else None
        return (identity["vendor"], identity["product"], vfield)
    return ("unknown", "unknown", None)


# ── Service-name aliases ──────────────────────────────────────────────────────

# Aliases keyed by the *raw* service label. Each alias carries a vendor and
# product and a flag indicating whether it is a recognized product (True) or a
# generic transport that must NOT be allowed to invent an identity (False).
_ALIASES: dict[str, tuple[str, str, bool]] = {
    "ssh": ("openbsd", "openssh", True),
    "openssh": ("openbsd", "openssh", True),
    "http": ("", "", False),                # generic — does NOT invent identity
    "https": ("", "", False),               # generic
    "http proxy": ("", "", False),          # generic — explicit non-identity
    "ssl/http": ("", "", False),
    "ftp": ("", "", False),                 # generic — no product inferred
    "smtp": ("", "", False),
    "smb": ("microsoft", "smb", True),
    "rdp": ("microsoft", "rdp", True),
    "iis": ("microsoft", "iis", True),
    "microsoft-iis": ("microsoft", "iis", True),
    "tomcat": ("apache", "tomcat", True),
    "jetty": ("eclipse", "jetty", True),
    "apache": ("apache", "httpd", True),
    "apache httpd": ("apache", "httpd", True),
    "httpd": ("apache", "httpd", True),
    "nginx": ("nginx", "nginx", True),
    "mysql": ("oracle", "mysql", True),
    "mariadb": ("mariadb", "mariadb", True),
    "postgres": ("postgresql", "postgresql", True),
    "postgresql": ("postgresql", "postgresql", True),
}


def normalize_service_label(label: str) -> tuple[str, str, bool]:
    """Return ``(vendor, product, is_known_product)`` for a raw service label.

    Generic labels (http, http proxy, ftp, smtp) intentionally return
    ``("unknown", "unknown", False)`` so they cannot fabricate a vendor/product
    identity.
    """
    key = (label or "").strip().lower()
    if key in _ALIASES:
        vendor, product, known = _ALIASES[key]
        if not known:
            return ("unknown", "unknown", False)
        return (vendor, product, True)
    # Unrecognised label: do not invent an identity by splitting on whitespace.
    return ("unknown", "unknown", False)


_CPE23_RE = re.compile(
    r"^cpe:2\.3:(?P<part>[aoh*-]):(?P<vendor>[^:]*):(?P<product>[^:]*):(?P<version>[^:]*)"
)


def split_cpe(value: str) -> tuple[str, str, str]:
    """Return ``(vendor, product, version)`` from a CPE 2.3 string.

    CPE 2.3 names join words with ``_`` (spec §5.3), while CVE records spell the
    same product with spaces; the two spellings are normalised to the record
    spelling here so a scanner-asserted CPE can key retrieval. Missing,
    ``*`` and ``-`` components return ``""``.
    """
    match = _CPE23_RE.match((value or "").strip().lower())
    if not match:
        return "", "", ""

    def _name(token: str) -> str:
        if token in {"", "*", "-"}:
            return ""
        return " ".join(token.replace("_", " ").split())

    return _name(match.group("vendor")), _name(match.group("product")), _name(match.group("version"))


def _cpe_for(vendor: str, product: str, version: str) -> str:
    if not vendor or not product or "unknown" in (vendor, product):
        return ""
    v = version or "*"
    return f"cpe:2.3:a:{vendor}:{product}:{v}:*:*:*:*:*:*:*"


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class ServiceObservation:
    """A single piece of evidence about a service (probe, banner, header)."""

    target_ip: str
    port: int
    protocol: str = "tcp"
    service_name: str = ""
    banner: str = ""
    version: str = ""
    observed_cpe: str = ""
    source: str = "probe"
    timestamp: float = 0.0


def fingerprint_service(obs: ServiceObservation, *,
                         ledger: EventLedger | None = None,
                         extra_probes: Iterable[ServiceObservation] | None = None,
                         ) -> Fingerprint:
    """Build a ``Fingerprint`` from one or more ``ServiceObservation`` records.

    Evidence precedence is enforced here:

      1. If *obs.observed_cpe* is non-empty, it is stored as ``observed_cpe``
         and overrides any inferred value.
      2. A product-specific banner parser (if it matches) sets the identity
         with ``observed=True`` and high confidence.
      3. If a second probe confirms the identity, confidence is bumped to high.
      4. Otherwise the service-label alias is used, but only when the alias
         is a *recognized product*. Generic labels leave the identity unknown.
    """
    # 1) observed CPE takes priority and is preserved verbatim.
    observed_cpe = (obs.observed_cpe or "").strip()
    cpe_vendor, cpe_product, cpe_version = split_cpe(observed_cpe)

    # 2) Banner parsing for product-specific identity.
    bv, bp, bversion = parse_banner_identity(obs.banner, source=obs.source, timestamp=obs.timestamp)

    # 4) Alias for service label (only if banner parser did not already win).
    sv, sp, alias_known = normalize_service_label(obs.service_name)
    if bv == "unknown" and sv != "unknown":
        bv, sp = sv, sp
        bp = sp

    # 3) An observed CPE is authoritative for the vendor/product it names, so a
    #    scanner-asserted identity reaches retrieval instead of being dropped.
    #    A live banner version still wins; the CPE supplies the version only
    #    when no banner declared one.
    if cpe_vendor or cpe_product:
        bv = cpe_vendor or bv
        bp = cpe_product or bp
    if bversion is None and cpe_version:
        bversion = _parse_version(cpe_version, source=obs.source, timestamp=obs.timestamp)

    # Cross-probe confirmation.
    confirmed_by = ""
    if extra_probes:
        for probe in extra_probes:
            pv, pp, _ = parse_banner_identity(probe.banner, source=probe.source,
                                                timestamp=probe.timestamp)
            if pv == "unknown" or pp == "unknown":
                continue
            if (pv, pp) == (bv, bp) and pv != "unknown":
                confirmed_by = probe.source
                break
            if pv == bv and pp == sp and pv != "unknown":
                confirmed_by = probe.source
                break

    confidence = "high" if confirmed_by else (
        "high" if (bv != "unknown" and bversion is not None and bversion.parsed != "unknown")
        else ("medium" if bv != "unknown" else "unknown")
    )

    vendor_field = _identity_field(bv, raw=obs.service_name, source=obs.source,
                                     timestamp=obs.timestamp, observed=bool(bversion or observed_cpe),
                                     confidence=confidence,
                                     reason=("CPE observed" if observed_cpe else
                                             ("banner parser matched" if bversion else
                                              ("alias recognized" if alias_known else
                                               "no recognized identity; staying unknown"))))
    product_field = _identity_field(bp, raw=obs.service_name, source=obs.source,
                                     timestamp=obs.timestamp, observed=bool(bversion or observed_cpe),
                                     confidence=confidence,
                                     reason="derived from vendor field")
    # Version: prefer banner-derived version; fall back to obs.version.
    if bversion is not None and bversion.parsed != "unknown":
        version_field = bversion
    else:
        version_field = _parse_version(obs.version, source=obs.source, timestamp=obs.timestamp)
    # Promote observed->True when a second probe confirms the identity.
    if confirmed_by and version_field.parsed != "unknown":
        version_field = _identity_field(
            version_field.parsed, raw=version_field.raw, source=version_field.source,
            timestamp=version_field.timestamp, observed=True, confidence="high",
            reason=f"confirmed by independent probe {confirmed_by}",
        )

    # Inferred CPE candidates are NEVER allowed to overwrite an observed CPE.
    inferred: list[str] = []
    if not observed_cpe and bv != "unknown" and bp != "unknown":
        if version_field.parsed != "unknown":
            inferred.append(_cpe_for(bv, bp, version_field.parsed))
        inferred.append(_cpe_for(bv, bp, ""))

    evidence_lines: list[str] = []
    evidence_sources: list[str] = []
    if obs.banner:
        evidence_lines.append(f"banner='{obs.banner}' from {obs.source}")
        evidence_sources.append("banner")
    if obs.service_name:
        evidence_lines.append(f"service_label='{obs.service_name}' from {obs.source}")
        evidence_sources.append("service_label")
    if obs.version:
        evidence_lines.append(f"version_field='{obs.version}' from {obs.source}")
        evidence_sources.append("version_field")
    if observed_cpe:
        evidence_lines.append(f"observed_cpe='{observed_cpe}'")
        evidence_sources.append("scanner_cpe")
    if confirmed_by:
        evidence_lines.append(f"identity confirmed by independent probe '{confirmed_by}'")
        evidence_sources.append("cross_probe")

    fp = Fingerprint(
        target_ip=obs.target_ip, port=obs.port, protocol=obs.protocol or "tcp",
        vendor=vendor_field, product=product_field, version=version_field,
        observed_cpe=observed_cpe, inferred_cpe_candidates=inferred,
        auth_hint="required" if obs.service_name.lower() in {"ssh", "ftp", "rdp"} else "unknown",
        evidence=evidence_lines, evidence_sources=evidence_sources,
    )

    if ledger is not None:
        ledger.record(
            phase="evidence", stage="applicability",
            service=f"{obs.target_ip}:{obs.port}:{obs.service_name or '?'}",
            detail=fp.applicability_grade(),
            payload={"fingerprint": fp.to_dict(), "evidence_sources": evidence_sources},
        )
    return fp


# ── Inclusive/exclusive version-bound normalisation ────────────────────────────


def normalize_version_bounds(
    start: str = "",
    end: str = "",
    *,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> tuple[str, bool, str, bool]:
    """Normalise affected-version bounds so a "fixed" version never becomes an
    inclusive vulnerable maximum.

    Returns ``(start, start_inclusive, end, end_inclusive)``.
    """
    def _clean(v: str) -> str:
        v = (v or "").strip()
        if not v:
            return ""
        # Strip leading "v", strip whitespace.
        if v[:1].lower() == "v":
            v = v[1:]
        return v

    return (
        _clean(start), bool(start_inclusive),
        _clean(end), bool(end_inclusive),
    )


def constraint_matches(constraint: VersionConstraint, fp: Fingerprint) -> str:
    """Return ``"exact"``, ``"partial"``, ``"unknown"``, or ``"mismatch"``.

    The result drives deterministic applicability ranking; consumers must
    hard-reject ``"mismatch"`` and never rank an unknown version as exact.
    """
    if constraint.vendor and fp.vendor.parsed != constraint.vendor.lower():
        return "mismatch"
    if constraint.product and fp.product.parsed != constraint.product.lower():
        return "mismatch"
    version = fp.version.parsed
    # If the constraint says it applies to an unknown-version fingerprint, or
    # the fingerprint has no usable version, we cannot grade applicability as
    # exact — and the constraint cannot be a mismatch on a dimension we cannot
    # observe.
    if constraint.is_unknown_version or not version or version == "unknown":
        return "unknown"
    # If the constraint specifies a version range, perform a simple string
    # comparison on the dotted-number tuple.
    def _parts(v: str) -> tuple[int, ...]:
        out: list[int] = []
        for piece in v.split("."):
            piece = piece.strip()
            digits = ""
            for ch in piece:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            out.append(int(digits) if digits else 0)
        return tuple(out)

    start_match = True
    end_match = True
    if constraint.version_start:
        if constraint.version_start_inclusive:
            start_match = _parts(version) >= _parts(constraint.version_start)
        else:
            start_match = _parts(version) > _parts(constraint.version_start)
    if constraint.version_end:
        if constraint.version_end_inclusive:
            end_match = _parts(version) <= _parts(constraint.version_end)
        else:
            end_match = _parts(version) < _parts(constraint.version_end)
    if not start_match or not end_match:
        return "mismatch"
    return "exact" if version != "unknown" else "unknown"


def cpe_in_scope(fp: Fingerprint, candidate_cpe: str) -> bool:
    """True when the candidate CPE matches the observed or inferred CPE family.

    Generic service identities with no observed CPE return True only for the
    exact same vendor/product (no product-on-vendor co-incidence false
    matches).
    """
    if not candidate_cpe:
        return False
    candidate_cpe = candidate_cpe.lower()
    observed = fp.observed_cpe.lower() if fp.observed_cpe else ""
    if observed and observed == candidate_cpe:
        return True
    for cpe in fp.inferred_cpe_candidates:
        if cpe.lower() == candidate_cpe:
            return True
    # Bare vendor/product (no version) CPE matches any version of the same product.
    if fp.vendor.parsed != "unknown" and fp.product.parsed != "unknown":
        vendor, product = fp.vendor.parsed, fp.product.parsed
        bare_a = f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*:*"
        bare_b = f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*"
        if candidate_cpe in (bare_a, bare_b):
            return True
    return False
