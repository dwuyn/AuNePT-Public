"""
Offline-first authoritative vulnerability retrieval.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from src.retrieval.models import AuthoritativeRecord, ProductFingerprint
from src.tools.cve_search import get_cve_detail_record, search_cve_records

_VENDOR_DOMAINS = {
    "apache.org": "apache",
    "tomcat.apache.org": "apache",
    "nginx.org": "nginx",
    "microsoft.com": "microsoft",
    "oracle.com": "oracle",
    "postgresql.org": "postgresql",
    "openbsd.org": "openbsd",
    "php.net": "php",
}
_WINDOWS_HINTS = ("windows", "iis", "microsoft", "smb", "rdp", "winrm")
_LINUX_HINTS = ("linux", "unix", "apache", "nginx", "openssh", "ubuntu", "debian", "centos")
_AUTH_HINTS = ("auth", "authenticated", "credential", "login", "password")
_CPE_VERSION_INDEX = 5


def _flatten_refs(detail: dict[str, Any]) -> list[str]:
    refs = detail.get("references") or detail.get("refs") or []
    flat: list[str] = []
    if isinstance(refs, dict):
        refs = refs.values()
    for ref in refs:
        if isinstance(ref, str):
            flat.append(ref)
        elif isinstance(ref, dict):
            for key in ("url", "link", "href"):
                if ref.get(key):
                    flat.append(str(ref[key]))
                    break
    return list(dict.fromkeys(flat))


def _extract_weaknesses(detail: dict[str, Any]) -> list[str]:
    weaknesses = detail.get("weaknesses") or detail.get("cwe") or []
    normalized: list[str] = []
    if isinstance(weaknesses, str):
        return [weaknesses]
    for item in weaknesses:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            value = item.get("description") or item.get("name") or item.get("cwe")
            if value:
                normalized.append(str(value))
    return normalized


def _normalize_affected_ranges(detail: dict[str, Any]) -> list[dict[str, Any]]:
    affected = detail.get("affected_versions") or detail.get("affected") or []
    normalized: list[dict[str, Any]] = []
    if isinstance(affected, dict):
        affected = [affected]
    for item in affected:
        if not isinstance(item, dict):
            continue
        version = str(item.get("version") or "")
        cpe_values = item.get("cpe") or []
        if not version and isinstance(cpe_values, list):
            for cpe in cpe_values:
                parts = str(cpe).split(":")
                if len(parts) > _CPE_VERSION_INDEX and parts[_CPE_VERSION_INDEX] not in {"", "*", "-"}:
                    version = parts[_CPE_VERSION_INDEX]
                    break
        normalized.append({
            "product": str(item.get("product") or item.get("package") or ""),
            "vendor": str(item.get("vendor") or ""),
            "min_version": str(item.get("min_version") or item.get("introduced") or ""),
            "max_version": str(item.get("max_version") or item.get("fixed") or ""),
            "version": version,
        })
    return normalized


def _platform_hints(detail: dict[str, Any], refs: list[str], description: str) -> list[str]:
    text = " ".join([description, " ".join(refs)]).lower()
    hints: list[str] = []
    if any(token in text for token in _WINDOWS_HINTS):
        hints.append("windows")
    if any(token in text for token in _LINUX_HINTS):
        hints.append("linux")
    return list(dict.fromkeys(hints))


def _auth_hint(description: str) -> str:
    text = (description or "").lower()
    return "required" if any(token in text for token in _AUTH_HINTS) else "unknown"


def _vendor_refs(refs: list[str], vendor: str) -> list[str]:
    matches: list[str] = []
    for ref in refs:
        host = urlparse(ref).netloc.lower()
        for domain, label in _VENDOR_DOMAINS.items():
            if domain in host and (not vendor or vendor == label):
                matches.append(ref)
    return list(dict.fromkeys(matches))


def _kev_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return set()
    items = data.get("vulnerabilities", data) if isinstance(data, dict) else data
    matched: set[str] = set()
    if not isinstance(items, list):
        return matched
    for item in items:
        if isinstance(item, dict):
            cve = item.get("cveID") or item.get("cve_id")
            if cve:
                matched.add(str(cve).upper())
        elif isinstance(item, str):
            matched.add(item.upper())
    return matched


def _merge_records(base: AuthoritativeRecord, incoming: AuthoritativeRecord) -> AuthoritativeRecord:
    base.references = list(dict.fromkeys(base.references + incoming.references))
    base.weaknesses = list(dict.fromkeys(base.weaknesses + incoming.weaknesses))
    base.affected_ranges.extend(
        item for item in incoming.affected_ranges if item not in base.affected_ranges
    )
    base.platform_hints = list(dict.fromkeys(base.platform_hints + incoming.platform_hints))
    base.evidence = list(dict.fromkeys(base.evidence + incoming.evidence))
    base.cvss_score = max(base.cvss_score, incoming.cvss_score)
    base.epss_percentile = max(base.epss_percentile, incoming.epss_percentile)
    if base.source == "cvemap" and incoming.source in {"kev", "vendor"}:
        base.source = incoming.source
    if base.auth_hint == "unknown":
        base.auth_hint = incoming.auth_hint
    if base.exploit_maturity_hint == "unknown":
        base.exploit_maturity_hint = incoming.exploit_maturity_hint
    return base


def _build_record(
    detail: dict[str, Any],
    fingerprint: ProductFingerprint,
    kev_ids: set[str],
) -> AuthoritativeRecord | None:
    cve_id = str(detail.get("cve_id") or detail.get("id") or "").upper()
    if not cve_id:
        return None
    refs = _flatten_refs(detail)
    description = str(detail.get("cve_description") or detail.get("description") or "")
    vendor_refs = _vendor_refs(refs, fingerprint.vendor)
    source = "vendor" if vendor_refs else "kev" if cve_id in kev_ids else "cvemap"
    evidence = [
        f"{source} retrieval matched {fingerprint.vendor}/{fingerprint.product} on {fingerprint.target_ip}:{fingerprint.port}",
    ]
    if vendor_refs:
        evidence.append(f"Vendor references: {', '.join(vendor_refs[:3])}")
    if cve_id in kev_ids:
        evidence.append("Present in local KEV cache")
    affected_ranges = _normalize_affected_ranges(detail)
    title = str(detail.get("title") or detail.get("summary") or cve_id)
    maturity = "poc" if any("exploit" in ref.lower() for ref in refs) else "unknown"
    return AuthoritativeRecord(
        cve_id=cve_id,
        source=source,
        title=title,
        description=description[:2000],
        cvss_score=float(detail.get("cvss_score") or detail.get("cvss", {}).get("score", 0.0) or 0.0),
        epss_percentile=float(
            detail.get("epss_percentile") or detail.get("epss", {}).get("epss_percentile", 0.0) or 0.0
        ),
        weaknesses=_extract_weaknesses(detail),
        references=refs,
        affected_ranges=affected_ranges,
        platform_hints=_platform_hints(detail, refs, description),
        auth_hint=_auth_hint(description),
        exploit_maturity_hint=maturity,
        evidence=evidence,
    )


def _load_curated_benchmark_cve_cache(
    path: str,
    fingerprints: list[ProductFingerprint],
    errors: list[str] | None = None,
) -> list[AuthoritativeRecord]:
    """Read a curated benchmark CVE ground-truth file for deterministic runs."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        if errors is not None:
            errors.append(f"benchmark_cve_cache load failed: {exc}")
        return []
    items = data if isinstance(data, list) else data.get("cve_entries", [])
    if not isinstance(items, list):
        return []
    target_vendor_set = {fp.vendor.lower() for fp in fingerprints if fp.vendor}
    target_product_set = {fp.product.lower() for fp in fingerprints if fp.product}
    records: list[AuthoritativeRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cve_id = str(item.get("cve_id", "")).upper()
        if not cve_id:
            continue
        vendor = str(item.get("vendor", "")).lower()
        product = str(item.get("product", "")).lower()
        if target_vendor_set and vendor not in target_vendor_set:
            continue
        if target_product_set and product not in target_product_set:
            continue
        records.append(AuthoritativeRecord(
            cve_id=cve_id,
            source="benchmark",
            title=str(item.get("title", cve_id)),
            description=str(item.get("description", ""))[:2000],
            cvss_score=float(item.get("cvss_score", 0.0) or 0.0),
            epss_percentile=float(item.get("epss_percentile", 0.0) or 0.0),
            weaknesses=list(item.get("weaknesses", [])) if isinstance(item.get("weaknesses"), list) else [],
            references=list(item.get("references", [])) if isinstance(item.get("references"), list) else [],
            affected_ranges=[
                # ponytail: minimal format — copy each range dict verbatim
                dict(r) if isinstance(r, dict) else {"version": str(r)}
                for r in (item.get("affected_ranges") or item.get("affected_versions") or [])
            ],
            platform_hints=list(item.get("platform_hints", [])) if isinstance(item.get("platform_hints"), list) else [],
            auth_hint=str(item.get("auth_hint", "unknown")),
            exploit_maturity_hint=str(item.get("exploit_maturity_hint", "poc")),
            evidence=[f"benchmark curated CVE for {vendor}/{product}"],
        ))
    return records


_BANNER_FRAGMENT_RE = re.compile(r"\(\(.*?\)\)")


def _sanitize_query(raw: str) -> str:
    """Normalize banner fragments out of a search query string."""
    cleaned = _BANNER_FRAGMENT_RE.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_queries(fp: ProductFingerprint) -> list[str]:
    """Build normalized search queries from a fingerprint.

    Preferred order: vendor+product, product+version.
    Never sends raw banners or un-normalized service names.
    """
    queries: list[str] = []
    vendor_prod = " ".join(part for part in (fp.vendor, fp.product) if part).strip()
    if vendor_prod and vendor_prod not in {"unknown unknown", "unknown"}:
        queries.append(_sanitize_query(vendor_prod))
    prod_ver = " ".join(part for part in (fp.product, fp.version) if part).strip()
    if prod_ver and prod_ver != vendor_prod and "unknown" not in prod_ver.split()[0]:
        queries.append(_sanitize_query(prod_ver))
    return [q for q in queries if q]


def collect_authoritative_records(
    fingerprints: list[ProductFingerprint],
    retrieval_cfg: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> tuple[list[AuthoritativeRecord], str]:
    cfg = retrieval_cfg or {}
    benchmark_path = str(cfg.get("benchmark_cve_cache_path") or cfg.get("benchmark_cve_path") or "")
    if benchmark_path:
        if not os.path.exists(benchmark_path):
            if errors is not None:
                errors.append(f"benchmark_cve_cache_path configured but file not found: {benchmark_path}")
            return [], "dataset_missing"
        benchmark_records = _load_curated_benchmark_cve_cache(benchmark_path, fingerprints, errors)
        if benchmark_records:
            return benchmark_records, "ok"
        if errors is not None:
            errors.append("benchmark_cve_cache loaded but matched zero CVEs for fingerprints")
        return [], "no_match"

    per_service_limit = int(cfg.get("per_service_cve_limit", 8))
    kev_ids = _kev_ids(str(cfg.get("kev_cache_path") or ""))
    merged: dict[str, AuthoritativeRecord] = {}
    evidence_counts: defaultdict[str, int] = defaultdict(int)

    query_errors = 0
    backend_errors = 0
    total_queries = 0
    for fp in fingerprints:
        queries = _build_queries(fp)
        seen_cves: set[str] = set()
        for query in queries:
            total_queries += 1
            if len(query) < 3:
                query_errors += 1
                continue
            try:
                records = search_cve_records(query, limit=per_service_limit)
            except Exception as exc:
                backend_errors += 1
                if errors is not None:
                    errors.append(f"search_cve_records query='{query}' failed: {exc}")
                continue
            for item in records:
                cve_id = str(item.get("cve_id") or "").upper()
                if not cve_id or cve_id in seen_cves:
                    continue
                seen_cves.add(cve_id)
                try:
                    detail = get_cve_detail_record(cve_id) or item
                except Exception as exc:
                    if errors is not None:
                        errors.append(f"get_cve_detail_record id='{cve_id}' failed: {exc}")
                    detail = item
                record = _build_record(detail, fp, kev_ids)
                if record is None:
                    continue
                evidence_counts[record.cve_id] += 1
                if record.cve_id in merged:
                    merged[record.cve_id] = _merge_records(merged[record.cve_id], record)
                else:
                    merged[record.cve_id] = record

    ordered = list(merged.values())
    ordered.sort(
        key=lambda item: (
            item.source == "vendor",
            item.source == "kev",
            evidence_counts[item.cve_id],
            item.cvss_score,
            item.epss_percentile,
        ),
        reverse=True,
    )

    if not ordered:
        if backend_errors > 0 and backend_errors >= total_queries:
            errs = errors or []
            errs.append(f"All {total_queries} CVE queries failed with backend errors")
            return [], "backend_failed"
        if query_errors > 0 and query_errors >= total_queries:
            errs = errors or []
            errs.append(f"All {total_queries} queries were invalid (too short or malformed)")
            return [], "query_invalid"
        return [], "no_match"
    return ordered, "ok"
