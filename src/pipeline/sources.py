"""
src/pipeline/sources.py
───────────────────────
Independent CVE source adapters.

Adapters implement a single method, ``fetch(product, vendor, version)``,
returning a list of :class:`RawCveRecord` records. Each adapter writes its
*raw* response to the run directory and the normalised record separately.

Run modes:

  * ``live`` — perform real HTTP / filesystem reads; preserve raw responses.
  * ``snapshot`` — read from a fixed snapshot directory only; raise if a
    requested product/version is missing.
  * ``replay`` — read pre-computed normalised records; no HTTP or I/O.

A failed or rate-limited backend never fails the others — each adapter is
isolated, reports a ``BackendStatus`` event into the ledger, and the registry
collapses the failure into a normalisable status without raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping
from urllib import error as urlerror
from urllib import parse, request

from src.pipeline.ledger import EventLedger

# ── Records ───────────────────────────────────────────────────────────────────


@dataclass
class RawCveRecord:
    """A single CVE record as emitted by an adapter, with raw + normalised fields."""

    source: str                       # cve_list_v5 | nvd | vulnx
    cve_id: str
    raw: dict[str, Any]               # original response payload
    raw_hash: str                     # SHA-256 of the raw payload
    retrieved_at: float

    vendor: str = ""
    product: str = ""
    version_start: str = ""
    version_end: str = ""
    version_start_inclusive: bool = True
    version_end_inclusive: bool = True
    cvss_score: float = 0.0
    cvss_vector: str = ""
    cpe_candidates: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    description: str = ""
    published_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "cve_id": self.cve_id,
            "raw": dict(self.raw or {}),
            "raw_hash": self.raw_hash, "retrieved_at": self.retrieved_at,
            "vendor": self.vendor, "product": self.product,
            "version_start": self.version_start, "version_end": self.version_end,
            "version_start_inclusive": self.version_start_inclusive,
            "version_end_inclusive": self.version_end_inclusive,
            "cvss_score": self.cvss_score, "cvss_vector": self.cvss_vector,
            "cpe_candidates": list(self.cpe_candidates),
            "references": list(self.references), "description": self.description,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawCveRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PrioritySignal:
    """KEV / EPSS enrichment signals (priority only, never applicability)."""

    cve_id: str
    in_kev: bool = False
    kev_date_added: float = 0.0
    epss_score: float = 0.0
    epss_percentile: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrioritySignal":
        return cls(
            cve_id=str(data.get("cve_id", "")),
            in_kev=bool(data.get("in_kev", False)),
            kev_date_added=float(data.get("kev_date_added", 0.0) or 0.0),
            epss_score=float(data.get("epss_score", 0.0) or 0.0),
            epss_percentile=float(data.get("epss_percentile", 0.0) or 0.0),
        )


# ── Backend statuses ──────────────────────────────────────────────────────────


class BackendStatus:
    OK = "ok"
    NO_MATCH = "no_match"
    QUERY_INVALID = "query_invalid"
    BACKEND_FAILED = "backend_failed"
    DATASET_MISSING = "dataset_missing"


# ── Adapters ──────────────────────────────────────────────────────────────────


def _hash_payload(payload: bytes | Mapping[str, Any]) -> str:
    if isinstance(payload, bytes):
        blob = payload
    else:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _cve_ids_from(*values: str) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in str(value or "").replace(",", " ").split():
            token = token.strip().upper()
            if token.startswith("CVE-") and token not in out:
                out.append(token)
    return out


def _cpe_parts(cpe: str) -> tuple[str, str, str]:
    parts = (cpe or "").split(":")
    if len(parts) >= 6:
        return parts[3].lower(), parts[4].lower(), parts[5]
    return "", "", ""


def _maybe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class BaseAdapter:
    name = "base"

    def __init__(self, *, mode: str = "live", snapshot_dir: str = "",
                 ledger: EventLedger | None = None, raw_dir: str = "",
                 timeout: int = 20) -> None:
        if mode not in {"live", "snapshot", "replay"}:
            raise ValueError(f"Invalid adapter mode: {mode}")
        self.mode = mode
        self.snapshot_dir = snapshot_dir
        self.ledger = ledger
        self.raw_dir = raw_dir
        self.timeout = timeout

    def _record_event(self, *, status: str, detail: str, payload: Mapping[str, Any] | None = None) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            phase="retrieval", stage="applicability",
            detail=detail, payload={"source": self.name, "status": status, **(payload or {})},
        )

    def _read_snapshot(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        if not self.snapshot_dir:
            self._record_event(status=BackendStatus.DATASET_MISSING,
                                detail=f"{self.name} snapshot_dir unset")
            return []
        path = os.path.join(self.snapshot_dir, "cves.json")
        if not os.path.exists(path):
            try:
                from src.pipeline.source_snapshot import read_indexed_records
                indexed = read_indexed_records(self.snapshot_dir, source=self.name, product=product, vendor=vendor)
                if indexed is not None:
                    out = [RawCveRecord.from_dict(row) for row in indexed if isinstance(row, dict)]
                    if not out:
                        self._record_event(status=BackendStatus.NO_MATCH,
                                            detail=f"{self.name} snapshot had no match",
                                            payload={"product": product, "version": version})
                    else:
                        self._record_event(status=BackendStatus.OK,
                                            detail=f"{self.name} snapshot records",
                                            payload={"count": len(out),
                                                     "cve_ids": [r.cve_id for r in out]})
                    return out
            except Exception:
                pass
            self._record_event(status=BackendStatus.DATASET_MISSING,
                                detail=f"{self.name} snapshot missing")
            return []
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:                              # noqa: BLE001
            self._record_event(status=BackendStatus.BACKEND_FAILED,
                                detail=f"{self.name} snapshot unreadable",
                                payload={"exception": str(exc)[:160]})
            return []
        entries = data.get(self.name, []) if isinstance(data, dict) else []
        out: list[RawCveRecord] = []
        for entry in entries:
            rec = RawCveRecord.from_dict(entry) if isinstance(entry, dict) else None
            if rec is None:
                continue
            # Source-of-truth precedence: snapshot records already encode
            # vendor/product; we filter on demand so products are returned
            # exactly as snapshotted.
            if product and rec.product and rec.product != product.lower():
                continue
            if vendor and rec.vendor and rec.vendor != vendor.lower():
                continue
            out.append(rec)
        if not out:
            self._record_event(status=BackendStatus.NO_MATCH,
                                detail=f"{self.name} snapshot had no match",
                                payload={"product": product, "version": version})
        else:
            self._record_event(status=BackendStatus.OK,
                                detail=f"{self.name} snapshot records",
                                payload={"count": len(out),
                                         "cve_ids": [r.cve_id for r in out]})
        return out

    def _write_raw(self, label: str, raw: bytes) -> str:
        if not self.raw_dir:
            return ""
        os.makedirs(os.path.join(self.raw_dir, self.name), exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in label)[:120]
        path = os.path.join(self.raw_dir, self.name, f"{safe}-{_hash_payload(raw)[:12]}.json")
        with open(path, "wb") as fh:
            fh.write(raw)
        return path

    def _fetch_json(self, url: str, *, label: str) -> tuple[dict[str, Any], str, float]:
        req = request.Request(url, headers={"User-Agent": "PentestAgent/1.0"})
        with request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        self._write_raw(label, raw)
        return json.loads(raw.decode("utf-8")), _hash_payload(raw), time.time()

    def _record_records(self, *, status: str, detail: str,
                        records: list[RawCveRecord], raw_hash: str = "") -> None:
        self._record_event(status=status, detail=detail, payload={
            "count": len(records),
            "raw_hash": raw_hash,
            "fetched_at": time.time(),
            "cve_ids": [r.cve_id for r in records],
        })

    def fetch(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        raise NotImplementedError

    def _live_unavailable(self) -> list[RawCveRecord]:
        # Never raise on a missing live backend; surface a no-match with status.
        self._record_event(status=BackendStatus.BACKEND_FAILED,
                            detail=f"{self.name} live fetch unavailable",
                            payload={"mode": self.mode})
        return []


class CveListV5Adapter(BaseAdapter):
    """Adapter for the canonical CVE Project list (cvelistV5)."""

    name = "cve_list_v5"

    def __init__(self, *, allow_github_fallback: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allow_github_fallback = allow_github_fallback

    def fetch(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        if not product or product == "unknown":
            if self.mode in {"snapshot", "replay"}:
                return self._read_snapshot("" if product == "unknown" else product, vendor, version)
            self._record_event(status=BackendStatus.QUERY_INVALID,
                                detail="cve_list_v5 requires known product")
            return []
        if self.mode in {"snapshot", "replay"}:
            return self._read_snapshot(product, vendor, version)
        records = self._read_snapshot(product, vendor, version) if self.snapshot_dir else []
        if records:
            return records
        records = self._scan_local_cvelist(product, vendor, version)
        if records:
            self._record_records(status=BackendStatus.OK, detail="cve_list_v5 local clone records",
                                 records=records)
            return records
        cve_ids = _cve_ids_from(product, vendor, version)
        if not self.allow_github_fallback or not cve_ids:
            return self._live_unavailable()
        out: list[RawCveRecord] = []
        for cve_id in cve_ids:
            try:
                url = f"https://raw.githubusercontent.com/CVEProject/cvelistV5/main/{_cvelist_relpath(cve_id)}"
                data, raw_hash, fetched = self._fetch_json(url, label=cve_id)
                rec = _normalise_cvelist_record(data, raw_hash=raw_hash, fetched=fetched,
                                                fallback_vendor=vendor, fallback_product=product)
                if rec:
                    out.append(rec)
            except urlerror.HTTPError as exc:
                self._record_event(status=BackendStatus.BACKEND_FAILED,
                                    detail="cve_list_v5 github fallback failed",
                                    payload={"code": exc.code, "cve_id": cve_id})
            except Exception as exc:  # noqa: BLE001
                self._record_event(status=BackendStatus.BACKEND_FAILED,
                                    detail="cve_list_v5 github fallback failed",
                                    payload={"exception": str(exc)[:160], "cve_id": cve_id})
        self._record_records(status=BackendStatus.OK if out else BackendStatus.NO_MATCH,
                             detail="cve_list_v5 github fallback records", records=out)
        return out

    def _scan_local_cvelist(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        root = self.snapshot_dir
        if not root:
            return []
        cves_root = os.path.join(root, "cves")
        if not os.path.isdir(cves_root):
            return []
        product_l = product.lower()
        vendor_l = vendor.lower()
        out: list[RawCveRecord] = []
        for base, _, files in os.walk(cves_root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(base, name)
                try:
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    if product_l.encode() not in raw.lower() and (
                        not vendor_l or vendor_l.encode() not in raw.lower()
                    ):
                        continue
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                rec = _normalise_cvelist_record(data, raw_hash=_hash_payload(raw),
                                                fetched=time.time(),
                                                fallback_vendor=vendor,
                                                fallback_product=product)
                if rec and _record_mentions(rec, product, vendor):
                    out.append(rec)
        if not out:
            self._record_event(status=BackendStatus.NO_MATCH,
                                detail="cve_list_v5 local clone had no match",
                                payload={"product": product, "version": version})
        return out


class NvdAdapter(BaseAdapter):
    """Adapter for the NVD CVE 2.0 API.

    Respects rate limiting; failures are isolated and reported via the ledger.
    """

    name = "nvd"

    def __init__(self, *, cpe_name: str = "", cve_ids: Iterable[str] | None = None,
                 last_mod_start: str = "", last_mod_end: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cpe_name = cpe_name
        self.cve_ids = list(cve_ids or [])
        self.last_mod_start = last_mod_start
        self.last_mod_end = last_mod_end

    def fetch(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        if not product or product == "unknown":
            if self.mode in {"snapshot", "replay"}:
                return self._read_snapshot("" if product == "unknown" else product, vendor, version)
            self._record_event(status=BackendStatus.QUERY_INVALID,
                                detail="nvd requires known product")
            return []
        if self.mode in {"snapshot", "replay"}:
            return self._read_snapshot(product, vendor, version)
        try:
            records: list[RawCveRecord] = []
            raw_hashes: list[str] = []
            cve_ids = self.cve_ids or _cve_ids_from(product, vendor, version)
            if cve_ids:
                for cve_id in cve_ids:
                    data, raw_hash, fetched = self._fetch_json(
                        "https://services.nvd.nist.gov/rest/json/cves/2.0?"
                        + parse.urlencode({"cveId": cve_id}),
                        label=cve_id,
                    )
                    raw_hashes.append(raw_hash)
                    records.extend(_normalise_nvd_response(data, raw_hash=raw_hash,
                                                           fetched=fetched,
                                                           fallback_vendor=vendor,
                                                           fallback_product=product))
            else:
                params = self._params(product, vendor, version)
                data, raw_hash, fetched = self._fetch_json(
                    "https://services.nvd.nist.gov/rest/json/cves/2.0?"
                    + parse.urlencode(params),
                    label=f"{vendor}-{product}-{version}",
                )
                raw_hashes.append(raw_hash)
                records.extend(_normalise_nvd_response(data, raw_hash=raw_hash,
                                                       fetched=fetched,
                                                       fallback_vendor=vendor,
                                                       fallback_product=product))
            records = [r for r in records if _record_mentions(r, product, vendor)]
            self._record_records(status=BackendStatus.OK if records else BackendStatus.NO_MATCH,
                                 detail="nvd live records", records=records,
                                 raw_hash=",".join(raw_hashes))
            return records
        except urlerror.HTTPError as exc:
            self._record_event(status=BackendStatus.BACKEND_FAILED,
                                detail="nvd live fetch failed",
                                payload={"code": exc.code,
                                         "rate_limited": exc.code == 429})
            return []
        except Exception as exc:  # noqa: BLE001
            self._record_event(status=BackendStatus.BACKEND_FAILED,
                                detail="nvd live fetch failed",
                                payload={"exception": str(exc)[:160]})
            return []

    def _params(self, product: str, vendor: str, version: str) -> dict[str, str]:
        params: dict[str, str] = {}
        cpe = self.cpe_name or _nvd_cpe(vendor, product, version)
        if cpe:
            params["cpeName"] = cpe
        else:
            params["keywordSearch"] = " ".join(p for p in (vendor, product, version) if p)
        if self.last_mod_start:
            params["lastModStartDate"] = self.last_mod_start
        if self.last_mod_end:
            params["lastModEndDate"] = self.last_mod_end
        return params


class VulnxAdapter(BaseAdapter):
    """Optional ``vulnx`` aggregator adapter (never the sole live source)."""

    name = "vulnx"

    def __init__(self, *, base_url: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url

    def fetch(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        if not product or product == "unknown":
            if self.mode in {"snapshot", "replay"}:
                return self._read_snapshot("" if product == "unknown" else product, vendor, version)
            self._record_event(status=BackendStatus.QUERY_INVALID,
                                detail="vulnx requires known product")
            return []
        if self.mode in {"snapshot", "replay"}:
            return self._read_snapshot(product, vendor, version)
        if not self.base_url:
            return self._live_unavailable()
        try:
            url = self.base_url.rstrip("/") + "/?" + parse.urlencode({
                "product": product, "vendor": vendor, "version": version,
            })
            data, raw_hash, fetched = self._fetch_json(url, label=f"{vendor}-{product}-{version}")
            records = _normalise_vulnx_response(data, raw_hash=raw_hash, fetched=fetched,
                                                fallback_vendor=vendor,
                                                fallback_product=product)
            self._record_records(status=BackendStatus.OK if records else BackendStatus.NO_MATCH,
                                 detail="vulnx live records", records=records,
                                 raw_hash=raw_hash)
            return records
        except urlerror.HTTPError as exc:
            self._record_event(status=BackendStatus.BACKEND_FAILED,
                                detail="vulnx live fetch failed",
                                payload={"code": exc.code,
                                         "rate_limited": exc.code == 429})
            return []
        except Exception as exc:  # noqa: BLE001
            self._record_event(status=BackendStatus.BACKEND_FAILED,
                                detail="vulnx live fetch failed",
                                payload={"exception": str(exc)[:160]})
            return []


# ── Normalisers ──────────────────────────────────────────────────────────────


def _nvd_cpe(vendor: str, product: str, version: str) -> str:
    if not vendor or not product or "unknown" in {vendor, product}:
        return ""
    v = version if version and version != "unknown" else "*"
    return f"cpe:2.3:a:{vendor}:{product}:{v}:*:*:*:*:*:*:*"


def _normalise_nvd_response(data: Mapping[str, Any], *, raw_hash: str,
                            fetched: float, fallback_vendor: str,
                            fallback_product: str) -> list[RawCveRecord]:
    out: list[RawCveRecord] = []
    for item in data.get("vulnerabilities", []) if isinstance(data, Mapping) else []:
        cve = item.get("cve", {}) if isinstance(item, Mapping) else {}
        cve_id = str(cve.get("id", "") or "").upper()
        if not cve_id:
            continue
        descriptions = cve.get("descriptions", []) or []
        description = ""
        for entry in descriptions:
            if entry.get("lang") == "en":
                description = entry.get("value", "")
                break
        references = [
            ref.get("url", "") for ref in (cve.get("references", {}) or {}).get("referenceData", [])
            if ref.get("url")
        ]
        cpes: list[str] = []
        version_start = version_end = ""
        start_inc = end_inc = True
        for cfg in cve.get("configurations", []) or []:
            for node in cfg.get("nodes", []) or []:
                for match in node.get("cpeMatch", []) or []:
                    criteria = match.get("criteria", "")
                    if criteria:
                        cpes.append(criteria)
                    version_start = match.get("versionStartIncluding") or match.get("versionStartExcluding") or version_start
                    version_end = match.get("versionEndIncluding") or match.get("versionEndExcluding") or version_end
                    start_inc = "versionStartExcluding" not in match
                    end_inc = "versionEndExcluding" not in match
        vendor, product, _ = _cpe_parts(cpes[0] if cpes else "")
        cvss_score, cvss_vector = _nvd_cvss(cve.get("metrics", {}) or {})
        out.append(RawCveRecord(
            source="nvd", cve_id=cve_id, raw=dict(item), raw_hash=raw_hash,
            retrieved_at=fetched, vendor=vendor or fallback_vendor.lower(),
            product=product or fallback_product.lower(),
            version_start=version_start, version_end=version_end,
            version_start_inclusive=start_inc, version_end_inclusive=end_inc,
            cvss_score=cvss_score, cvss_vector=cvss_vector,
            cpe_candidates=list(dict.fromkeys(cpes)), references=references,
            description=description,
            published_at=_parse_time(cve.get("published", "")),
        ))
    return out


def _nvd_cvss(metrics: Mapping[str, Any]) -> tuple[float, str]:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, []) or []
        if not entries:
            continue
        cvss = entries[0].get("cvssData", {}) or {}
        return _maybe_float(cvss.get("baseScore")), str(cvss.get("vectorString", "") or "")
    return 0.0, ""


def _cvelist_relpath(cve_id: str) -> str:
    parts = cve_id.upper().split("-")
    year = parts[1]
    number = parts[2]
    group = f"{number[:-3] or '0'}xxx"
    return f"cves/{year}/{group}/{cve_id.upper()}.json"


def _normalise_cvelist_record(data: Mapping[str, Any], *, raw_hash: str,
                              fetched: float, fallback_vendor: str,
                              fallback_product: str) -> RawCveRecord | None:
    meta = data.get("cveMetadata", {}) if isinstance(data, Mapping) else {}
    cve_id = str(meta.get("cveId", "") or "").upper()
    if not cve_id:
        return None
    cna = (data.get("containers", {}) or {}).get("cna", {}) or {}
    affected = cna.get("affected", []) or []
    vendor = fallback_vendor.lower()
    product = fallback_product.lower()
    version_start = version_end = ""
    start_inc = end_inc = True
    if affected:
        first = affected[0] or {}
        vendor = str(first.get("vendor") or vendor).lower()
        product = str(first.get("product") or product).lower()
        for version in first.get("versions", []) or []:
            if version.get("status") == "affected":
                version_start = str(version.get("version", "") or version_start)
                version_end = str(version.get("lessThan") or version.get("lessThanOrEqual") or version_end)
                end_inc = "lessThanOrEqual" in version
                break
    descriptions = cna.get("descriptions", []) or []
    description = ""
    for entry in descriptions:
        if entry.get("lang") == "en":
            description = entry.get("value", "")
            break
    refs = [ref.get("url", "") for ref in cna.get("references", []) or [] if ref.get("url")]
    cvss_score = 0.0
    cvss_vector = ""
    for metric in cna.get("metrics", []) or []:
        for key in ("cvssV4_0", "cvssV3_1", "cvssV3_0", "cvssV2_0"):
            if key in metric:
                cvss_score = _maybe_float(metric[key].get("baseScore"))
                cvss_vector = str(metric[key].get("vectorString", "") or "")
                break
        if cvss_score:
            break
    return RawCveRecord(
        source="cve_list_v5", cve_id=cve_id, raw=dict(data), raw_hash=raw_hash,
        retrieved_at=fetched, vendor=vendor, product=product,
        version_start=version_start, version_end=version_end,
        version_start_inclusive=start_inc, version_end_inclusive=end_inc,
        cvss_score=cvss_score, cvss_vector=cvss_vector,
        references=refs, description=description,
        published_at=_parse_time(meta.get("datePublished", "")),
    )


def _normalise_vulnx_response(data: Mapping[str, Any], *, raw_hash: str,
                              fetched: float, fallback_vendor: str,
                              fallback_product: str) -> list[RawCveRecord]:
    entries = data.get("cves", data.get("results", [])) if isinstance(data, Mapping) else []
    out: list[RawCveRecord] = []
    for entry in entries or []:
        cve_id = str(entry.get("cve_id") or entry.get("cve") or "").upper()
        if not cve_id:
            continue
        out.append(RawCveRecord(
            source="vulnx", cve_id=cve_id, raw=dict(entry), raw_hash=raw_hash,
            retrieved_at=fetched,
            vendor=str(entry.get("vendor") or fallback_vendor).lower(),
            product=str(entry.get("product") or fallback_product).lower(),
            cvss_score=_maybe_float(entry.get("cvss_score") or entry.get("cvss")),
            references=list(entry.get("references", []) or []),
            description=str(entry.get("description", "") or ""),
            published_at=_parse_time(str(entry.get("published_at", "") or "")),
        ))
    return out


def _record_mentions(rec: RawCveRecord, product: str, vendor: str) -> bool:
    product = product.lower()
    vendor = vendor.lower()
    if product.startswith("cve-") and rec.cve_id.lower() == product:
        return True
    if rec.product and rec.product == product:
        return True
    haystack = " ".join([rec.description, " ".join(rec.references), json.dumps(rec.raw, default=str)]).lower()
    return product in haystack and (not vendor or vendor in haystack or not rec.vendor)


# ── KEV / EPSS enrichment ─────────────────────────────────────────────────────


class KEVAdapter(BaseAdapter):
    name = "kev"

    def fetch(self, product: str, vendor: str, version: str) -> list[PrioritySignal]:  # type: ignore[override]
        if self.mode in {"snapshot", "replay"} and self.snapshot_dir:
            path = os.path.join(self.snapshot_dir, "kev.json")
            if not os.path.exists(path):
                self._record_event(status=BackendStatus.DATASET_MISSING,
                                    detail="kev dataset missing in snapshot")
                return []
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception:
                self._record_event(status=BackendStatus.BACKEND_FAILED,
                                    detail="kev dataset unreadable")
                return []
            signals: list[PrioritySignal] = []
            for entry in data.get("cves", []) if isinstance(data, dict) else data:
                cve_id = entry.get("cve_id") if isinstance(entry, dict) else None
                if not cve_id:
                    continue
                signals.append(PrioritySignal(cve_id=cve_id, in_kev=True,
                                                kev_date_added=float(entry.get("date_added", 0) or 0)))
            return signals
        return []


class EpssAdapter(BaseAdapter):
    name = "epss"

    def fetch(self, product: str, vendor: str, version: str) -> list[PrioritySignal]:  # type: ignore[override]
        if self.mode in {"snapshot", "replay"} and self.snapshot_dir:
            path = os.path.join(self.snapshot_dir, "epss.json")
            if not os.path.exists(path):
                self._record_event(status=BackendStatus.DATASET_MISSING,
                                    detail="epss dataset missing in snapshot")
                return []
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception:
                self._record_event(status=BackendStatus.BACKEND_FAILED,
                                    detail="epss dataset unreadable")
                return []
            signals: list[PrioritySignal] = []
            entries = data.get("cves", []) if isinstance(data, dict) else data
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                cve_id = entry.get("cve_id") or entry.get("cve") or ""
                if not cve_id:
                    continue
                signals.append(PrioritySignal(
                    cve_id=cve_id,
                    epss_score=float(entry.get("epss_score", 0.0) or 0.0),
                    epss_percentile=float(entry.get("percentile", 0.0) or 0.0),
                ))
            return signals
        return []


# ── Snapshot writer ──────────────────────────────────────────────────────────


def write_snapshot(snapshot_dir: str, records: list[RawCveRecord]) -> str:
    """Write *records* into *snapshot_dir* as immutable JSON.

    Returns the SHA-256 manifest hash for the snapshot.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    blob: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        blob.setdefault(rec.source, []).append(rec.to_dict())
    out_path = os.path.join(snapshot_dir, "cves.json")
    with open(out_path, "w") as fh:
        json.dump(blob, fh, sort_keys=True, indent=2, default=str)
    manifest = {
        "snapshot_dir": snapshot_dir,
        "record_count": len(records),
        "by_source": {src: len(items) for src, items in blob.items()},
        "snapshot_hash": hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest(),
    }
    with open(os.path.join(snapshot_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
    return str(manifest["snapshot_hash"])


# ── Registry ──────────────────────────────────────────────────────────────────


class SourceRegistry:
    """Runs all configured adapters and aggregates results per CVE.

    Distinct per-adapter ``BackendStatus`` values are preserved and emitted
    into the ledger; one failing backend never fails the others.
    """

    def __init__(self, adapters: Iterable[BaseAdapter] | None = None,
                 *, ledger: EventLedger | None = None) -> None:
        self.adapters: list[BaseAdapter] = list(adapters or [])
        self.ledger = ledger

    def collect_cves(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        merged: list[RawCveRecord] = []
        for adapter in self.adapters:
            try:
                results = adapter.fetch(product, vendor, version)
            except Exception as exc:                              # noqa: BLE001
                if self.ledger is not None:
                    self.ledger.record(
                        phase="retrieval", stage="applicability",
                        detail=f"{adapter.name} raised {type(exc).__name__}",
                        payload={"source": adapter.name, "status": BackendStatus.BACKEND_FAILED,
                                 "exception": str(exc)[:160]},
                    )
                continue
            if results:
                merged.extend(results)
        # Stable order: by source priority then cve_id.
        priority = {a.name: i for i, a in enumerate(self.adapters)}
        merged.sort(key=lambda r: (priority.get(r.source, 99), r.cve_id))
        return merged

    def collect_priority(self, product: str, vendor: str, version: str) -> dict[str, PrioritySignal]:
        out: dict[str, PrioritySignal] = {}
        for adapter in self.adapters:
            if not isinstance(adapter, (KEVAdapter, EpssAdapter)):
                continue
            try:
                results = adapter.fetch(product, vendor, version)
            except Exception:
                continue
            for sig in results:
                cur = out.setdefault(sig.cve_id, PrioritySignal(cve_id=sig.cve_id))
                if isinstance(adapter, KEVAdapter):
                    cur.in_kev = cur.in_kev or sig.in_kev
                    cur.kev_date_added = max(cur.kev_date_added, sig.kev_date_added)
                elif isinstance(adapter, EpssAdapter):
                    cur.epss_score = max(cur.epss_score, sig.epss_score)
                    cur.epss_percentile = max(cur.epss_percentile, sig.epss_percentile)
        return out
