"""Vendor-advisory source for ``K`` (paper §3.2).

The pinned CVE List V5 snapshot is the experiment's primary corpus, but a case's
advisory can be missing from it (published after the snapshot's cutoff), carry a
placeholder product name, or state a version range the snapshot records
incompletely. Those are data defects of the snapshot, not of the runtime, so the
experiment may mount a **vendor advisory** source alongside it: a JSON Lines file
of advisory records (the vendor's own publication) that the retrieval layer
queries exactly like the CVE-list snapshot — by normalised product and vendor.

Records are ``RawCveRecord``-shaped objects; every record carries the advisory's
own ``cve_id``, ``vendor``, ``product`` and affected-version bounds, so a
candidate is still admitted only when the retrieval returns the CVE it names.

Fail-closed: a path that exists but cannot be read or parsed raises, because a
truncated advisory set must not silently degrade into an empty one.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from src.pipeline.sources import RawCveRecord

#: Where a deployment mounts vendor advisories.
DEFAULT_VENDOR_ADVISORY_PATHS: tuple[str, ...] = (
    "/run/veriplanpt/vendor-advisories.jsonl",
)
VENDOR_ADVISORY_ENV = "PENTEST_VENDOR_ADVISORIES"
SOURCE_NAME = "vendor_advisory"


def resolve_vendor_advisory_path(path: str | Path | None = None) -> Path | None:
    """The advisory file to load, or ``None`` when the run has none."""
    if path is not None:
        return Path(path)
    override = str(os.environ.get(VENDOR_ADVISORY_ENV) or "").strip()
    if override:
        return Path(override)
    for candidate in DEFAULT_VENDOR_ADVISORY_PATHS:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def load_vendor_advisories(path: str | Path | None = None) -> list[RawCveRecord]:
    """Load the mounted vendor advisories (empty when none is mounted)."""
    resolved = resolve_vendor_advisory_path(path)
    if resolved is None or not resolved.is_file():
        return []
    try:
        payload = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"vendor advisory set {resolved} is unreadable: {exc}") from exc
    records: list[RawCveRecord] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"vendor advisory set {resolved} line {number} is not JSON: {exc}") from exc
        if not isinstance(item, dict) or not str(item.get("cve_id") or ""):
            raise ValueError(f"vendor advisory set {resolved} line {number} is not an advisory record")
        item.setdefault("source", SOURCE_NAME)
        item.setdefault("retrieved_at", 0.0)
        # ``raw`` is the advisory as published; when the file does not carry it
        # separately, the record itself is the raw payload.
        raw = item.get("raw")
        if not isinstance(raw, dict):
            raw = {key: value for key, value in item.items() if key != "raw"}
            item["raw"] = raw
        if not str(item.get("raw_hash") or ""):
            item["raw_hash"] = hashlib.sha256(
                json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        records.append(RawCveRecord.from_dict(item))
    return records


def match_advisories(records: Iterable[RawCveRecord], *, product: str, vendor: str) -> list[RawCveRecord]:
    """Advisories whose normalised product (and vendor) match the fingerprint."""
    wanted_product = " ".join(str(product or "").lower().split())
    wanted_vendor = " ".join(str(vendor or "").lower().split())
    out: list[RawCveRecord] = []
    for record in records:
        record_product = " ".join(str(getattr(record, "product", "") or "").lower().split())
        record_vendor = " ".join(str(getattr(record, "vendor", "") or "").lower().split())
        if wanted_product and record_product != wanted_product:
            continue
        if wanted_vendor and record_vendor != wanted_vendor:
            continue
        out.append(record)
    return out


class VendorAdvisoryAdapter:
    """``K`` source backed by a mounted vendor-advisory file."""

    name = SOURCE_NAME

    def __init__(self, *, path: str | Path | None = None, ledger: Any | None = None) -> None:
        self.path = resolve_vendor_advisory_path(path)
        self.ledger = ledger
        self._records: list[RawCveRecord] | None = None

    def records(self) -> list[RawCveRecord]:
        if self._records is None:
            self._records = load_vendor_advisories(self.path)
        return self._records

    def fetch(self, product: str, vendor: str, version: str) -> list[RawCveRecord]:
        if self.path is None or not self.path.is_file():
            return []
        matched = match_advisories(self.records(), product=product, vendor=vendor)
        if self.ledger is not None:
            self.ledger.record(
                phase="retrieval", stage="applicability",
                detail=(f"{SOURCE_NAME} records" if matched else f"{SOURCE_NAME} had no match"),
                payload={"count": len(matched), "product": product, "version": version,
                         "cve_ids": [record.cve_id for record in matched]},
            )
        return matched
