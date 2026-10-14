"""Indexed, immutable CVE source snapshots for production runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA = "veriplanpt-cve-source-snapshot-1.0"
INDEX_NAME = "source-snapshot.sqlite3"
RAW_ARCHIVE = "raw/cves.zip"
CANONICAL_INDEX_SHA256 = "31421ea39ed809a34e3bdbfeda5fa34b26ff5ed194247309c60f015b710811bd"
CANONICAL_RECORD_COUNT = 542975
CANONICAL_CVE_COUNT = 354521


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _metric(cna: Mapping[str, Any]) -> tuple[float, str]:
    for metric in cna.get("metrics", []) or []:
        for key in ("cvssV4_0", "cvssV3_1", "cvssV3_0", "cvssV2_0"):
            value = metric.get(key)
            if isinstance(value, Mapping):
                try:
                    score = float(value.get("baseScore", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                return score, str(value.get("vectorString", "") or "")
    return 0.0, ""


def _english_description(cna: Mapping[str, Any]) -> str:
    descriptions = cna.get("descriptions", []) or []
    for entry in descriptions:
        if isinstance(entry, Mapping) and entry.get("lang") == "en":
            return str(entry.get("value", "") or "")
    return ""


def _version_bounds(affected: Mapping[str, Any]) -> tuple[str, str, bool, bool]:
    for version in affected.get("versions", []) or []:
        if not isinstance(version, Mapping) or version.get("status") != "affected":
            continue
        start = str(version.get("version", "") or "")
        end = str(version.get("lessThan") or version.get("lessThanOrEqual") or "")
        return start, end, True, "lessThanOrEqual" in version
    return "", "", True, True


def _rows(raw: bytes, member: str) -> Iterator[tuple[Any, ...]]:
    data = json.loads(raw)
    metadata = data.get("cveMetadata", {}) if isinstance(data, Mapping) else {}
    if metadata.get("state") != "PUBLISHED":
        return
    cve_id = str(metadata.get("cveId", "") or "").upper()
    if not cve_id:
        return
    cna = (data.get("containers", {}) or {}).get("cna", {}) or {}
    score, vector = _metric(cna)
    description = _english_description(cna)
    references = json.dumps(
        [
            str(ref.get("url", ""))
            for ref in cna.get("references", []) or []
            if isinstance(ref, Mapping) and ref.get("url")
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    published_at = _parse_time(str(metadata.get("datePublished", "") or ""))
    raw_hash = hashlib.sha256(raw).hexdigest()
    affected_items = [item for item in cna.get("affected", []) or [] if isinstance(item, Mapping)]
    if not affected_items:
        affected_items = [{}]
    for affected in affected_items:
        start, end, start_inclusive, end_inclusive = _version_bounds(affected)
        yield (
            cve_id,
            str(affected.get("vendor", "") or "").lower(),
            str(affected.get("product", "") or "").lower(),
            start,
            end,
            int(start_inclusive),
            int(end_inclusive),
            score,
            vector,
            references,
            description,
            published_at,
            member,
            raw_hash,
        )


def build_snapshot_index(
    snapshot_dir: str | Path,
    *,
    cutoff: str,
    upstream_release: str,
    upstream_asset_url: str,
    upstream_asset_path: str,
    upstream_asset_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic lookup index over an official CVE List V5 archive."""
    root = Path(snapshot_dir).resolve()
    archive = root / RAW_ARCHIVE
    if not archive.is_file():
        raise ValueError(f"source snapshot archive missing: {archive}")
    upstream_asset = root / upstream_asset_path
    if not upstream_asset.is_file() or _sha256(upstream_asset) != upstream_asset_sha256:
        raise ValueError("source snapshot upstream asset hash mismatch")
    index = root / INDEX_NAME
    if index.exists():
        raise ValueError(f"source snapshot index already exists: {index}")
    connection = sqlite3.connect(index)
    cve_ids: set[str] = set()
    record_count = 0
    semantic = hashlib.sha256()
    next_commit = 10_000
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA page_size=4096;
            CREATE TABLE records (
                cve_id TEXT NOT NULL,
                vendor TEXT NOT NULL,
                product TEXT NOT NULL,
                version_start TEXT NOT NULL,
                version_end TEXT NOT NULL,
                version_start_inclusive INTEGER NOT NULL,
                version_end_inclusive INTEGER NOT NULL,
                cvss_score REAL NOT NULL,
                cvss_vector TEXT NOT NULL,
                references_json TEXT NOT NULL,
                description TEXT NOT NULL,
                published_at REAL NOT NULL,
                zip_member TEXT NOT NULL,
                raw_hash TEXT NOT NULL,
                PRIMARY KEY (cve_id, vendor, product, version_start, version_end)
            );
            """
        )
        with zipfile.ZipFile(archive) as source:
            members = sorted(
                name
                for name in source.namelist()
                if name.startswith("cves/") and name.endswith(".json") and "/CVE-" in name
            )
            for member in members:
                raw = source.read(member)
                for row in _rows(raw, member):
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row
                    )
                    if cursor.rowcount == 1:
                        record_count += 1
                        cve_ids.add(str(row[0]))
                        semantic.update(_canonical({
                            "cve_id": row[0], "vendor": row[1], "product": row[2],
                            "version_start": row[3], "version_end": row[4],
                            "raw_hash": row[13],
                        }))
                if record_count >= next_commit:
                    connection.commit()
                    next_commit += 10_000
        connection.executescript(
            """
            CREATE INDEX records_product_vendor ON records(product, vendor);
            CREATE INDEX records_cve_id ON records(cve_id);
            VACUUM;
            """
        )
        connection.commit()
    except Exception:
        connection.close()
        index.unlink(missing_ok=True)
        raise
    finally:
        if index.exists():
            connection.close()

    identity = {
        "schema": SCHEMA,
        "cutoff": cutoff,
        "source": "CVE List V5",
        "upstream_release": upstream_release,
        "upstream_asset_url": upstream_asset_url,
        "upstream_asset_path": upstream_asset_path,
        "upstream_asset_sha256": upstream_asset_sha256,
        "raw_archive": RAW_ARCHIVE,
        "raw_archive_sha256": _sha256(archive),
        "index": INDEX_NAME,
        "index_sha256": _sha256(index),
        "semantic_index_sha256": semantic.hexdigest(),
        "cve_count": len(cve_ids),
        "record_count": record_count,
        "sqlite_version": sqlite3.sqlite_version,
    }
    manifest = {**identity, "snapshot_hash": hashlib.sha256(_canonical(identity)).hexdigest()}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def _assert_regular_tree(root: Path) -> None:
    """Reject symlinked snapshot roots and members before resolving them."""
    if root.is_symlink() or not root.is_dir():
        if not root.exists():
            raise ValueError("source snapshot manifest missing")
        raise ValueError("source snapshot root must be a real directory")
    for path in (root / "manifest.json", root / INDEX_NAME, root / "raw"):
        if path.is_symlink():
            raise ValueError(f"source snapshot must not contain symlinks: {path}")
    raw = root / RAW_ARCHIVE
    if raw.is_symlink():
        raise ValueError(f"source snapshot must not contain symlinks: {raw}")


def validate_source_snapshot(
    snapshot_dir: str | Path, *, full: bool = True, expected_hash: str = "",
    official: bool = False,
) -> dict[str, Any]:
    """Validate hashes, schema and a read-only index query without provider access."""
    root = Path(snapshot_dir)
    _assert_regular_tree(root)
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source snapshot manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("source snapshot schema mismatch")
    if official:
        expected_official = {
            "cutoff": "2026-08-01T00:00:00Z",
            "source": "CVE List V5",
            "upstream_release": "cve_2026-08-01_0000Z",
            "upstream_asset_sha256": "700234c38e2e4158a320f6fd9c7aeecbe5dc5510fbfb34bc4195d1f012c263cc",
            "index_sha256": CANONICAL_INDEX_SHA256,
            "cve_count": CANONICAL_CVE_COUNT,
            "record_count": CANONICAL_RECORD_COUNT,
        }
        for key, value in expected_official.items():
            if manifest.get(key) != value:
                raise ValueError(f"official source snapshot {key} mismatch")
    identity = {key: value for key, value in manifest.items() if key != "snapshot_hash"}
    expected = hashlib.sha256(_canonical(identity)).hexdigest()
    if manifest.get("snapshot_hash") != expected:
        raise ValueError("source snapshot identity hash mismatch")
    if expected_hash and expected_hash != expected:
        raise ValueError("source snapshot expected hash mismatch")
    for field, path_field in (
        ("upstream_asset_sha256", "upstream_asset_path"),
        ("raw_archive_sha256", "raw_archive"),
        ("index_sha256", "index"),
    ):
        path = root / str(manifest.get(path_field, ""))
        if not path.is_file() or (full and _sha256(path) != manifest.get(field)):
            if path_field != "index" and not path.is_file():
                continue
            raise ValueError(f"source snapshot {path_field} hash mismatch")
    index = root / str(manifest["index"])
    if index.is_symlink():
        raise ValueError("source snapshot index must not be a symlink")
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        count = int(connection.execute("SELECT count(*) FROM records").fetchone()[0])
        integrity = str(
            connection.execute("PRAGMA integrity_check").fetchone()[0] if full else "ok"
        )
    finally:
        connection.close()
    if count != int(manifest.get("record_count", -1)) or integrity != "ok":
        raise ValueError("source snapshot index validation failed")
    return manifest


PLACEHOLDER_PRODUCTS = frozenset({"", "-", "*", "n/a", "na", "none", "unknown", "various"})


def read_indexed_records(
    snapshot_dir: str | Path, *, source: str, product: str, vendor: str
) -> list[dict[str, Any]] | None:
    """Return indexed records, or ``None`` when the snapshot uses the legacy JSON contract."""
    root = Path(snapshot_dir).resolve()
    index = root / INDEX_NAME
    archive = root / RAW_ARCHIVE
    if not index.is_file():
        return None
    if source != "cve_list_v5":
        return []
    # Retrieval is product-keyed: a placeholder is not a product identity, and
    # querying one would return an arbitrary slice of the corpus.
    if str(product or "").strip().lower() in PLACEHOLDER_PRODUCTS:
        return []
    query = "SELECT * FROM records WHERE product = ?"
    params: list[Any] = [product.lower()]
    if vendor:
        query += " AND vendor = ?"
        params.append(vendor.lower())
    # The consumer truncates K(p) to its per-service budget, so the newest
    # advisories must come first; cve_id keeps the order deterministic.
    query += " ORDER BY published_at DESC, cve_id"
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()
    try:
        retrieved_at = datetime.fromisoformat(
            str(validate_source_snapshot(root, full=False)["cutoff"]).replace("Z", "+00:00")
        ).astimezone(UTC).timestamp()
    except Exception:
        manifest_file = root / "manifest.json"
        cutoff_val = "2026-08-01T00:00:00Z"
        if manifest_file.is_file():
            try:
                cutoff_val = json.loads(manifest_file.read_text()).get("cutoff", cutoff_val)
            except Exception:
                pass
        retrieved_at = datetime.fromisoformat(cutoff_val.replace("Z", "+00:00")).astimezone(UTC).timestamp()
    out: list[dict[str, Any]] = []
    if archive.is_file():
        with zipfile.ZipFile(archive) as source_zip:
            for row in rows:
                raw_bytes = source_zip.read(str(row["zip_member"]))
                if hashlib.sha256(raw_bytes).hexdigest() != row["raw_hash"]:
                    raise ValueError(f"source snapshot raw record hash mismatch: {row['cve_id']}")
                out.append({
                    "source": "cve_list_v5",
                    "cve_id": row["cve_id"],
                    "raw": json.loads(raw_bytes),
                    "raw_hash": row["raw_hash"],
                    "retrieved_at": retrieved_at,
                    "vendor": row["vendor"],
                    "product": row["product"],
                    "version_start": row["version_start"],
                    "version_end": row["version_end"],
                    "version_start_inclusive": bool(row["version_start_inclusive"]),
                    "version_end_inclusive": bool(row["version_end_inclusive"]),
                    "cvss_score": row["cvss_score"],
                    "cvss_vector": row["cvss_vector"],
                    "cpe_candidates": [],
                    "references": json.loads(row["references_json"]),
                    "description": row["description"],
                    "published_at": row["published_at"],
                })
    else:
        for row in rows:
            out.append({
                "source": "cve_list_v5",
                "cve_id": row["cve_id"],
                "raw": {"description": row["description"]},
                "raw_hash": row["raw_hash"],
                "retrieved_at": retrieved_at,
                "vendor": row["vendor"],
                "product": row["product"],
                "version_start": row["version_start"],
                "version_end": row["version_end"],
                "version_start_inclusive": bool(row["version_start_inclusive"]),
                "version_end_inclusive": bool(row["version_end_inclusive"]),
                "cvss_score": row["cvss_score"],
                "cvss_vector": row["cvss_vector"],
                "cpe_candidates": [],
                "references": json.loads(row["references_json"]),
                "description": row["description"],
                "published_at": row["published_at"],
            })
    return out
