"""Fail-closed approval v2 verification for paid experiment stages."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

APPROVAL_SCHEMA_VERSION = "2.0.0"
# This is deliberately a public value.  The matching private key never lives
# in Git; deployments may override the pinned key through their secret store.
PINNED_APPROVER_PUBLIC_KEY = ""


def canonical_approval_payload(approval: Mapping[str, Any]) -> bytes:
    """Serialize approval JSON canonically, excluding transport metadata."""
    payload = {key: value for key, value in approval.items() if key not in {"signature", "signature_path"}}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def approval_plan_hash(approval: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_approval_payload(approval)).hexdigest()


def _parse_utc(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"approval {name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"approval {name} must include a timezone")
    return parsed.astimezone(UTC)


def verify_approval(
    approval: Mapping[str, Any], *, scope: str, plan_hash: str, cell_count: int,
    cost_ceiling_usd: float, now: datetime | None = None, signature_path: str | Path | None = None,
    public_key: str | None = None,
) -> dict[str, Any]:
    """Validate scope/plan/cost and a detached minisign signature.

    Missing minisign, key, signature, stale plans and wrong scope all fail
    closed before a paid runner is allowed to start.
    """
    required = {
        "schema_version", "scope", "plan_hash", "cell_count", "issued_at",
        "expires_at", "cost_ceiling_usd", "approver_key_id",
    }
    missing = sorted(required.difference(approval))
    if missing:
        raise ValueError(f"approval missing field(s): {', '.join(missing)}")
    if str(approval["schema_version"]) != APPROVAL_SCHEMA_VERSION:
        raise ValueError("approval schema_version must be 2.0.0")
    if str(approval["scope"]) != scope:
        raise ValueError("approval scope mismatch")
    if str(approval["plan_hash"]) != plan_hash:
        raise ValueError("approval plan hash mismatch")
    if int(approval["cell_count"]) != int(cell_count):
        raise ValueError("approval cell count mismatch")
    ceiling = float(approval["cost_ceiling_usd"])
    if ceiling <= 0 or ceiling > float(cost_ceiling_usd):
        raise ValueError("approval cost ceiling exceeds authority")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = _parse_utc(approval["issued_at"], "issued_at")
    expires = _parse_utc(approval["expires_at"], "expires_at")
    if expires <= issued or current < issued or current >= expires:
        raise ValueError("approval is outside its validity window")

    key = public_key or PINNED_APPROVER_PUBLIC_KEY
    sig = Path(signature_path) if signature_path is not None else None
    if not key or sig is None or not sig.is_file():
        raise ValueError("approval requires a pinned public key and detached .minisig")
    minisign = shutil.which("minisign")
    if minisign is None:
        raise ValueError("minisign is unavailable; refusing unsigned approval")
    with tempfile.TemporaryDirectory(prefix="veriplanpt-approval-") as temp:
        payload_path = Path(temp) / "approval.json"
        payload_path.write_bytes(canonical_approval_payload(approval))
        result = subprocess.run(
            [minisign, "-Vm", str(payload_path), "-P", key, "-x", str(sig)],
            capture_output=True, text=True, check=False,
        )
    if result.returncode != 0:
        raise ValueError("approval minisign verification failed")
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "scope": scope,
        "plan_hash": plan_hash,
        "cell_count": int(cell_count),
        "cost_ceiling_usd": ceiling,
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
    }
