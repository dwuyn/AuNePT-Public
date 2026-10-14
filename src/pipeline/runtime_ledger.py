"""Thread-safe, durable ledger for observed gateway invocations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


class BillingUnknownError(RuntimeError):
    """A provider response exists but billing evidence is not durable/complete."""

    billing_unknown = True
    model_response_received = True

    def __init__(self, message: str, *, failure_id: str = "") -> None:
        super().__init__(message)
        self.failure_id = failure_id
        self.retryable = False


class BillableInvocationError(RuntimeError):
    """A known-billed invocation failed after the provider response."""

    billing_unknown = False
    model_response_received = True
    billable_model_response = True

    def __init__(self, message: str, *, cost_usd: float) -> None:
        super().__init__(message)
        self.cost_usd = float(cost_usd)


class InvocationConflictError(RuntimeError):
    """A restarted call reused an invocation slot with a different request."""


def _hash(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class InvocationLedger:
    """Capture response state and atomically persist it before gateway return."""

    CURRENT_SCHEMA = "1.3.0"
    _PHASES = {"canary", "smoke", "sweep", "confirmation", "benchmark"}

    def __init__(
        self,
        *,
        phase: str,
        gateway_relay_lock_hash: str,
        path: str | Path | None = None,
        epoch: str = "",
    ) -> None:
        if phase not in self._PHASES:
            raise ValueError("invocation ledger phase is not a locked runtime stage")
        self.phase = phase
        self.gateway_relay_lock_hash = gateway_relay_lock_hash
        self.epoch = str(epoch)
        self.path = Path(path).resolve() if path is not None else None
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._attempts: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "InvocationLedger":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("invocation ledger must be a JSON object")
        schema = str(value.get("schema_version", ""))
        if schema not in {"1.0.0", "1.1.0", "1.2.0", "1.3.0", cls.CURRENT_SCHEMA}:
            raise ValueError("unsupported invocation ledger schema")
        phase = str(value.get("phase", ""))
        lock_hash = str(value.get("gateway_relay_lock_hash", ""))
        ledger = cls(
            phase=phase, gateway_relay_lock_hash=lock_hash, path=source,
            epoch=str(value.get("epoch", "")),
        )
        rows = value.get("invocations", [])
        if not isinstance(rows, list):
            raise ValueError("invocation ledger invocations must be an array")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("invocation ledger record must be an object")
            record = dict(raw)
            # r5 records are completed, known invocations.  Normalize them at
            # read time while keeping the original file immutable.
            if schema == "1.0.0":
                record.setdefault("outcome", "completed")
                record.setdefault("model_response_received", True)
            ledger._validate_record(record)
            ledger._records.append(record)
        attempts = value.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError("invocation ledger attempts must be an array")
        for raw in attempts:
            if not isinstance(raw, Mapping):
                raise ValueError("invocation ledger attempt must be an object")
            record = dict(raw)
            ledger._validate_attempt(record)
            ledger._attempts.append(record)
        failures = value.get("failure_evidence", [])
        if not isinstance(failures, list):
            raise ValueError("invocation ledger failure_evidence must be an array")
        for raw in failures:
            if not isinstance(raw, Mapping):
                raise ValueError("invocation failure evidence record must be an object")
            record = dict(raw)
            ledger._validate_failure(record)
            ledger._failures.append(record)
        return ledger

    read = from_file

    @staticmethod
    def _normalized_usage(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
        if usage is None:
            return None
        normalized = {
            "input_tokens": int(usage.get("input_tokens", -1)),
            "output_tokens": int(usage.get("output_tokens", -1)),
            "total_tokens": int(usage.get("total_tokens", -1)),
        }
        if any(value < 0 for value in normalized.values()):
            raise ValueError("observed invocation usage is incomplete")
        return normalized

    @staticmethod
    def _validate_record(record: Mapping[str, Any]) -> None:
        if not str(record.get("run_id", "")) or not str(record.get("model_label", "")):
            raise ValueError("invocation ledger record requires run_id and model label")
        outcome = str(record.get("outcome", ""))
        status = str(record.get("billing_status", ""))
        if outcome not in {"completed", "post_response_failure"}:
            raise ValueError("invocation ledger outcome is invalid")
        if status not in {"known", "unknown"}:
            raise ValueError("invocation ledger billing status is invalid")
        if status == "known":
            if record.get("cost_usd") is None or record.get("usage") is None:
                raise ValueError("known billing requires usage and cost")
            if float(record["cost_usd"]) < 0:
                raise ValueError("invocation cost cannot be negative")
        elif record.get("cost_usd") is not None or record.get("usage") is not None:
            raise ValueError("unknown billing cannot carry partial cost evidence")

    @staticmethod
    def _validate_failure(record: Mapping[str, Any]) -> None:
        required = {
            "failure_id", "run_id", "model_label", "model_profile_hash", "request_sha256",
            "upstream_status", "exception_class", "google_request_id", "error_body_sha256",
            "retryable", "model_response_received",
        }
        if not required.issubset(record):
            raise ValueError("pre-response failure evidence is incomplete")
        for name in ("failure_id", "run_id", "model_label", "exception_class"):
            if not str(record[name]).strip():
                raise ValueError(f"pre-response failure evidence {name} is required")
        for name in ("model_profile_hash", "request_sha256", "error_body_sha256"):
            value = str(record[name])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"pre-response failure evidence {name} must be SHA-256")
        status = record["upstream_status"]
        if status is not None and (not isinstance(status, int) or not 100 <= status <= 599):
            raise ValueError("pre-response failure upstream_status is invalid")
        if not isinstance(record["retryable"], bool) or record["model_response_received"] is not False:
            raise ValueError("pre-response failure response/retry flags are invalid")

    @staticmethod
    def _validate_attempt(record: Mapping[str, Any]) -> None:
        required = {
            "event_type", "run_id", "model_label", "model_profile_hash",
            "request_sha256", "gateway_request_count", "provider_attempt_count",
            "provider_response_count", "observed_at",
        }
        if not required.issubset(record) or record.get("event_type") != "attempt_started":
            raise ValueError("invocation attempt_started evidence is incomplete")
        for name in ("run_id", "model_label", "model_profile_hash", "request_sha256"):
            if not str(record.get(name, "")).strip():
                raise ValueError(f"invocation attempt {name} is required")
        for name in ("model_profile_hash", "request_sha256"):
            value = str(record[name])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"invocation attempt {name} must be SHA-256")
        for name in ("gateway_request_count", "provider_attempt_count", "provider_response_count"):
            if not isinstance(record[name], int) or int(record[name]) < 0:
                raise ValueError(f"invocation attempt {name} is invalid")

    def _value(self) -> dict[str, Any]:
        return {
            "schema_version": self.CURRENT_SCHEMA,
            "phase": self.phase,
            "epoch": self.epoch,
            "gateway_relay_lock_hash": self.gateway_relay_lock_hash,
            "attempts": [dict(record) for record in self._attempts],
            "invocations": [dict(record) for record in self._records],
            "failure_evidence": [dict(record) for record in self._failures],
        }

    def _write_locked(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._value(), indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    def counter_snapshot(self, run_id: str, *, epoch: str = "") -> dict[str, int]:
        """Return durable gateway/attempt/response counters for one cell."""
        with self._lock:
            attempts = [
                row for row in self._attempts
                if row.get("run_id") == run_id
                and (not epoch or row.get("epoch") == str(epoch))
            ]
            responses = [
                row for row in self._records
                if row.get("run_id") == run_id
                and row.get("model_response_received", True) is not False
                and (not epoch or row.get("epoch") == str(epoch))
            ]
            return {
                "gateway_request_count": len(attempts),
                "provider_attempt_count": len(attempts),
                "provider_response_count": len(responses),
            }

    def record_attempt_started(
        self, *, run_id: str, model_label: str, request: Any,
        model_profile_hash: str, epoch: str = "", call_index: int | None = None,
    ) -> dict[str, Any]:
        """Durably append ``attempt_started`` before touching the provider."""
        if not run_id or not model_label:
            raise ValueError("invocation attempt requires run ID and model label")
        if call_index is not None and int(call_index) < 0:
            raise ValueError("invocation attempt call_index cannot be negative")
        identity_epoch = str(epoch or self.epoch)
        with self._lock:
            current = self.counter_snapshot(run_id, epoch=identity_epoch)
            record: dict[str, Any] = {
                "event_type": "attempt_started",
                "run_id": str(run_id),
                "model_label": str(model_label),
                "model_profile_hash": str(model_profile_hash),
                "request_sha256": _hash(request),
                "gateway_request_count": current["gateway_request_count"] + 1,
                "provider_attempt_count": current["provider_attempt_count"] + 1,
                "provider_response_count": current["provider_response_count"],
                "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "gateway_relay_lock_hash": self.gateway_relay_lock_hash,
            }
            if identity_epoch:
                record["epoch"] = identity_epoch
            if call_index is not None:
                record["call_index"] = int(call_index)
            self._validate_attempt(record)
            self._attempts.append(record)
            if self.path is not None:
                try:
                    self._write_locked(self.path)
                except Exception as exc:
                    # The provider has not been touched, but durable billing
                    # state is already unknowable. Keep a sticky in-memory
                    # tombstone so callers cannot misclassify this as an
                    # unused slot or retry it.
                    self._records.append({
                        "run_id": str(run_id), "model_label": str(model_label),
                        "phase": self.phase, "request_sha256": _hash(request),
                        "response_sha256": None, "usage": None, "cost_usd": None,
                        "billing_status": "unknown", "outcome": "completed",
                        "model_response_received": False,
                        "observed_at": record["observed_at"],
                        "gateway_relay_lock_hash": self.gateway_relay_lock_hash,
                        "epoch": identity_epoch,
                        "call_index": call_index,
                    })
                    raise BillingUnknownError(
                        "gateway could not durably persist attempt_started state"
                    ) from exc
            return dict(record)

    def record(
        self,
        *,
        run_id: str,
        model_label: str,
        request: Any,
        response: Any = None,
        response_hash: str | None = None,
        usage: Mapping[str, Any] | None = None,
        cost_usd: float | None,
        billing_status: str,
        outcome: str = "completed",
        model_response_received: bool = True,
        epoch: str = "",
        call_index: int | None = None,
        model_profile_hash: str = "",
        replay_count: int = 0,
    ) -> dict[str, Any]:
        if not run_id or not model_label:
            raise ValueError("invocation ledger requires run_id and model label")
        if billing_status not in {"known", "unknown"}:
            raise ValueError("invocation ledger billing status is invalid")
        if outcome not in {"completed", "post_response_failure"}:
            raise ValueError("invocation ledger outcome is invalid")
        if not model_response_received:
            raise ValueError("invocation ledger records require a provider response")
        normalized = self._normalized_usage(usage)
        cost = None if cost_usd is None else float(cost_usd)
        if billing_status == "known":
            if normalized is None or cost is None:
                raise ValueError("known billing requires complete usage and cost")
            if cost < 0:
                raise ValueError("invocation cost cannot be negative")
        elif normalized is not None or cost is not None:
            raise ValueError("unknown billing cannot carry partial cost evidence")
        if response_hash is None and response is not None:
            response_hash = _hash(response)
        record = {
            "run_id": run_id,
            "model_label": model_label,
            "phase": self.phase,
            "request_sha256": _hash(request),
            "response_sha256": response_hash,
            "usage": normalized,
            "cost_usd": cost,
            "billing_status": billing_status,
            "outcome": outcome,
            "model_response_received": model_response_received,
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "gateway_relay_lock_hash": self.gateway_relay_lock_hash,
        }
        identity_epoch = str(epoch or self.epoch)
        if identity_epoch:
            record["epoch"] = identity_epoch
        if call_index is not None:
            if int(call_index) < 0:
                raise ValueError("invocation call_index cannot be negative")
            record["call_index"] = int(call_index)
        if model_profile_hash:
            record["model_profile_hash"] = str(model_profile_hash)
        if response is not None:
            record["response"] = response
        counters = self.counter_snapshot(run_id, epoch=identity_epoch)
        counters["provider_response_count"] += 1
        record.update(counters)
        record["replay_count"] = int(replay_count)
        self._validate_record(record)
        with self._lock:
            self._records.append(record)
            try:
                if self.path is not None:
                    self._write_locked(self.path)
            except Exception:
                # Atomic write-through failed.  An in-memory tombstone is
                # sticky so a durable classification is never destroyed:
                # later lookups report "unknown", never "none".  The gateway
                # converts this into BillingUnknownError and the coordinator
                # halts with billing_unknown instead of misreading the moment
                # as a pre-response infrastructure failure.
                record["billing_status"] = "unknown"
                record["usage"] = None
                record["cost_usd"] = None
                raise
        return dict(record)

    def record_failure(
        self,
        *,
        failure_id: str,
        run_id: str,
        model_label: str,
        model_profile_hash: str,
        request: Any,
        upstream_status: int | None,
        exception_class: str,
        google_request_id: str = "",
        error_body_hash: str,
        retryable: bool,
        epoch: str = "",
        call_index: int | None = None,
    ) -> dict[str, Any]:
        """Append redacted pre-response evidence without retaining request/body data."""
        record: dict[str, Any] = {
            "failure_id": str(failure_id),
            "run_id": str(run_id),
            "model_label": str(model_label),
            "model_profile_hash": str(model_profile_hash),
            "request_sha256": _hash(request),
            "upstream_status": upstream_status,
            "exception_class": str(exception_class),
            "google_request_id": str(google_request_id),
            "error_body_sha256": str(error_body_hash),
            "retryable": bool(retryable),
            "model_response_received": False,
            "phase": self.phase,
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "gateway_relay_lock_hash": self.gateway_relay_lock_hash,
        }
        identity_epoch = str(epoch or self.epoch)
        if identity_epoch:
            record["epoch"] = identity_epoch
        if call_index is not None:
            if int(call_index) < 0:
                raise ValueError("failure call_index cannot be negative")
            record["call_index"] = int(call_index)
        record.update(self.counter_snapshot(run_id, epoch=identity_epoch))
        self._validate_failure(record)
        with self._lock:
            self._failures.append(record)
            if self.path is not None:
                self._write_locked(self.path)
        return dict(record)

    def replay_or_conflict(
        self, *, epoch: str, run_id: str, call_index: int, request: Any,
        model_profile_hash: str,
    ) -> dict[str, Any] | None:
        """Replay one durable response, or fail closed on slot reuse.

        The identity is intentionally scoped to this ledger's epoch and never
        searches another run or another ledger file.
        """
        request_hash = _hash(request)
        with self._lock:
            matches = [
                row for row in self._records
                if row.get("epoch") == str(epoch)
                and row.get("run_id") == run_id
                and int(row.get("call_index", -1)) == int(call_index)
            ]
            if not matches:
                return None
            row = matches[0]
            if row.get("request_sha256") != request_hash or row.get("model_profile_hash") != model_profile_hash:
                raise InvocationConflictError(
                    "durable invocation slot reused with a different request hash or model profile"
                )
            if int(row.get("replay_count", 0)) >= 1:
                raise InvocationConflictError(
                    "durable invocation restart limit exceeded for this run"
                )
            row["replay_count"] = int(row.get("replay_count", 0)) + 1
            if self.path is not None:
                self._write_locked(self.path)
            return dict(row)

    def provider_call_count(self, run_id: str, *, epoch: str = "") -> int:
        """Count provider attempts, including pre-response failures."""
        with self._lock:
            attempts = [
                row for row in self._attempts
                if row.get("run_id") == run_id
                and (not epoch or row.get("epoch") == str(epoch))
            ]
            if attempts:
                return len(attempts)
            # Read compatibility for r10.5/r10.6 ledgers that predate the
            # durable attempt_started event.
            return sum(
                1 for row in self._records
                if row.get("run_id") == run_id
                and (not epoch or row.get("epoch") == str(epoch))
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def failure_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._failures]

    def lookup(self, run_id: str) -> dict[str, Any]:
        rows = [row for row in self.snapshot() if row["run_id"] == run_id]
        if not rows:
            return {"found": False, "billing_status": "none", "cost_usd": 0.0}
        if any(row["billing_status"] == "unknown" for row in rows):
            return {
                "found": True, "billing_status": "unknown", "cost_usd": None,
                "records": rows,
            }
        return {
            "found": True,
            "billing_status": "known",
            "cost_usd": sum(float(row["cost_usd"]) for row in rows),
            "records": rows,
        }

    def latest_terminal_state(
        self, *, run_id: str, epoch: str = "", call_index: int | None = None,
    ) -> dict[str, Any]:
        """Return the terminal state for the latest exact invocation attempt.

        This query is deliberately slot-aware.  A relay timeout may race a
        provider response, so callers must settle the exact attempt rather
        than infer retryability from the HTTP error text.  The returned
        evidence contains hashes and identifiers only; request and response
        payloads are never included.
        """
        if not run_id:
            raise ValueError("terminal-state lookup requires a run ID")
        if call_index is not None and int(call_index) < 0:
            raise ValueError("terminal-state lookup call_index cannot be negative")
        requested_epoch = str(epoch)
        with self._lock:
            attempts = [
                row for row in self._attempts
                if row.get("run_id") == run_id
                and (not requested_epoch or row.get("epoch") == requested_epoch)
                and (call_index is None or int(row.get("call_index", -1)) == int(call_index))
            ]
            attempt = dict(attempts[-1]) if attempts else None
            identity_epoch = requested_epoch or str((attempt or {}).get("epoch", ""))
            identity_call_index = (
                int(call_index) if call_index is not None else
                (int(attempt["call_index"]) if attempt and attempt.get("call_index") is not None else None)
            )

            def matches(row: Mapping[str, Any]) -> bool:
                return (
                    row.get("run_id") == run_id
                    and (not identity_epoch or row.get("epoch") == identity_epoch)
                    and (identity_call_index is None or int(row.get("call_index", -1)) == identity_call_index)
                )

            records = [row for row in self._records if matches(row)]
            failures = [row for row in self._failures if matches(row)]
            base: dict[str, Any] = {
                "run_id": run_id,
                "epoch": identity_epoch,
                "call_index": identity_call_index,
                "attempt": attempt,
            }
            known = [row for row in records if row.get("billing_status") == "known"]
            if known:
                base.update({
                    "state": "known_response",
                    "cost_usd": sum(float(row["cost_usd"]) for row in known),
                    "record": dict(known[-1]),
                })
                return base
            unknown = [row for row in records if row.get("billing_status") == "unknown"]
            if unknown:
                base.update({"state": "unknown_billing", "record": dict(unknown[-1])})
                return base
            pre_response = [
                row for row in failures
                if row.get("model_response_received") is False
                and not bool(row.get("billing_unknown", False))
            ]
            if pre_response:
                base.update({"state": "pre_response_failure", "record": dict(pre_response[-1])})
                return base
            if any(bool(row.get("billing_unknown", False)) for row in failures):
                base.update({"state": "unknown_billing", "record": dict(failures[-1])})
                return base
            base["state"] = "none"
            return base

    def aggregate(self, run_id: str) -> dict[str, Any]:
        state = self.lookup(run_id)
        if not state["found"]:
            raise ValueError(f"no observed gateway invocation for {run_id}")
        if state["billing_status"] != "known":
            raise BillingUnknownError(f"billing is unknown for {run_id}")
        rows = state["records"]
        return {
            "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in rows),
            "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
            "usd": float(state["cost_usd"]),
        }

    def write(self, path: str | Path | None = None) -> Path:
        destination = Path(path).resolve() if path is not None else self.path
        if destination is None:
            raise ValueError("invocation ledger write requires a path")
        with self._lock:
            self.path = destination
            return self._write_locked(destination)


def invocation_ledger_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
