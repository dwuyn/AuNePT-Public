"""
src/pipeline/ledger.py
──────────────────────
Append-only structured event ledger.

This is the single source of truth for metrics. Nothing is recomputed from
agent/executor state; every measurement derives from these structured events.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

_PUBLIC_FORBIDDEN_KEYS = frozenset({
    "cve", "cve_id", "truth", "oracle_truth", "hidden_truth", "hidden_version",
    "canonical_case_id", "alias", "alias_metadata", "decoy", "decoy_metadata",
    "access_token", "refresh_token", "service_account", "private_key", "adc",
    "google_adc", "google_application_credentials", "docker_socket", "docker_sock",
})


def _public_key_forbidden(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _PUBLIC_FORBIDDEN_KEYS


def _public_project(value: Any) -> Any:
    """Project nested event payloads onto the public, non-sensitive wire shape."""
    if isinstance(value, Mapping):
        return {
            str(key): _public_project(child)
            for key, child in value.items()
            if not _public_key_forbidden(key)
        }
    if isinstance(value, list):
        return [_public_project(child) for child in value]
    if isinstance(value, tuple):
        return [_public_project(child) for child in value]
    return value

# Allowed semantic outcomes. ``vulnerability_confirmed`` and
# ``task_proof_obtained`` are progress facts, not interchangeable success
# states. Only independent ``task_proof_obtained`` counts toward the primary
# metric.
ALLOWED_OUTCOMES = frozenset({
    "vulnerability_confirmed",
    "task_proof_obtained",
    "execution_failed",
    "not_applicable",
    "not_executable",
    "blocked_by_policy",
    "no_truth",          # B6: run completed but no evaluator truth was supplied
    "infrastructure_failure",
})

# Normalized failure classes assigned when an outcome is a failure.
FAILURE_CLASSES = frozenset({
    "identity_mismatch",
    "version_mismatch",
    "platform_mismatch",
    "auth_prereq",
    "scope_violation",
    "policy_block",
    "procedure_incomplete",
    "missing_dependency",
    "timeout",
    "command_invalid",
    "oracle_reject",
    "cleanup_failed",
    "budget_exceeded",
    "backend_failed",
    "dataset_missing",
    "infrastructure_failure",
    "unknown",
    # Exploit-skill compiler/runtime taxonomy.  Legacy values above remain
    # readable so frozen v1-v6 ledgers replay unchanged.
    "artifact_missing",
    "dependency_missing",
    "syntax_invalid",
    "option_invalid",
    "unsupported_target",
    "negative_check",
    "listener_failed",
    "job_failed",
    "session_not_created",
    "runtime_error",
    "proof_rejected",
    "fabricated_cve",
    "wrong_applicability",
    "false_success",
})

# Stages tracked separately per the handoff.
TRACKED_STAGES = frozenset({
    "vulnerability_confirmation",
    "task_proof",
    "execution_failure",
    "applicability",
    "policy_decision",
    "cleanup",
})


@dataclass
class Event:
    run_id: str = ""
    event_id: str = ""
    phase: str = ""                 # recon|evidence|retrieval|candidates|queue|execution|oracle|cleanup
    stage: str = ""                 # one of TRACKED_STAGES when applicable
    service: str = ""               # target_ip:port:name
    cve_id: str = ""
    candidate_id: str = ""
    method: str = ""                # candidate kind
    command_id: str = ""
    attempt_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: float = 0.0
    outcome: str = ""               # one of ALLOWED_OUTCOMES (or "" for non-terminal)
    failure_class: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    artifact_ref: str = ""
    proof_ref: str = ""
    scope_decision: str = ""        # "allowed" | "blocked" | ""
    policy_decision: str = ""      # "execute" | "blocked" | ""
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize only the public event contract; retain raw fields internally."""
        return {
            key: _public_project(value)
            for key, value in asdict(self).items()
            if key != "cve_id" and not _public_key_forbidden(key)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class EventLedger:
    """Thread-safe append-only ledger backed by an in-memory list and a JSONL file."""

    def __init__(self, run_id: str, path: str | None = None) -> None:
        self.run_id = run_id
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._path = path
        self._counter = 0
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
            # Start a fresh ledger file (atomic truncation at construction).
            with open(path, "w"):
                pass

    def _next_id(self) -> str:
        self._counter += 1
        return f"evt-{self._counter:06d}"

    def append(self, event: Event) -> Event:
        if event.outcome and event.outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"Invalid outcome '{event.outcome}'; expected one of {sorted(ALLOWED_OUTCOMES)}")
        if event.failure_class and event.failure_class not in FAILURE_CLASSES:
            raise ValueError(f"Invalid failure_class '{event.failure_class}'")
        with self._lock:
            if not event.event_id:
                event.event_id = self._next_id()
            if not event.run_id:
                event.run_id = self.run_id
            if event.ended_at and event.started_at and not event.duration_ms:
                event.duration_ms = round((event.ended_at - event.started_at) * 1000.0, 3)
            self._events.append(event)
            if self._path:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")
        return event

    def record(
        self,
        *,
        phase: str = "",
        stage: str = "",
        service: str = "",
        cve_id: str = "",
        candidate_id: str = "",
        method: str = "",
        command_id: str = "",
        attempt_id: str = "",
        started_at: float | None = None,
        ended_at: float | None = None,
        outcome: str = "",
        failure_class: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        artifact_ref: str = "",
        proof_ref: str = "",
        scope_decision: str = "",
        policy_decision: str = "",
        detail: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        now = time.time()
        ev = Event(
            phase=phase, stage=stage, service=service, cve_id=cve_id,
            candidate_id=candidate_id, method=method, command_id=command_id,
            attempt_id=attempt_id,
            started_at=started_at if started_at is not None else now,
            ended_at=ended_at if ended_at is not None else now,
            outcome=outcome, failure_class=failure_class,
            tokens_in=tokens_in, tokens_out=tokens_out, cost=cost,
            artifact_ref=artifact_ref, proof_ref=proof_ref,
            scope_decision=scope_decision, policy_decision=policy_decision,
            detail=detail, payload=dict(payload or {}),
        )
        return self.append(ev)

    # ── Read access ────────────────────────────────────────────────────────────
    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events]

    @classmethod
    def load(cls, path: str, run_id: str = "") -> "EventLedger":
        ledger = cls(run_id=run_id or "replay", path=None)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ledger._events.append(Event.from_dict(json.loads(line)))
        return ledger

    @classmethod
    def resume(cls, path: str, run_id: str = "") -> "EventLedger":
        """Open an existing JSONL ledger for append without truncating it."""
        ledger = cls(run_id=run_id or "replay", path=None)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ledger._events.append(Event.from_dict(json.loads(line)))
        ledger._path = path
        ledger._counter = len(ledger._events)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        return ledger

    # ── Repeated-action detection ─────────────────────────────────────────────
    def is_repeated_action(self, candidate_id: str, rendered_params: str) -> bool:
        """A repeated action = same candidate + rendered params attempted again without new evidence."""
        seen = 0
        with self._lock:
            for e in self._events:
                if e.candidate_id == candidate_id and e.payload.get("rendered_params") == rendered_params:
                    seen += 1
                    if seen >= 2:
                        return True
        return False

    def alternate_method_rescue(self, cve_id: str) -> bool:
        """True if an earlier applicable method for cve failed/nonexecutable and a
        distinct later method obtained task proof."""
        first_proof: str | None = None
        prior_failure = False
        prior_methods: set[str] = set()
        with self._lock:
            for e in self._events:
                if e.cve_id != cve_id:
                    continue
                if e.outcome in {"execution_failed", "not_executable"}:
                    prior_failure = True
                    if e.method:
                        prior_methods.add(e.method)
                if e.outcome == "task_proof_obtained":
                    first_proof = e.method
                    break
        if first_proof is None:
            return False
        return prior_failure and first_proof not in prior_methods
