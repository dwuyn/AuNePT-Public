"""Local, resumable coordinator for explicitly approved experiment plans.

The coordinator has no Vertex client.  A caller supplies the cell executor;
this keeps approval verification and retry/cost policy independent from the
framework that performs a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.pipeline.approval import verify_approval


def _canonical_hash(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _cell_worst_case_cost(cell: Mapping[str, Any]) -> float:
    """Read the reservation amount without accepting an unbounded default."""
    for key in (
        "cell_worst_case_cost_usd",
        "cell_worst_case_cost",
        "worst_case_cost_usd",
        "estimated_cost_usd",
    ):
        if key in cell:
            value = float(cell[key])
            if value < 0:
                raise ValueError(f"cell {cell.get('run_id', '')} has a negative worst-case cost")
            return value
    raise ValueError(f"cell {cell.get('run_id', '')} is missing a worst-case cost")


class BillingUnknownError(RuntimeError):
    """An executor could not determine whether a model response was billed."""

    billing_unknown = True
    model_response_received = True


@dataclass(frozen=True)
class CellResult:
    """Normalized executor result used to decide retry eligibility."""

    status: str
    cost_usd: float = 0.0
    billable_model_response: bool = False
    billing_unknown: bool = False
    model_response_received: bool = False
    artifact: Mapping[str, Any] | None = None
    strict_artifact: bool = False
    retryable: bool = False
    failure_id: str = ""


Executor = Callable[[Mapping[str, Any], Path, Mapping[str, str]], CellResult]
CredentialValidator = Callable[[], bool]
GatewayStarter = Callable[[], Any]


def validate_runtime_preflight(
    plan: Mapping[str, Any], *, profile_hashes: Mapping[str, str] | None = None,
    reservation_ceiling_usd: float | None = None,
    credential_validator: CredentialValidator | None = None,
) -> dict[str, Any]:
    """Check every non-provider precondition before a gateway can start."""
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("runtime preflight requires a non-empty plan")
    expected = dict(profile_hashes or {})
    if expected:
        for cell in cells:
            label = str(cell.get("model_label", ""))
            if str(cell.get("model_profile_hash", "")) != str(expected.get(label, "")):
                raise ValueError(f"runtime preflight model-profile hash mismatch for {label}")
    reserved = sum(_cell_worst_case_cost(cell) for cell in cells)
    if reservation_ceiling_usd is not None and reserved > float(reservation_ceiling_usd) + 1e-12:
        raise ValueError("runtime reservation ceiling is below the plan worst-case cost")
    if credential_validator is not None:
        try:
            valid = bool(credential_validator())
        except Exception as exc:
            raise ValueError("ADC/service-account preflight failed") from exc
        if not valid:
            raise ValueError("ADC/service-account preflight failed")
    return {"cell_count": len(cells), "reserved_cost_usd": reserved, "credentials_checked": credential_validator is not None}


class ExperimentRunner:
    """SQLite-backed, at-most-two-worker coordinator for one approved plan."""

    def __init__(self, *, artifact_root: str | Path, workers: int = 2) -> None:
        if workers < 1 or workers > 4:
            raise ValueError("workers must be between 1 and 4")
        self.root = Path(artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(exist_ok=True)
        self.workers = workers
        self.db = self._connect()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.root / "experiment-coordinator.sqlite3",
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def close(self) -> None:
        self.db.close()

    def _init_db(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              cell_json TEXT NOT NULL,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              lease_owner TEXT,
              lease_until REAL,
              artifact_hash TEXT,
              cost_usd REAL NOT NULL DEFAULT 0,
              reserved_cost_usd REAL NOT NULL DEFAULT 0,
              worst_case_cost_usd REAL NOT NULL DEFAULT 0,
              error TEXT
            );
            CREATE TABLE IF NOT EXISTS coordinator_meta (
              key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            """
        )
        columns = {
            str(row["name"])
            for row in self.db.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "reserved_cost_usd" not in columns:
            self.db.execute(
                "ALTER TABLE runs ADD COLUMN reserved_cost_usd REAL NOT NULL DEFAULT 0"
            )
        if "worst_case_cost_usd" not in columns:
            self.db.execute(
                "ALTER TABLE runs ADD COLUMN worst_case_cost_usd REAL NOT NULL DEFAULT 0"
            )
        self.db.commit()

    def register_plan(self, plan: Mapping[str, Any]) -> None:
        cells = plan.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("plan requires a non-empty cells array")
        ids = [str(cell.get("run_id", "")) for cell in cells if isinstance(cell, dict)]
        if len(ids) != len(cells) or not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("plan cells require unique run_id values")
        if any(Path(run_id).name != run_id or run_id in {".", ".."} for run_id in ids):
            raise ValueError("plan run_id values must be safe artifact directory names")
        costs = [_cell_worst_case_cost(cell) for cell in cells]
        expected = plan.get("cell_count")
        if expected is not None and int(expected) != len(cells):
            raise ValueError("plan cell_count does not match cells")
        plan_hash = _canonical_hash(plan)
        previous = self.db.execute(
            "SELECT value FROM coordinator_meta WHERE key='plan_hash'"
        ).fetchone()
        if previous is not None and previous["value"] != plan_hash:
            raise ValueError("coordinator already contains a different plan")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO coordinator_meta(key, value) VALUES('plan_hash', ?)",
                (plan_hash,),
            )
            for cell, worst_case_cost in zip(cells, costs):
                self.db.execute(
                    """INSERT OR IGNORE INTO runs(
                        run_id, cell_json, status, worst_case_cost_usd
                    ) VALUES(?, ?, 'pending', ?)""",
                    (
                        cell["run_id"],
                        json.dumps(cell, sort_keys=True, separators=(",", ":")),
                        worst_case_cost,
                    ),
                )

    @staticmethod
    def _meta(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute(
            "SELECT value FROM coordinator_meta WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def _halt(
        self,
        reason: str,
        detail: str = "",
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        connection = connection or self.db
        payload = json.dumps(
            {"reason": reason, "detail": detail[:2000], "halted_at": time.time()},
            sort_keys=True,
            separators=(",", ":"),
        )
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO coordinator_meta(key, value) VALUES('stage_halt', ?)",
                (payload,),
            )

    def _reclaim_expired_leases(
        self, connection: sqlite3.Connection, now: float | None = None
    ) -> None:
        connection.execute(
            "UPDATE runs SET status='pending', lease_owner=NULL, lease_until=NULL, "
            "reserved_cost_usd=0 WHERE status='running' AND lease_until < ?",
            (time.time() if now is None else now,),
        )

    def _claim(
        self,
        owner: str,
        ceiling: float,
        lease_seconds: float = 900,
        *,
        eligible_run_ids: set[str] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> sqlite3.Row | None:
        """Atomically reserve one cell using a connection owned by one worker."""
        connection = connection or self.db
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._reclaim_expired_leases(connection)
            if self._meta(connection, "stage_halt") is not None:
                connection.commit()
                return None
            spent = float(
                connection.execute(
                    "SELECT COALESCE(SUM(cost_usd + reserved_cost_usd), 0) AS cost FROM runs"
                ).fetchone()["cost"]
            )
            where = "status='pending' AND cost_usd + reserved_cost_usd + worst_case_cost_usd <= ?"
            values: list[Any] = [ceiling]
            if eligible_run_ids is not None:
                if not eligible_run_ids:
                    connection.commit()
                    return None
                where += " AND run_id IN (" + ",".join("?" for _ in eligible_run_ids) + ")"
                values.extend(sorted(eligible_run_ids))
            row = connection.execute(
                "SELECT run_id, cell_json, attempts, worst_case_cost_usd FROM runs WHERE " + where + " ORDER BY run_id LIMIT 1",
                values,
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            worst_case_cost = float(row["worst_case_cost_usd"])
            if spent + worst_case_cost > ceiling + 1e-12:
                connection.commit()
                return None
            updated = connection.execute(
                """UPDATE runs SET status='running', attempts=attempts+1,
                   lease_owner=?, lease_until=?, reserved_cost_usd=?
                   WHERE run_id=? AND status='pending'""",
                (
                    owner,
                    time.time() + lease_seconds,
                    worst_case_cost,
                    row["run_id"],
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT run_id, cell_json, attempts, worst_case_cost_usd "
                "FROM runs WHERE run_id=?",
                (row["run_id"],),
            ).fetchone()
            connection.commit()
            return claimed
        except Exception:
            connection.rollback()
            raise

    def _finish(
        self,
        run_id: str,
        result: CellResult,
        artifact_hash: str,
        error: str = "",
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if result.status not in {
            "completed",
            "infrastructure_failure",
            "failed",
            "billing_unknown",
        }:
            raise ValueError("executor result status is invalid")
        if result.cost_usd < 0:
            raise ValueError("executor result cost cannot be negative")
        connection = connection or self.db
        row = connection.execute(
            "SELECT attempts, cell_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown run id")
        unknown = result.billing_unknown or result.status == "billing_unknown"
        cell = json.loads(str(row["cell_json"]))
        retry_policy = cell.get("retry_policy", {}) if isinstance(cell, Mapping) else {}
        max_attempts = int(retry_policy.get("max_attempts", 2)) if isinstance(retry_policy, Mapping) else 2
        retry = (
            result.status == "infrastructure_failure"
            and not result.billable_model_response
            and not unknown
            and result.retryable
            and row["attempts"] < min(max_attempts, 2)
        )
        status = "pending" if retry else result.status
        with connection:
            connection.execute(
                """UPDATE runs SET status=?, lease_owner=NULL, lease_until=NULL,
                   artifact_hash=?, reserved_cost_usd=0, cost_usd=cost_usd+?, error=?
                   WHERE run_id=?""",
                (status, artifact_hash, float(result.cost_usd), error, run_id),
            )

    def _run_dir(self, stage: str, run_id: str, attempt: int, *, max_attempts: int = 2) -> Path:
        if attempt < 1 or attempt > max_attempts:
            raise ValueError(f"attempt must be between 1 and {max_attempts}")
        parent = self.runs_root / stage / run_id
        parent.mkdir(parents=True, exist_ok=True)
        final = parent / f"attempt-{attempt:02d}"
        if final.exists():
            raise FileExistsError(f"attempt artifact already exists: {final}")
        staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=parent))
        os.replace(staging, final)
        return final

    @staticmethod
    def _exception_has_unknown_billing(exc: BaseException) -> bool:
        if getattr(exc, "cost_usd", None) is not None and getattr(exc, "billing_unknown", None) is False:
            return False
        return bool(
            getattr(exc, "billing_unknown", False)
            or getattr(exc, "model_response_received", False)
            or getattr(exc, "after_model_response", False)
        )

    @staticmethod
    def _approval_expired(expires_at: str) -> bool:
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if expires.tzinfo is None:
            return True
        return datetime.now(UTC) >= expires.astimezone(UTC)

    def _stage_health(
        self,
        *,
        expires_at: str,
        credential_validator: CredentialValidator | None,
        connection: sqlite3.Connection,
    ) -> bool:
        if self._meta(connection, "stage_halt") is not None:
            return False
        if self._approval_expired(expires_at):
            self._halt("approval_expired", connection=connection)
            return False
        if credential_validator is not None:
            try:
                valid = bool(credential_validator())
            except Exception as exc:
                self._halt("credential_invalid", str(exc), connection=connection)
                return False
            if not valid:
                self._halt("credential_invalid", connection=connection)
                return False
        return True

    def _cleanup_attempt(
        self, *, run_id: str, stage: str, attempt_dir: Path
    ) -> dict[str, Any]:
        try:
            result = self.cleanup_labeled_docker_resources(run_id, stage=stage)
        except Exception as exc:
            result = {
                "run_id": run_id,
                "stage": stage,
                "success": False,
                "error": str(exc)[:2000],
            }
        try:
            (attempt_dir / "cleanup.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            result = {**result, "artifact_write_error": str(exc)[:2000]}
        return result

    def _execute_claim(
        self,
        *,
        claimed: sqlite3.Row,
        executor: Executor,
        labels: Mapping[str, str],
        stage: str,
        connection: sqlite3.Connection,
    ) -> None:
        run_id = str(claimed["run_id"])
        cell_for_attempt = json.loads(str(claimed["cell_json"]))
        retry_for_attempt = cell_for_attempt.get("retry_policy", {}) if isinstance(cell_for_attempt, Mapping) else {}
        max_attempts = int(retry_for_attempt.get("max_attempts", 2)) if isinstance(retry_for_attempt, Mapping) else 2
        attempt_dir = self._run_dir(stage, run_id, int(claimed["attempts"]), max_attempts=max_attempts)
        result: CellResult | None = None
        artifact_hash = ""
        error = ""
        try:
            cell = json.loads(claimed["cell_json"])
            result = executor(cell, attempt_dir, labels)
            if not isinstance(result, CellResult):
                raise TypeError("executor must return CellResult")
            if result.artifact is not None:
                termination = str(result.artifact.get("termination_status", ""))
                if termination == "infrastructure_failure" or (
                    result.strict_artifact and termination not in {
                        "completed", "missing_proof", "budget_exhausted", "timeout",
                        "task_timeout", "model_no_visible_text",
                    }
                ):
                    result = CellResult(
                        "infrastructure_failure", cost_usd=result.cost_usd,
                        artifact=result.artifact, strict_artifact=result.strict_artifact,
                    )
                    raise ValueError(
                        "coordinator rejected non-model runtime termination: " + termination
                    )
            marker = {"run_id": run_id, "result": result.__dict__}
            marker_path = attempt_dir / "coordinator-result.json"
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            artifact_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
        except Exception as exc:
            unknown = (
                self._exception_has_unknown_billing(exc)
                or result is not None
                and (
                    result.model_response_received
                    or result.billable_model_response
                    or result.status == "completed"
                )
            )
            known_cost = getattr(exc, "cost_usd", None)
            if known_cost is not None and not unknown:
                result = CellResult(
                    "failed", cost_usd=float(known_cost),
                    billable_model_response=True, model_response_received=True,
                    failure_id=str(getattr(exc, "failure_id", "")),
                )
            else:
                result = CellResult(
                    "billing_unknown" if unknown else "infrastructure_failure",
                    billing_unknown=unknown,
                    model_response_received=unknown,
                    retryable=bool(getattr(exc, "retryable", False)),
                    failure_id=str(getattr(exc, "failure_id", "")),
                )
            error = str(exc)[:2000]
            error_path = attempt_dir / "coordinator-error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id, "error": error, "billing_unknown": unknown,
                        "failure_id": result.failure_id, "retryable": result.retryable,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_hash = hashlib.sha256(error_path.read_bytes()).hexdigest()
        finally:
            self._cleanup_attempt(run_id=run_id, stage=stage, attempt_dir=attempt_dir)
            self._finish(
                run_id,
                result or CellResult("billing_unknown", billing_unknown=True),
                artifact_hash,
                error,
                connection=connection,
            )
            if result is not None and (
                result.billing_unknown or result.status == "billing_unknown"
            ):
                self._halt("billing_unknown", error, connection=connection)

    def _worker_loop(
        self,
        *,
        worker_id: str,
        ceiling: float,
        stage: str,
        expires_at: str,
        credential_validator: CredentialValidator | None,
        executor: Executor,
        eligible_run_ids: set[str] | None = None,
    ) -> None:
        connection = self._connect()
        try:
            while True:
                if not self._stage_health(
                    expires_at=expires_at,
                    credential_validator=credential_validator,
                    connection=connection,
                ):
                    return
                claimed = self._claim(
                    worker_id, ceiling, eligible_run_ids=eligible_run_ids, connection=connection
                )
                if claimed is None:
                    if self._meta(connection, "stage_halt") is not None:
                        return
                    query, values = "SELECT status, COUNT(*) AS count FROM runs", []
                    if eligible_run_ids is not None:
                        query += " WHERE run_id IN (" + ",".join("?" for _ in eligible_run_ids) + ")"
                        values = sorted(eligible_run_ids)
                    query += " GROUP BY status"
                    counts = {
                        str(row["status"]): int(row["count"])
                        for row in connection.execute(query, values).fetchall()
                    }
                    if counts.get("running", 0):
                        time.sleep(0.02)
                        continue
                    if counts.get("pending", 0):
                        self._halt("cost_ceiling_exhausted", connection=connection)
                    return
                labels = {
                    "veriplanpt.run_id": str(claimed["run_id"]),
                    "veriplanpt.stage": stage,
                    "veriplanpt.managed": "true",
                }
                self._execute_claim(
                    claimed=claimed,
                    executor=executor,
                    labels=labels,
                    stage=stage,
                    connection=connection,
                )
        except Exception as exc:
            self._halt("coordinator_error", str(exc), connection=connection)
        finally:
            connection.close()

    def execute(
        self,
        *,
        plan: Mapping[str, Any],
        approval: Mapping[str, Any],
        signature_path: str | Path,
        stage: str,
        public_key: str,
        executor: Executor,
        credential_validator: CredentialValidator | None = None,
        profile_hashes: Mapping[str, str] | None = None,
        gateway_start: GatewayStarter | None = None,
        reservation_ceiling_usd: float | None = None,
        eligible_run_ids: set[str] | None = None,
        approval_scope: str | None = None,
    ) -> dict[str, Any]:
        """Execute cells after validating signed scope/hash/count before claiming."""
        self.register_plan(plan)
        plan_hash = _canonical_hash(plan)
        verified = verify_approval(
            approval,
            scope=approval_scope or stage,
            plan_hash=plan_hash,
            cell_count=len(plan["cells"]),
            cost_ceiling_usd=float(approval.get("cost_ceiling_usd", 0)),
            signature_path=signature_path,
            public_key=public_key,
        )
        validate_runtime_preflight(
            plan,
            profile_hashes=profile_hashes,
            reservation_ceiling_usd=(
                float(verified["cost_ceiling_usd"])
                if reservation_ceiling_usd is None else reservation_ceiling_usd
            ),
            credential_validator=credential_validator,
        )
        all_ids = {str(cell["run_id"]) for cell in plan["cells"]}
        if eligible_run_ids is not None and (not eligible_run_ids or not eligible_run_ids.issubset(all_ids)):
            raise ValueError("eligible_run_ids must be non-empty and belong to the approved plan")
        if gateway_start is not None:
            gateway_start()
        expires_at = str(verified["expires_at"])
        threads = [
            threading.Thread(
                target=self._worker_loop,
                kwargs={
                    "worker_id": f"local-{os.getpid()}-worker-{index:02d}",
                    "ceiling": float(verified["cost_ceiling_usd"]),
                    "stage": stage,
                    "expires_at": expires_at,
                    "credential_validator": credential_validator,
                    "executor": executor,
                    "eligible_run_ids": eligible_run_ids,
                },
                name=f"veriplanpt-worker-{index:02d}",
            )
            for index in range(1, self.workers + 1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.status(eligible_run_ids=eligible_run_ids)

    def status(self, *, eligible_run_ids: set[str] | None = None) -> dict[str, Any]:
        query, values = "SELECT status, COUNT(*) AS count FROM runs", []
        if eligible_run_ids is not None:
            query += " WHERE run_id IN (" + ",".join("?" for _ in eligible_run_ids) + ")"
            values = sorted(eligible_run_ids)
        rows = self.db.execute(query + " GROUP BY status", values).fetchall()
        totals = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS cost, "
            "COALESCE(SUM(reserved_cost_usd), 0) AS reserved FROM runs"
        ).fetchone()
        halted = self._meta(self.db, "stage_halt")
        halt_data = json.loads(halted) if halted else None
        states = {row["status"]: row["count"] for row in rows}
        return {
            "states": states, "selected": sum(states.values()), "completed": int(states.get("completed", 0)),
            "failed": sum(int(states.get(key, 0)) for key in ("failed", "billing_unknown", "infrastructure_failure")),
            "accumulated_cost_usd": float(totals["cost"]),
            "reserved_cost_usd": float(totals["reserved"]),
            "halted": halt_data is not None,
            "halt_reason": halt_data["reason"] if halt_data else "",
        }

    def cleanup_labeled_docker_resources(
        self, run_id: str, *, stage: str = ""
    ) -> dict[str, Any]:
        """Remove only resources owned by one run; never prune globally."""
        label_filters = [f"label=veriplanpt.run_id={run_id}"]
        if stage:
            label_filters.append(f"label=veriplanpt.stage={stage}")
        evidence: dict[str, Any] = {
            "run_id": run_id,
            "stage": stage,
            "resources": {},
            "success": True,
        }
        for kind in ("container", "network"):
            listed = subprocess.run(
                [
                    "docker",
                    kind,
                    "ls",
                    "-q",
                    *sum((["--filter", value] for value in label_filters), []),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            ids = [item for item in listed.stdout.splitlines() if item]
            record: dict[str, Any] = {
                "listed_returncode": listed.returncode,
                "ids": ids,
            }
            if listed.returncode != 0:
                record["stderr"] = listed.stderr[-2000:]
                evidence["success"] = False
            if ids:
                command = ["docker", kind, "rm", *ids]
                if kind == "container":
                    command.insert(3, "-f")
                removed = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                record["remove_returncode"] = removed.returncode
                record["removed_stdout"] = removed.stdout[-2000:]
                record["removed_stderr"] = removed.stderr[-2000:]
                if removed.returncode != 0:
                    evidence["success"] = False
            evidence["resources"][kind] = record
        return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an already approved VeriPlanPT plan locally."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["canary_smoke", "sweep", "confirmation", "benchmark"],
    )
    parser.add_argument("--public-key", required=True)
    args = parser.parse_args(argv)
    # A CLI invocation is intentionally validation-only until an integration
    # supplies the framework executor. This prevents an accidental paid call.
    runner = ExperimentRunner(artifact_root=args.artifact_root)
    try:
        plan, approval = _load_object(args.plan), _load_object(args.approval)
        runner.register_plan(plan)
        verify_approval(
            approval,
            scope=args.stage,
            plan_hash=_canonical_hash(plan),
            cell_count=len(plan["cells"]),
            cost_ceiling_usd=float(approval.get("cost_ceiling_usd", 0)),
            signature_path=args.signature,
            public_key=args.public_key,
        )
        print(json.dumps({"approved": True, **runner.status()}, sort_keys=True))
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
