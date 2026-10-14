"""Small lifecycle-aware runtimes for compiled exploit candidates."""

from __future__ import annotations

import ast
import os
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.pipeline.budget import BudgetExceeded, ResourceBudget
from src.pipeline.candidates import ExploitCandidate
from src.pipeline.ledger import EventLedger
from src.pipeline.manifest import Scope
from src.pipeline.runner import ExecutionResult
from src.pipeline.scope import ScopeValidator


@dataclass
class SessionArtifact:
    runtime: str
    session_id: str
    target: str
    source_candidate_id: str
    capability: str
    opened_at: float
    last_verified_at: float = 0.0
    verification_evidence: str = ""
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeResult:
    result: ExecutionResult
    failure_class: str = ""
    evidence_kind: str = ""
    session: SessionArtifact | None = None


class ExecutionGateway:
    """Only command path available to benchmark execution.

    It enforces structured argv, scope and budget before giving an isolated
    attacker runtime the command.  No benchmark caller may fall back to a host
    shell when the runtime is absent or rejects a command.
    """

    def __init__(self, *, runtime: "IsolatedContainerRuntime", scope: Scope,
                 budget: ResourceBudget, ledger: EventLedger) -> None:
        self.runtime = runtime
        self.validator = ScopeValidator(scope)
        self.budget = budget
        self.ledger = ledger

    def execute(self, argv: list[str], *, timeout: int, stage: str, candidate_id: str = "",
                cve_id: str = "") -> RuntimeResult:
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            self.ledger.record(phase="execution", stage="command", candidate_id=candidate_id, cve_id=cve_id,
                               failure_class="command_invalid",
                               payload={"event_type": "command", "validator_rejected": True, "argv": argv})
            return RuntimeResult(ExecutionResult(2, "", "invalid structured argv", 0), "command_invalid")
        decision = self.validator.validate_args(argv, stage=stage)
        if not decision:
            self.ledger.record(phase="execution", stage="command", candidate_id=candidate_id, cve_id=cve_id,
                               failure_class="scope_violation", scope_decision="blocked",
                               payload={"event_type": "command", "validator_rejected": True, "argv": argv,
                                        "reason": decision.reason})
            return RuntimeResult(ExecutionResult(2, "", decision.reason, 0), "scope_violation")
        try:
            self.budget.record_tool_call()
            self.budget.record_command()
        except BudgetExceeded as exc:
            self.ledger.record(phase="lifecycle", stage="budget_exhausted", candidate_id=candidate_id,
                               cve_id=cve_id, outcome="execution_failed", failure_class="budget_exceeded",
                               payload={"event_type": "budget_exhausted", "reason": str(exc),
                                        "budget_state": self.budget.state_to_dict()})
            return RuntimeResult(ExecutionResult(124, "", str(exc), 0), "budget_exceeded")
        self.ledger.record(phase="execution", stage="command", candidate_id=candidate_id, cve_id=cve_id,
                           scope_decision="allowed", policy_decision="execute",
                           payload={"event_type": "command", "argv": argv, "stage": stage})
        return self.runtime.run(argv, timeout=timeout)


def static_preflight(path: str, language: str) -> str:
    """Return an empty string when a copied/generated artifact is safe to prepare."""
    if not os.path.isfile(path):
        return "artifact_missing"
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return "artifact_missing"
    if language == "python":
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            return "syntax_invalid"
        blocked = {"system", "popen", "run", "call", "Popen"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in blocked:
                return "scope_violation"
        return ""
    if language == "shell":
        checked = subprocess.run(["bash", "-n", path], check=False, capture_output=True, text=True)
        return "" if checked.returncode == 0 else "syntax_invalid"
    return ""


class IsolatedContainerRuntime:
    """Builds one disposable, network-scoped Docker invocation per step."""

    def __init__(self, *, image: str, network: str, run_dir: str, scope: Scope,
                 docker: str = "docker", execute: Callable[..., Any] = subprocess.run) -> None:
        self.image, self.network, self.run_dir, self.scope = image, network, run_dir, scope
        self.docker, self.execute = docker, execute

    def argv(self, command: list[str]) -> list[str]:
        if not self.network or self.network == "host":
            raise ValueError("scope_violation: lab network is required")
        workspace = os.path.join(self.run_dir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        return [self.docker, "run", "--rm", "--network", self.network, "--read-only",
                "--tmpfs", "/tmp:rw,nosuid,nodev", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "128",
                "--memory", "512m", "--cpus", "1", "--ulimit", "nofile=1024:1024",
                "--user", "65534:65534",
                "--mount", f"type=bind,src={self.run_dir},dst={self.run_dir},readonly",
                "--mount", f"type=bind,src={workspace},dst=/work",
                "-w", "/work", self.image, *command]

    def run(self, command: list[str], *, timeout: int) -> RuntimeResult:
        start = time.time()
        try:
            proc = self.execute(self.argv(command), check=False, capture_output=True, timeout=timeout)
            # Tool output is raw bytes: decode leniently so a binary body is
            # classified instead of raising inside subprocess.
            return RuntimeResult(ExecutionResult(proc.returncode,
                                                 proc.stdout.decode("utf-8", "replace"),
                                                 proc.stderr.decode("utf-8", "replace"),
                                                 round((time.time() - start) * 1000, 3)))
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return RuntimeResult(ExecutionResult(124, stdout, stderr, timeout * 1000), "timeout")
        except (OSError, ValueError) as exc:
            return RuntimeResult(ExecutionResult(1, "", str(exc), 0), "runtime_error")


class MetasploitRuntime:
    """Protocol-shaped runtime; an RPC client supplies check/jobs/sessions."""

    def __init__(self, client: Any, *, target: str, candidate: ExploitCandidate) -> None:
        self.client, self.target, self.candidate = client, target, candidate

    def check(self, options: dict[str, str]) -> RuntimeResult:
        answer = self.client.check(self.candidate.locator, options)
        ok = bool(answer.get("vulnerable") or answer.get("code") == "vulnerable")
        return RuntimeResult(ExecutionResult(0 if ok else 1, str(answer), "", 0),
                             "" if ok else "negative_check", "vulnerability_confirmation" if ok else "")

    def execute(self, options: dict[str, str]) -> RuntimeResult:
        answer = self.client.execute(self.candidate.locator, options)
        job_id = str(answer.get("job_id") or "")
        if not job_id:
            return RuntimeResult(ExecutionResult(1, str(answer), "", 0), "job_failed")
        session = self.client.wait_for_session(job_id)
        if not session:
            return RuntimeResult(ExecutionResult(1, str(answer), "", 0), "session_not_created")
        artifact = SessionArtifact("metasploit_rpc", str(session.get("id", "")), self.target,
                                  self.candidate.candidate_id, self.candidate.capability, time.time())
        return RuntimeResult(ExecutionResult(0, str(answer), "", 0), "", "session_created", artifact)

    def cleanup(self, session: SessionArtifact | None) -> None:
        if session:
            self.client.stop_session(session.session_id)
            session.status = "closed"


def rpc_secret() -> str:
    """Per-run RPC password; callers write it only to an ephemeral 0600 file."""
    return secrets.token_urlsafe(32)
