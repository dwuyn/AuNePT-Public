"""
Regex-first command output classification for deterministic execution.
"""

from __future__ import annotations

import re
from typing import Any

_GENERIC_SUCCESS = (
    "uid=",
    "root@",
    "www-data@",
    "meterpreter >",
    "flag{",
    "htb{",
    "access granted",
    "pwned",
    "session opened",
    "shell obtained",
    "remote code execution",
    "command executed",
)

_GENERIC_SUCCESS_PATTERNS = (
    r"session\s+\d+\s+opened",
    r"reverse shell",
    r"\bwhoami\b.*\n",
)

_PATTERNS = {
    "missing_dependency": [
        r"modulenotfounderror",
        r"no module named",
        r"importerror",
        r"command not found",
        r"is not installed",
        r"cannot import name",
    ],
    "syntax": [
        r"syntaxerror",
        r"unrecognized arguments",
        r"invalid choice",
        r"missing required (?:argument|option)",
        r"the following arguments are required",
        r"usage:",
    ],
    "network": [
        r"connection refused",
        r"connection reset",
        r"failed to connect",
        r"timed out while",
        r"network is unreachable",
        r"name or service not known",
        r"no route to host",
    ],
    "auth": [
        r"permission denied",
        r"authentication failed",
        r"login failed",
        r"unauthorized",
        r"\b401\b",
    ],
    "forbidden": [
        r"\b403\b",
        r"forbidden",
        r"rejected",
    ],
    "timeout": [
        r"^\[timeout\]",
        r"timed out",
    ],
    "not_vulnerable": [
        r"not vulnerable",
        r"target .* patched",
        r"does not seem vulnerable",
        r"patch(ed)?",
    ],
    "invalid_command": [
        r"\[invalid_command\]",
        r"foreign literal IP",
        r"malformed fragment",
    ],
}


# ── Design routing vocabulary (v75, §3.3) ────────────────────────────────────
# The regex classifier below is intentionally fine-grained; the design routes
# recoverable outcomes by class:
#
#   auth_missing     missing session/credential  -> credential-acquisition subgoal
#   version_unknown  version evidence missing    -> version subgoal (verifier tier)
#   payload_rejected target/oversight rejected the payload -> replan
#   transport_error  network/timeout failure     -> retry once
#
# ``version_unknown`` is produced by the verifier (it is an evidence state, not
# a command-output pattern), so it has no entry in this mapping.
DESIGN_ROUTES: dict[str, str] = {
    "auth": "auth_missing",
    "forbidden": "payload_rejected",
    "network": "transport_error",
    "timeout": "transport_error",
    "not_vulnerable": "payload_rejected",
    "invalid_command": "payload_rejected",
    "syntax": "payload_rejected",
    "missing_dependency": "payload_rejected",
    "missing_placeholder": "payload_rejected",
    "policy_block": "payload_rejected",
    "procedure_incomplete": "payload_rejected",
}

# v75 retries ``transport_error`` once and routes ``auth_missing`` to the
# credential branch instead of replaying the same command indefinitely.
DESIGN_RETRYABLE: frozenset[str] = frozenset({"transport_error", "auth_missing"})

#: routing decisions the orchestration loop acts on (Algorithm 2, step 27).
ROUTE_RETRY_ONCE = "retry_once"
ROUTE_SUBGOAL = "requeue_subgoal"
ROUTE_REPLAN = "replan"
ROUTE_STOP = "stop"

#: the four recoverable outcome classes of paper §3.3.
OUTCOME_CLASSES: tuple[str, ...] = ("auth_missing", "version_unknown", "payload_rejected", "transport_error")


def route_outcome(outcome_class: str, *, attempts: int = 0) -> str:
    """Map an outcome class onto the loop's routing decision (paper §3.3).

    ``transport_error`` is retried once; ``auth_missing`` and
    ``version_unknown`` become evidence-acquisition subgoals;
    ``payload_rejected`` returns the candidate for replanning with an increased
    penalty.
    """
    cls = str(outcome_class or "")
    if cls == "transport_error":
        return ROUTE_RETRY_ONCE if int(attempts) <= 1 else ROUTE_STOP
    if cls in {"auth_missing", "version_unknown"}:
        return ROUTE_SUBGOAL
    if cls == "payload_rejected":
        return ROUTE_REPLAN
    return ROUTE_STOP


def classify_outcome(outcome: Any, world_state: Any | None = None) -> str:
    """``ClassifyOutcome(res, M_W)`` — the paper's routing class for one dispatch.

    Returns one of :data:`OUTCOME_CLASSES`, or ``"none"`` for a dispatch that
    completed without a recoverable failure.
    """
    fine = str(getattr(outcome, "failure_class", "") or "")
    routed = design_route(fine)
    if routed in OUTCOME_CLASSES:
        return routed
    if fine:
        return "payload_rejected"
    if int(getattr(outcome, "exit_code", 0) or 0) != 0:
        return "payload_rejected"
    return "none"


def design_route(failure_class: str) -> str:
    """Map a fine-grained failure class onto the design routing vocabulary."""
    return DESIGN_ROUTES.get(str(failure_class or "unknown"), "unrouted")


def classify_output(
    output: str,
    *,
    success_indicators: list[str] | None = None,
    failure_indicators: list[str] | None = None,
) -> dict[str, Any]:
    text = str(output or "")
    lower = text.lower()
    if lower.startswith("[blocked]"):
        return {"status": "blocked", "failure_class": "unknown", "matched": "[BLOCKED]"}
    for item in success_indicators or []:
        if item and item.lower() in lower:
            return {"status": "success", "failure_class": "none", "matched": item}
    for item in _GENERIC_SUCCESS:
        if item in lower:
            return {"status": "success", "failure_class": "none", "matched": item}
    for pattern in _GENERIC_SUCCESS_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE | re.MULTILINE):
            return {"status": "success", "failure_class": "none", "matched": pattern}
    for item in failure_indicators or []:
        if item and item.lower() in lower:
            return {"status": "failed", "failure_class": "unknown", "matched": item}
    for failure_class, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower, flags=re.IGNORECASE | re.MULTILINE):
                return {"status": "failed", "failure_class": failure_class, "matched": pattern}
    if lower.startswith("[error]"):
        return {"status": "failed", "failure_class": "unknown", "matched": "[ERROR]"}
    return {"status": "uncertain", "failure_class": "unknown", "matched": ""}
