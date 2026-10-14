"""
Execution tracker helpers for deterministic-first exploit execution.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

TERMINAL_STATUSES = {"success", "failed", "blocked", "skipped", "preflight_failed"}


def _sanitize_candidate_id(candidate_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id or "candidate")
    return cleaned.strip("._") or "candidate"


def workspace_dir_for_candidate(workspace_root: str, candidate_id: str) -> str:
    return os.path.join(workspace_root, _sanitize_candidate_id(candidate_id))


def build_execution_tracker(state: Mapping[str, Any], execution_cfg: Mapping[str, Any]) -> dict[str, Any]:
    exploit_plan = list(state.get("exploit_plan", []))
    max_candidates = int(execution_cfg.get("max_candidates", 3) or 3)
    selected = exploit_plan[:max_candidates]
    planning_output_dir = state.get("planning_output_dir")
    workspace_root = (
        os.path.join(planning_output_dir, "execution")
        if planning_output_dir else str(execution_cfg.get("workspace_root", "data/execution_runs"))
    )
    os.makedirs(workspace_root, exist_ok=True)

    tracker: dict[str, Any] = {
        "candidate_order": [item.get("candidate_id", "") for item in selected if item.get("candidate_id")],
        "current_candidate_index": 0,
        "current_command_index": 0,
        "llm_fallback_used": False,
        "workspace_root": workspace_root,
        "resolved_placeholders": {},
        "candidate_results": {},
    }
    for item in selected:
        candidate_id = item.get("candidate_id", "")
        if not candidate_id:
            continue
        workspace_dir = workspace_dir_for_candidate(workspace_root, candidate_id)
        tracker["candidate_results"][candidate_id] = {
            "status": "pending",
            "failure_class": "none",
            "attempt_count": 0,
            "workspace_dir": workspace_dir,
            "rendered_commands": [],
            "attempt_log": [],
            "verification_command": "",
            "proof": "",
            "setup_done": False,
            "verify_passed": False,
            "llm_fallback_used": False,
        }
    return tracker


def ensure_candidate_result(tracker: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    results = tracker.setdefault("candidate_results", {})
    if candidate_id not in results:
        results[candidate_id] = {
            "status": "pending",
            "failure_class": "none",
            "attempt_count": 0,
            "workspace_dir": "",
            "rendered_commands": [],
            "attempt_log": [],
            "verification_command": "",
            "proof": "",
            "setup_done": False,
            "verify_passed": False,
            "llm_fallback_used": False,
        }
    return results[candidate_id]


def get_current_candidate_id(tracker: dict[str, Any]) -> str:
    order = list(tracker.get("candidate_order", []))
    index = int(tracker.get("current_candidate_index", 0) or 0)
    if 0 <= index < len(order):
        return order[index]
    return ""


def get_candidate_by_id(exploit_plan: list[dict[str, Any]], candidate_id: str) -> dict[str, Any] | None:
    return next((item for item in exploit_plan if item.get("candidate_id") == candidate_id), None)


def current_candidate(tracker: dict[str, Any], exploit_plan: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    candidate_id = get_current_candidate_id(tracker)
    if not candidate_id:
        return "", None
    return candidate_id, get_candidate_by_id(exploit_plan, candidate_id)


def advance_candidate(tracker: dict[str, Any]) -> None:
    tracker["current_candidate_index"] = int(tracker.get("current_candidate_index", 0) or 0) + 1
    tracker["current_command_index"] = 0
    tracker["resolved_placeholders"] = {}


def append_attempt(
    tracker: dict[str, Any],
    candidate_id: str,
    *,
    stage: str,
    command: str,
    outcome: str,
    output: str = "",
    failure_class: str = "none",
) -> None:
    result = ensure_candidate_result(tracker, candidate_id)
    result.setdefault("attempt_log", []).append({
        "stage": stage,
        "command": command,
        "outcome": outcome,
        "failure_class": failure_class,
        "output": output[:500],
    })


def set_candidate_status(
    tracker: dict[str, Any],
    candidate_id: str,
    *,
    status: str,
    failure_class: str | None = None,
) -> None:
    result = ensure_candidate_result(tracker, candidate_id)
    result["status"] = status
    if failure_class is not None:
        result["failure_class"] = failure_class


def mark_success(
    tracker: dict[str, Any],
    candidate_id: str,
    *,
    verification_command: str,
    proof: str,
) -> None:
    result = ensure_candidate_result(tracker, candidate_id)
    result["status"] = "success"
    result["failure_class"] = "none"
    result["verification_command"] = verification_command.strip()
    result["proof"] = proof[:240]
    result["verify_passed"] = True


def mark_terminal_failure(
    tracker: dict[str, Any],
    candidate_id: str,
    *,
    status: str,
    failure_class: str,
) -> None:
    result = ensure_candidate_result(tracker, candidate_id)
    result["status"] = status
    result["failure_class"] = failure_class
    result["verify_passed"] = False


def tracker_done(tracker: dict[str, Any]) -> bool:
    order = list(tracker.get("candidate_order", []))
    if not order:
        return True
    index = int(tracker.get("current_candidate_index", 0) or 0)
    return index >= len(order)


def write_execution_artifacts(tracker: dict[str, Any]) -> None:
    workspace_root = tracker.get("workspace_root", "")
    if not workspace_root:
        return
    os.makedirs(workspace_root, exist_ok=True)
    tracker_path = os.path.join(workspace_root, "execution_tracker.json")
    results_path = os.path.join(workspace_root, "candidate_results.json")
    try:
        with open(tracker_path, "w", encoding="utf-8") as handle:
            json.dump(tracker, handle, indent=2)
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(tracker.get("candidate_results", {}), handle, indent=2)
    except Exception:
        return
