"""
src/utils/structured_logger.py
────────────────────────────────
Structured JSON logger that emits one JSON object per line (JSONL) to
logs/structured.jsonl.

Every agent node calls slog.node_event() after each LLM call or tool
execution, providing a machine-readable audit trail for post-hoc
metric analysis (M7-M11, M12-M15).

Usage:
    from src.utils.structured_logger import get_structured_logger
    slog = get_structured_logger()
    slog.node_event("recon", step=3, action="nmap ...", tokens_in=200, ...)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = os.environ.get("LOG_DIR", "logs")
_STRUCTURED_LOG = os.path.join(_LOG_DIR, "structured.jsonl")


def _token_count(value: Any) -> int:
    """Narrow untrusted provider metadata to a usable integer token count."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def extract_token_usage(response: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LangChain LLM response.

    Checks multiple response metadata locations to support:
    - OpenAI-compatible (usage_metadata)
    - Anthropic (response_metadata.usage)
    - Ollama (response_metadata.eval_count)
    Falls back to character-count estimate if nothing is found.
    """
    if response is None:
        return 0, 0

    # LangChain usage_metadata (OpenAI, Anthropic via langchain-anthropic)
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        tokens_in = meta.get("input_tokens", meta.get("prompt_tokens", 0))
        tokens_out = meta.get("output_tokens", meta.get("completion_tokens", 0))
        if tokens_in or tokens_out:
            return _token_count(tokens_in), _token_count(tokens_out)

    # response_metadata (Ollama, vLLM, etc.)
    rmeta = getattr(response, "response_metadata", None) or {}
    if isinstance(rmeta, dict):
        usage = rmeta.get("usage", rmeta.get("token_usage", {}))
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            tokens_out = usage.get("completion_tokens", usage.get("output_tokens", 0))
            if tokens_in or tokens_out:
                return _token_count(tokens_in), _token_count(tokens_out)
        # Ollama specific
        if "eval_count" in rmeta:
            return _token_count(rmeta.get("prompt_eval_count", 0)), _token_count(rmeta.get("eval_count", 0))

    # Rough estimate from content length if no metadata
    content = getattr(response, "content", "")
    text = content if isinstance(content, str) else str(content)
    estimated_out = max(1, len(text) // 4)
    return 0, estimated_out


class StructuredLogger:
    """
    Writes JSON events to a JSONL file for offline metric analysis.
    Thread-safe via line-level atomicity of write() on most systems.
    """

    def __init__(self, path: str = _STRUCTURED_LOG) -> None:
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _emit(self, record: dict) -> None:
        record.setdefault("ts", time.time())
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("StructuredLogger write failed: %s", exc)

    def node_event(
        self,
        node: str,
        *,
        step: int = 0,
        phase: str = "",
        action: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        outcome: str = "ok",
        detail: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Emit a node-level execution event."""
        record: dict = {
            "event": "node",
            "node": node,
            "step": step,
            "phase": phase or node,
            "action": action[:120] if action else "",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "duration_ms": round(duration_ms, 1),
            "outcome": outcome,
        }
        if detail:
            record["detail"] = detail[:200]
        record.update(extra)
        self._emit(record)

    def tool_event(
        self,
        node: str,
        tool: str,
        *,
        command: str = "",
        outcome: str = "ok",
        blocked: bool = False,
        duration_ms: float = 0.0,
        step: int = 0,
        **extra: Any,
    ) -> None:
        """Emit a tool-call event."""
        record: dict = {
            "event": "tool",
            "node": node,
            "tool": tool,
            "command": command[:120] if command else "",
            "outcome": "blocked" if blocked else outcome,
            "blocked": blocked,
            "duration_ms": round(duration_ms, 1),
            "step": step,
        }
        record.update(extra)
        self._emit(record)

    def phase_event(self, phase: str, event_type: str, **extra: Any) -> None:
        """Emit a phase-lifecycle event (start, end, block)."""
        record = {"event": "phase", "phase": phase, "type": event_type}
        record.update(extra)
        self._emit(record)

    def run_summary(self, state: dict) -> None:
        """Emit a final run summary with all key metrics from state."""
        record = {
            "event": "run_summary",
            "target_ip": state.get("target_ip", ""),
            "execution_success": state.get("execution_success", False),
            "total_tokens": state.get("total_tokens", 0),
            "total_tokens_in": state.get("total_tokens_in", 0),
            "total_tokens_out": state.get("total_tokens_out", 0),
            "total_llm_requests": state.get("total_llm_requests", 0),
            "total_invalid_commands": state.get("total_invalid_commands", 0),
            "total_repeated_actions": state.get("total_repeated_actions", 0),
            "retry_spent": state.get("retry_spent", 0),
            "recon_steps": state.get("recon_step_count", 0),
            "exec_steps": state.get("execution_step_count", 0),
            "verifier_blocks": state.get("verifier_blocks", 0),
            "debate_rounds": state.get("debate_round", 0),
            "run_start_time": state.get("run_start_time", 0),
            "run_end_time": state.get("run_end_time", 0),
        }
        self._emit(record)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[StructuredLogger] = None


def get_structured_logger(path: str = _STRUCTURED_LOG) -> StructuredLogger:
    """Return the process-wide StructuredLogger singleton."""
    global _instance
    if _instance is None:
        _instance = StructuredLogger(path)
    return _instance
