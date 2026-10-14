"""Budgeted LLM wrapper and normalized usage telemetry."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from src.pipeline.budget import BudgetExceeded, ResourceBudget
from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.ledger import EventLedger


class UsageMetadataMissing(ValueError):
    """A paid model response lacked the provider usage contract."""


@dataclass
class NormalizedUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    model_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "llm_usage",
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 8),
            "latency_ms": round(self.latency_ms, 3),
            "model_revision": self.model_revision,
        }


def _usage_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        for key in ("usage", "usage_metadata", "token_usage"):
            value = response.get(key)
            if isinstance(value, Mapping):
                return value
        return response
    for attr in ("usage_metadata", "usage", "token_usage"):
        value = getattr(response, attr, None)
        if isinstance(value, Mapping):
            return value
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, Mapping):
        value = meta.get("token_usage") or meta.get("usage")
        if isinstance(value, Mapping):
            return value
    return {}


def normalize_usage(response: Any, profile: ModelProfile, latency_ms: float = 0.0) -> NormalizedUsage:
    raw = _usage_mapping(response)
    required_any = (
        "input_tokens", "prompt_tokens", "prompt_token_count", "output_tokens",
        "completion_tokens", "candidates_token_count", "total_tokens",
    )
    if not raw or not any(key in raw for key in required_any):
        raise UsageMetadataMissing("Vertex response is missing usage metadata")
    input_tokens = int(
        raw.get("input_tokens", raw.get("prompt_tokens", raw.get("prompt_token_count", 0))) or 0
    )
    cached_tokens = int(
        raw.get("cached_input_tokens")
        or raw.get("cached_tokens")
        or raw.get("cache_read_input_tokens")
        or raw.get("cached_content_token_count")
        or 0
    )
    output_tokens = int(
        raw.get("output_tokens", raw.get("completion_tokens", raw.get("candidates_token_count", 0))) or 0
    )
    thinking_tokens = int(
        raw.get("thinking_tokens") or raw.get("reasoning_tokens") or raw.get("thoughts_token_count") or 0
    )
    counts = {
        "input": input_tokens, "cached_input": cached_tokens,
        "output": output_tokens, "thinking": thinking_tokens,
    }
    if any(value < 0 for value in counts.values()):
        raise UsageMetadataMissing("Vertex response contains negative token counts")
    if cached_tokens > input_tokens:
        raise UsageMetadataMissing("Vertex cached input tokens exceed input tokens")
    pricing = profile.pricing
    formula = profile.usage_semantics["total_formula"]
    if formula == "input+output":
        output_tokens += thinking_tokens
        total_tokens = input_tokens + output_tokens
        usd = (
            (input_tokens - cached_tokens) * pricing["input_per_million"]
            + cached_tokens * pricing["cached_input_per_million"]
            + output_tokens * pricing["output_per_million"]
        ) / 1_000_000
    else:
        total_tokens = input_tokens + output_tokens + thinking_tokens
        usd = (
            (input_tokens - cached_tokens) * pricing["input_per_million"]
            + cached_tokens * pricing["cached_input_per_million"]
            + output_tokens * pricing["output_per_million"]
            + thinking_tokens * pricing["thinking_per_million"]
        ) / 1_000_000
    reported_total = raw.get("total_tokens")
    if reported_total is not None:
        try:
            reported_total_value = int(reported_total)
        except (TypeError, ValueError) as exc:
            raise UsageMetadataMissing("Vertex total token count is invalid") from exc
        if reported_total_value != total_tokens:
            raise UsageMetadataMissing("Vertex total token count is inconsistent")
    if not math.isfinite(usd) or usd < 0:
        raise UsageMetadataMissing("Vertex response produced an invalid cost")
    return NormalizedUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        usd=usd,
        latency_ms=latency_ms,
        model_revision=profile.resource_revision,
    )


class BudgetedLLM:
    """Single LLM-call authority for budget preflight, cost, and ledger events."""

    def __init__(
        self,
        llm: Any,
        *,
        budget: ResourceBudget,
        ledger: EventLedger,
        model_profile: ModelProfile,
        role: str = "",
    ) -> None:
        self.llm = llm
        self.budget = budget
        self.ledger = ledger
        self.model_profile = model_profile
        self.role = role

    def invoke(self, prompt: Any, *, estimated_tokens: int = 0, **kwargs: Any) -> Any:
        try:
            self.budget.check_llm_call(estimated_tokens=estimated_tokens)
        except BudgetExceeded as exc:
            self.ledger.record(
                phase="lifecycle",
                stage="budget_exhausted",
                outcome="execution_failed",
                failure_class="budget_exceeded",
                detail=str(exc),
                payload={
                    "event_type": "budget_exhausted",
                    "role": self.role,
                    "limit": exc.limit,
                    "used": exc.used,
                    "maximum": exc.maximum,
                    "budget_state": self.budget.state_to_dict(),
                },
            )
            raise

        started = time.time()
        response = self.llm.invoke(prompt, **kwargs) if hasattr(self.llm, "invoke") else self.llm(prompt)
        latency_ms = (time.time() - started) * 1000.0
        usage = normalize_usage(response, self.model_profile, latency_ms=latency_ms)
        try:
            self.budget.record_llm_usage(
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
                thinking_tokens=(
                    usage.thinking_tokens
                    if self.model_profile.usage_semantics["total_formula"] == "input+output+thinking"
                    else 0
                ),
                usd=usage.usd,
            )
        except BudgetExceeded as exc:
            payload = usage.to_dict()
            payload.update({
                "role": self.role,
                "limit": exc.limit,
                "used": exc.used,
                "maximum": exc.maximum,
                "budget_state": self.budget.state_to_dict(),
            })
            self.ledger.record(
                phase="lifecycle",
                stage="budget_exhausted",
                outcome="execution_failed",
                failure_class="budget_exceeded",
                detail=str(exc),
                payload=payload,
            )
            raise
        payload = usage.to_dict()
        payload.update({"role": self.role, "budget_state": self.budget.state_to_dict()})
        self.ledger.record(
            phase="llm",
            stage="llm_usage",
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost=usage.usd,
            payload=payload,
        )
        return response

    def bind_tools(self, tools: Any, **kwargs: Any) -> "BudgetedLLM":
        """Preserve budget enforcement when a LangChain model is tool-bound."""
        if not hasattr(self.llm, "bind_tools"):
            raise TypeError("wrapped model does not support bind_tools")
        return BudgetedLLM(
            self.llm.bind_tools(tools, **kwargs),
            budget=self.budget,
            ledger=self.ledger,
            model_profile=self.model_profile,
            role=self.role,
        )
