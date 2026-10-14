"""Credential-owning in-process gateway for isolated benchmark containers.

The HTTP serving layer may expose this gateway only after the experiment
coordinator has verified a signed approval.  Baseline containers receive a
short-lived gateway token, never Vertex credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socketserver
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Sequence

from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.runtime_ledger import (
    BillingUnknownError,
    InvocationConflictError,
    InvocationLedger,
)
from src.pipeline.vertex_runtime import (
    LOCKED_MODEL_INVOCATIONS,
    GeminiExecutor,
    GemmaMaaSExecutor,
    GoogleGenAITransport,
    InvocationResult,
    OpenAICompatibleClientTransport,
    PostResponseFailure,
    validate_gemma_endpoint_url,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GatewayError(ValueError):
    """A container request was not authorized for the pinned experiment."""


class ProviderGatewayError(RuntimeError):
    """A provider or post-response failure that must be returned as HTTP 502."""

    def __init__(
        self, message: str, *, failure_id: str = "", upstream_status: int | None = None,
        google_request_id: str = "", error_body_hash: str = "", retryable: bool = False,
        model_response_received: bool = False, billing_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_id = failure_id
        self.upstream_status = upstream_status
        self.google_request_id = google_request_id
        self.error_body_hash = error_body_hash
        self.retryable = retryable
        self.model_response_received = model_response_received
        self.billing_unknown = billing_unknown


def _provider_failure_details(exc: BaseException) -> tuple[int | None, str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status_value = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_value = None
    headers = getattr(response, "headers", {}) if response is not None else {}
    request_id = ""
    if isinstance(headers, Mapping):
        for name in ("x-goog-request-id", "x-request-id", "request-id"):
            if headers.get(name):
                request_id = str(headers[name])
                break
    body = getattr(response, "content", b"") if response is not None else b""
    if isinstance(body, str):
        body = body.encode("utf-8", errors="replace")
    elif not isinstance(body, bytes):
        body = b""
    return status_value, request_id, hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class GatewayRequest:
    run_id: str
    model_label: str
    profile_hash: str
    contents: Any
    epoch: str = ""
    call_index: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GatewayRequest":
        request = cls(
            run_id=str(value.get("run_id", "")),
            model_label=str(value.get("model_label", "")),
            profile_hash=str(value.get("profile_hash", "")),
            contents=value.get("contents", value.get("prompt")),
            epoch=str(value.get("epoch", "")),
            call_index=(int(value["call_index"]) if value.get("call_index") is not None else None),
        )
        if not _RUN_ID.fullmatch(request.run_id):
            raise GatewayError("gateway request run_id is invalid")
        if request.contents is None:
            raise GatewayError("gateway request contents is required")
        if request.call_index is not None and request.call_index < 0:
            raise GatewayError("gateway request call_index is invalid")
        return request


class VertexGateway:
    """Authorize a pinned model request before delegating to an injected executor."""

    def __init__(
        self,
        *,
        profiles: Sequence[ModelProfile],
        allowed_run_ids: set[str],
        token: str,
        gemini: GeminiExecutor,
        gemma: GemmaMaaSExecutor,
        token_expires_at: str = "",
        invocation_ledger: InvocationLedger | None = None,
        max_llm_calls_by_run: Mapping[str, int] | None = None,
        epoch: str = "",
        max_input_tokens_by_run: Mapping[str, int] | None = None,
        max_output_tokens_by_run: Mapping[str, int] | None = None,
        require_signed_plan: bool = False,
    ) -> None:
        if not token:
            raise GatewayError("gateway token is required")
        self.profiles = {profile.logical_label: profile for profile in profiles}
        if len(self.profiles) != len(profiles):
            raise GatewayError("gateway profiles must have unique labels")
        if not allowed_run_ids or any(not _RUN_ID.fullmatch(run_id) for run_id in allowed_run_ids):
            raise GatewayError("gateway requires safe approved run IDs")
        self.allowed_run_ids = set(allowed_run_ids)
        self.token = token
        self.token_expires_at = token_expires_at
        if token_expires_at:
            try:
                expiry = datetime.fromisoformat(token_expires_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise GatewayError("gateway token expiry is invalid") from exc
            if expiry.tzinfo is None:
                raise GatewayError("gateway token expiry must include a timezone")
        self.gemini = gemini
        self.gemma = gemma
        self.invocation_ledger = invocation_ledger
        self.max_llm_calls_by_run = {
            str(run_id): int(limit) for run_id, limit in (max_llm_calls_by_run or {}).items()
        }
        if any(limit <= 0 for limit in self.max_llm_calls_by_run.values()):
            raise GatewayError("gateway call limits must be positive")
        self.epoch = str(epoch)
        self.max_input_tokens_by_run = {
            str(run_id): int(limit) for run_id, limit in (max_input_tokens_by_run or {}).items()
        }
        self.max_output_tokens_by_run = {
            str(run_id): int(limit) for run_id, limit in (max_output_tokens_by_run or {}).items()
        }
        if any(limit <= 0 for limit in (*self.max_input_tokens_by_run.values(), *self.max_output_tokens_by_run.values())):
            raise GatewayError("gateway token caps must be positive")
        self.require_signed_plan = require_signed_plan
        if self.require_signed_plan:
            if not self.epoch:
                raise GatewayError("signed gateway plan requires an epoch")
            expected = self.allowed_run_ids
            if set(self.max_llm_calls_by_run) != expected:
                raise GatewayError("signed gateway plan must cap every allowed run ID")
            if set(self.max_input_tokens_by_run) != expected or set(self.max_output_tokens_by_run) != expected:
                raise GatewayError("signed gateway plan must provide token caps for every allowed run ID")
            if self.invocation_ledger is None:
                raise GatewayError("signed gateway plan requires a durable invocation ledger")
        self._run_locks = {run_id: threading.RLock() for run_id in self.allowed_run_ids}

    def invoke(self, request: Mapping[str, Any], *, token: str) -> InvocationResult:
        self.authorize(token)
        parsed = GatewayRequest.from_dict(request)
        if parsed.run_id not in self.allowed_run_ids:
            raise GatewayError("gateway request run_id is not approved")
        profile = self.profiles.get(parsed.model_label)
        if profile is None or profile.profile_hash != parsed.profile_hash:
            raise GatewayError("gateway request profile is not pinned")
        if self.epoch and parsed.epoch and parsed.epoch != self.epoch:
            raise GatewayError("gateway request epoch differs from the approved epoch")
        if self.require_signed_plan and (parsed.epoch != self.epoch or parsed.call_index is None):
            raise GatewayError("signed gateway request requires the approved epoch and call_index")
        requested_output = request.get("max_output_tokens", request.get("max_tokens"))
        planned_output = self.max_output_tokens_by_run.get(parsed.run_id)
        if self.require_signed_plan and planned_output is None:
            raise GatewayError("signed gateway plan is missing the output cap")
        if requested_output is not None and planned_output is not None and int(requested_output) > planned_output:
            raise GatewayError("gateway request output cap exceeds the signed plan")
        if planned_output is not None and planned_output != 2048:
            raise GatewayError("gateway signed plan must pin max_output_tokens=2048")
        planned_input = self.max_input_tokens_by_run.get(parsed.run_id)
        if self.require_signed_plan and planned_input is None:
            raise GatewayError("signed gateway plan is missing the input cap")
        requested_input = request.get("max_input_tokens")
        if requested_input is not None and planned_input is not None and int(requested_input) > planned_input:
            raise GatewayError("gateway request input cap exceeds the signed plan")
        if planned_input is not None and planned_input != 4096:
            raise GatewayError("gateway signed plan must pin max_input_tokens=4096")
        with self._run_locks[parsed.run_id]:
            return self._invoke_locked(parsed, request, profile)

    def _invoke_locked(
        self, parsed: GatewayRequest, request: Mapping[str, Any], profile: ModelProfile,
    ) -> InvocationResult:
        durable_identity = (
            self.invocation_ledger is not None
            and bool(parsed.epoch or self.epoch)
            and parsed.call_index is not None
        )
        identity_epoch = parsed.epoch or self.epoch
        if durable_identity and self.invocation_ledger is not None:
            try:
                replay = self.invocation_ledger.replay_or_conflict(
                    epoch=identity_epoch, run_id=parsed.run_id,
                    call_index=parsed.call_index or 0, request=dict(request),
                    model_profile_hash=parsed.profile_hash,
                )
            except InvocationConflictError as exc:
                raise GatewayError(str(exc)) from exc
            if replay is not None:
                if replay.get("response") is not None and replay.get("billing_status") == "known":
                    return InvocationResult.from_dict(replay["response"])
                if replay.get("billing_status") == "unknown":
                    raise BillingUnknownError(
                        "durable invocation replay is billing-unknown"
                    )
                raise ProviderGatewayError(
                    "known-billed invocation has no replayable normalized response"
                )
        limit = self.max_llm_calls_by_run.get(parsed.run_id)
        if limit is not None and self.invocation_ledger is not None:
            used = self.invocation_ledger.provider_call_count(
                parsed.run_id, epoch=identity_epoch if durable_identity else "",
            )
            if used >= limit:
                raise GatewayError("gateway max_llm_calls exceeded before provider call")
        if self.invocation_ledger is not None:
            # This is the durable admission point for the provider attempt.
            # It is intentionally before the SDK/transport call so a local
            # validation failure cannot be mistaken for an unused call slot.
            self.invocation_ledger.record_attempt_started(
                run_id=parsed.run_id, model_label=parsed.model_label,
                request=request, model_profile_hash=parsed.profile_hash,
                epoch=identity_epoch if durable_identity else "",
                call_index=parsed.call_index if durable_identity else None,
            )
        try:
            if profile.logical_label.startswith("gemini-"):
                result = self.gemini.invoke(profile, parsed.contents)
            else:
                if isinstance(parsed.contents, (str, bytes)):
                    messages = [{"role": "user", "content": str(parsed.contents)}]
                elif isinstance(parsed.contents, Sequence):
                    messages = [dict(item) for item in parsed.contents if isinstance(item, Mapping)]
                    if len(messages) != len(parsed.contents):
                        raise GatewayError("Gemma gateway messages must be objects")
                else:
                    raise GatewayError("Gemma gateway contents must be chat messages or a prompt string")
                result = self.gemma.invoke(profile, messages)
        except PostResponseFailure as exc:
            self._record_post_response_failure(parsed, request, exc)
            raise AssertionError("post-response failure handler must raise")
        except GatewayError:
            raise
        except Exception as exc:
            status, google_request_id, error_body_hash = _provider_failure_details(exc)
            failure_id = f"failure-{uuid.uuid4().hex}"
            retryable = status in {408, 425, 429, 500, 502, 503, 504}
            if self.invocation_ledger is not None:
                self.invocation_ledger.record_failure(
                    failure_id=failure_id, run_id=parsed.run_id,
                    model_label=parsed.model_label, model_profile_hash=parsed.profile_hash,
                    request=request, upstream_status=status,
                    exception_class=type(exc).__name__, google_request_id=google_request_id,
                    error_body_hash=error_body_hash, retryable=retryable,
                    epoch=identity_epoch if durable_identity else "",
                    call_index=parsed.call_index if durable_identity else None,
                )
            raise ProviderGatewayError(
                "provider request failed before a model response",
                failure_id=failure_id, upstream_status=status,
                google_request_id=google_request_id, error_body_hash=error_body_hash,
                retryable=retryable,
            ) from exc
        if self.invocation_ledger is not None:
            try:
                self.invocation_ledger.record(
                    run_id=parsed.run_id, model_label=parsed.model_label,
                    request=dict(request), response=result.to_dict(),
                    response_hash=result.response_hash,
                    usage=result.usage.to_dict(), cost_usd=result.usage.usd,
                    billing_status="known", outcome="completed",
                    epoch=identity_epoch if durable_identity else "",
                    call_index=parsed.call_index if durable_identity else None,
                    model_profile_hash=parsed.profile_hash if durable_identity else "",
                )
            except Exception as exc:
                # A provider response exists, but without an atomic ledger
                # commit the coordinator cannot safely classify its billing.
                raise BillingUnknownError(
                    "gateway could not durably persist known provider usage"
                ) from exc
        return result

    def authorize(self, token: str) -> None:
        # Bound cell credentials use exactly phase_token~run_id~profile_hash.
        # Keep the unbound phase token for metadata requests, but do not retain
        # the retired dotted-token compatibility path.
        if token != self.token:
            parts = token.split("~")
            if len(parts) != 3 or parts[0] != self.token:
                raise GatewayError("gateway token is invalid")
        if self.token_expires_at:
            expiry = datetime.fromisoformat(self.token_expires_at.replace("Z", "+00:00"))
            if datetime.now(UTC) >= expiry.astimezone(UTC):
                raise GatewayError("gateway token has expired")

    def token_context(self, token: str) -> dict[str, str]:
        """Decode the local OpenAI client bearer binding, never a provider key."""
        self.authorize(token)
        if token == self.token:
            return {}
        parts = token.split("~")
        if len(parts) != 3 or parts[0] != self.token:
            raise GatewayError("gateway bearer binding is invalid")
        run_id, profile_hash = parts[1], parts[2]
        if run_id not in self.allowed_run_ids:
            raise GatewayError("gateway bearer run ID is not approved")
        if profile_hash not in {profile.profile_hash for profile in self.profiles.values()}:
            raise GatewayError("gateway bearer profile is not pinned")
        return {"run_id": run_id, "profile_hash": profile_hash}

    def _record_post_response_failure(
        self, parsed: GatewayRequest, request: Mapping[str, Any], exc: PostResponseFailure,
    ) -> None:
        usage = exc.usage
        if self.invocation_ledger is not None:
            try:
                self.invocation_ledger.record(
                    run_id=parsed.run_id, model_label=parsed.model_label,
                    request=dict(request), response_hash=exc.response_hash,
                    usage=usage.to_dict() if usage is not None else None,
                    cost_usd=usage.usd if usage is not None else None,
                    billing_status="known" if usage is not None else "unknown",
                    outcome="post_response_failure", model_response_received=True,
                    epoch=(parsed.epoch or self.epoch) if parsed.call_index is not None else "",
                    call_index=parsed.call_index,
                    model_profile_hash=parsed.profile_hash if parsed.call_index is not None else "",
                )
            except Exception as persist_exc:
                raise BillingUnknownError(
                    "gateway could not durably persist post-response billing state"
                ) from persist_exc
        failure_id = f"failure-{uuid.uuid4().hex}"
        if usage is None:
            raise BillingUnknownError(
                "provider response received but usage/cost is unknown", failure_id=failure_id,
            ) from exc
        raise ProviderGatewayError(
            "provider response received, but post-response processing failed",
            failure_id=failure_id, model_response_received=True,
        ) from exc


def build_host_gateway(
    *, profiles: Sequence[ModelProfile], allowed_run_ids: set[str], token: str,
    token_expires_at: str, project: str,
    gemini_client_factory: Callable[[str, str], Any],
    gemma_client_factory: Callable[[str], Any],
    invocation_ledger: InvocationLedger | None = None,
    max_llm_calls_by_run: Mapping[str, int] | None = None,
    epoch: str = "",
    max_input_tokens_by_run: Mapping[str, int] | None = None,
    max_output_tokens_by_run: Mapping[str, int] | None = None,
    require_signed_plan: bool = False,
    source_snapshot_root: str = "",
    source_snapshot_hash: str = "",
) -> VertexGateway:
    """Build credential-owning transports from verified host-side metadata.

    Factories are injected so verify-only tests never import an SDK or make a
    request.  The Gemma endpoint is taken from its pinned profile; this
    function has no endpoint constant to drift from Model Garden metadata.
    """
    if not project.strip():
        raise GatewayError("host gateway requires a project")
    if source_snapshot_root:
        from src.pipeline.source_snapshot import validate_source_snapshot
        validate_source_snapshot(
            source_snapshot_root, full=True, expected_hash=source_snapshot_hash, official=True,
        )
    gemini_profiles = [profile for profile in profiles if profile.logical_label.startswith("gemini-")]
    gemma_profiles = [profile for profile in profiles if profile.logical_label == "gemma-4-26b-a4b-it"]
    if len(gemini_profiles) != 2 or len(gemma_profiles) != 1:
        raise GatewayError("host gateway requires exactly two Gemini and one Gemma profile")
    gemma_endpoint = gemma_profiles[0].endpoint_url
    try:
        validate_gemma_endpoint_url(gemma_endpoint)
    except ValueError as exc:
        raise GatewayError(str(exc)) from exc
    gemini_client = gemini_client_factory(project, "global")
    gemma_client = gemma_client_factory(gemma_endpoint)
    return VertexGateway(
        profiles=profiles,
        allowed_run_ids=allowed_run_ids,
        token=token,
        token_expires_at=token_expires_at,
        gemini=GeminiExecutor(GoogleGenAITransport(gemini_client)),
        gemma=GemmaMaaSExecutor(OpenAICompatibleClientTransport(gemma_client)),
        invocation_ledger=invocation_ledger,
        max_llm_calls_by_run=max_llm_calls_by_run,
        epoch=epoch,
        max_input_tokens_by_run=max_input_tokens_by_run,
        max_output_tokens_by_run=max_output_tokens_by_run,
        require_signed_plan=require_signed_plan,
    )


def gateway_handler(gateway: VertexGateway) -> type[BaseHTTPRequestHandler]:
    """Create the internal-only HTTP handler used by the container proxy."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _token(self) -> str:
            auth = self.headers.get("Authorization", "")
            return auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else ""

        def _read_json(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise GatewayError("gateway content length is invalid") from exc
            if length <= 0 or length > 1_048_576:
                raise GatewayError("gateway request size is invalid")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, Mapping):
                raise GatewayError("gateway request must be an object")
            return value

        def _write_json(self, value: Mapping[str, Any], *, status: int = 200) -> None:
            body = json.dumps(dict(value), sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _model_label(model: str, value: Mapping[str, Any]) -> str:
            supplied = str(value.get("model_label", ""))
            if supplied:
                return supplied
            for label, identity in LOCKED_MODEL_INVOCATIONS.items():
                if model in {label, identity["model_id"]}:
                    return label
            profile_hash = str(value.get("profile_hash", ""))
            for label, profile in gateway.profiles.items():
                if profile.profile_hash == profile_hash:
                    return label
            raise GatewayError("chat completion model is not a pinned cell model")

        @staticmethod
        def _openai_result(result: InvocationResult) -> dict[str, Any]:
            usage = result.usage.to_dict()
            return {
                "id": f"chatcmpl-{result.response_hash[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason or "stop",
                }],
                "usage": {
                    "prompt_tokens": int(usage["input_tokens"]),
                    "completion_tokens": int(usage["output_tokens"]),
                    "total_tokens": int(usage["total_tokens"]),
                },
                "response_status": result.response_status,
                "response_hash": result.response_hash,
            }

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/v1/models":
                self.send_error(404)
                return
            try:
                gateway.authorize(self._token())
                self._write_json({
                    "object": "list",
                    "data": [
                        {"id": identity["model_id"], "object": "model", "owned_by": "google"}
                        for label, identity in LOCKED_MODEL_INVOCATIONS.items()
                        if label in gateway.profiles
                    ],
                })
            except GatewayError:
                self.send_error(403)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/v1/generate", "/v1/chat/completions"}:
                self.send_error(404)
                return
            try:
                value = dict(self._read_json())
                token = self._token()
                stream = False
                if self.path == "/v1/generate":
                    binding = gateway.token_context(token)
                    for key, bound in binding.items():
                        value.setdefault(key, bound)
                    if gateway.require_signed_plan:
                        value.setdefault("epoch", gateway.epoch)
                        if "call_index" not in value:
                            used = (
                                gateway.invocation_ledger.provider_call_count(
                                    str(value.get("run_id", "")),
                                    epoch=gateway.epoch,
                                )
                                if gateway.invocation_ledger is not None else 0
                            )
                            value["call_index"] = used
                if self.path == "/v1/chat/completions":
                    binding = gateway.token_context(token)
                    for key, bound in binding.items():
                        value.setdefault(key, bound)
                    if gateway.require_signed_plan:
                        value.setdefault("epoch", gateway.epoch)
                        if "call_index" not in value:
                            used = (
                                gateway.invocation_ledger.provider_call_count(
                                    str(value.get("run_id", "")),
                                    epoch=gateway.epoch,
                                )
                                if gateway.invocation_ledger is not None else 0
                            )
                            value["call_index"] = used
                    messages = value.get("messages")
                    if not isinstance(messages, list):
                        raise GatewayError("chat completions requires messages")
                    value["model_label"] = self._model_label(str(value.get("model", "")), value)
                    value["contents"] = messages
                    stream = value.get("stream") is True
                result = gateway.invoke(value, token=token)
                counters = (
                    gateway.invocation_ledger.counter_snapshot(
                        str(value.get("run_id", "")),
                        epoch=str(value.get("epoch", "")),
                    )
                    if gateway.invocation_ledger is not None else {}
                )
                if self.path == "/v1/chat/completions":
                    response = self._openai_result(result)
                    response.update(counters)
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        delta = {"role": "assistant", "content": result.text}
                        chunk = {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"], "model": result.model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(chunk, sort_keys=True) + "\n\n").encode("utf-8"))
                        final = {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"], "model": result.model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason or "stop"}]}
                        self.wfile.write(("data: " + json.dumps(final, sort_keys=True) + "\n\ndata: [DONE]\n\n").encode("utf-8"))
                        self.wfile.flush()
                    else:
                        self._write_json(response)
                else:
                    response = result.to_dict()
                    response.update(counters)
                    self._write_json(response)
            except (GatewayError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self.send_error(403)
            except BillingUnknownError as exc:
                self._write_json(
                    {
                        "error": "provider response billing is unknown", "failure_id": exc.failure_id,
                        "upstream_status": None, "google_request_id": "", "error_body_sha256": "",
                        "retryable": False, "model_response_received": True, "billing_unknown": True,
                    }, status=502,
                )
            except ProviderGatewayError as exc:
                self._write_json(
                    {
                        "error": "provider request failed", "failure_id": exc.failure_id,
                        "upstream_status": exc.upstream_status,
                        "google_request_id": exc.google_request_id,
                        "error_body_sha256": exc.error_body_hash,
                        "retryable": exc.retryable,
                        "model_response_received": exc.model_response_received,
                        "billing_unknown": exc.billing_unknown,
                    }, status=502,
                )
            except Exception:
                # Provider and post-response failures are deliberately
                # sanitized at the HTTP boundary. Detailed state remains in
                # the host ledger.
                self.send_error(502)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def serve_gateway(gateway: VertexGateway, *, host: str, port: int) -> ThreadingHTTPServer:
    """Construct, but do not start, the internal gateway server."""
    if host not in {"127.0.0.1", "::1", "0.0.0.0"} or port < 1 or port > 65535:
        raise GatewayError("gateway bind address is invalid")
    return ThreadingHTTPServer((host, port), gateway_handler(gateway))


class _ThreadingUnixGatewayServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve_gateway_unix(gateway: VertexGateway, *, socket_path: str) -> socketserver.UnixStreamServer:
    """Bind the host gateway only to a fresh owner-only Unix socket."""
    path = os.path.abspath(socket_path)
    if os.path.lexists(path):
        raise GatewayError("gateway Unix socket path must be new and not a symlink")
    parent = os.path.dirname(path)
    if not os.path.isdir(parent) or os.path.islink(parent):
        raise GatewayError("gateway Unix socket parent directory must be a real directory")
    stat = os.stat(parent)
    if stat.st_uid != os.geteuid() or stat.st_mode & 0o777 != 0o700:
        raise GatewayError("gateway Unix socket parent directory is not owner-only")
    server = _ThreadingUnixGatewayServer(path, gateway_handler(gateway))
    # The coordinator owns the server and uses this explicit handle to persist
    # host-observed usage.  Containers never receive the object or its ledger.
    server.veriplanpt_gateway = gateway  # type: ignore[attr-defined]
    os.chmod(path, 0o600)
    return server
