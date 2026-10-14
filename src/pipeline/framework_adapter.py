"""
src/pipeline/framework_adapter.py
──────────────────────────────────
Public interface for running VeriPlanPT on a benchmark task.

Defines the three locked interfaces:
  - PublicTask  : what the framework receives (blind or guided)
  - BudgetTier  : re-exported from budget.py
  - RunArtifact : what the framework returns

FrameworkAdapter.run() is the single entry point for all benchmark runs.
The external evaluator calls this and receives a RunArtifact; it never reads
the internal state or the evaluator truth directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, cast
from urllib.parse import urlparse

from src.pipeline.budget import BudgetTier
from src.pipeline.ledger import EventLedger

# Re-export BudgetTier so importers only need this module.
__all__ = [
    "BudgetTier",
    "ModelProfile",
    "PublicTask",
    "RunArtifact",
    "FrameworkAdapter",
]

RUN_ARTIFACT_VERSION = "2.1.0"
ADAPTER_CONTRACT_VERSION = "adapter-3.0"
_HEX64 = set("0123456789abcdef")


@dataclass
class ModelProfile:
    """Pinned Vertex profile; credentials stay in the environment.

    Provider aliases are permitted only through the explicit ``provider_alias``
    resolution mode and must carry separately hashed resolution evidence.
    """
    model_name: str
    location: str
    resource_id: str
    resource_revision: str
    pricing: dict[str, float]
    provider: str = "vertexai"
    resolution_mode: str = "immutable"
    resolution_evidence_hash: str = ""
    resolution_resolved_at: str = ""
    endpoint_url: str = ""
    # Compatibility-only runtime input. It is intentionally not serialized;
    # benchmark project selection comes from Vertex environment credentials.
    project: str = field(default="", repr=False)
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    usage_semantics: dict[str, str] = field(default_factory=dict)
    pricing_currency: str = "USD"
    pricing_effective_at: str = ""
    pricing_billing_basis: str = "per_million_tokens"

    ALLOWED_MODELS: ClassVar[frozenset[str]] = frozenset({
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemma-4-26b-a4b-it",
    })
    REQUIRED_PRICING_KEYS: ClassVar[frozenset[str]] = frozenset({
        "input_per_million", "cached_input_per_million", "output_per_million",
    })

    def __post_init__(self) -> None:
        if self.model_name not in self.ALLOWED_MODELS:
            raise ValueError(
                f"ModelProfile.model_name {self.model_name!r} is not in the preregistered set "
                f"{sorted(self.ALLOWED_MODELS)}"
            )
        if self.provider != "vertexai":
            raise ValueError("VeriPlanPT benchmark profiles must use provider='vertexai'")
        missing_fields = [name for name in ("location", "resource_id", "resource_revision")
                          if not str(getattr(self, name, "")).strip()]
        if missing_fields:
            raise ValueError(f"ModelProfile missing required field(s): {', '.join(missing_fields)}")
        if self.resource_id == self.model_name:
            raise ValueError("ModelProfile.resource_id must be the real Vertex resource ID, not the logical label")
        if self.resolution_mode not in {"immutable", "provider_alias"}:
            raise ValueError("ModelProfile.resolution_mode must be immutable or provider_alias")
        if self.resolution_mode == "provider_alias":
            if self.model_name not in {"gemini-3.5-flash", "gemini-3.6-flash"}:
                raise ValueError("provider_alias resolution is allowed only for locked Gemini models")
            if self.resource_revision.lower() != "default":
                raise ValueError("provider_alias profiles must declare resource_revision=default")
            if len(self.resolution_evidence_hash) != 64 or any(
                char not in _HEX64 for char in self.resolution_evidence_hash.lower()
            ):
                raise ValueError("provider_alias profiles require a resolution evidence SHA-256")
            if not self.resolution_resolved_at.strip():
                raise ValueError("provider_alias profiles require resolution_resolved_at")
        elif self.resource_revision.lower() in {"benchmark-pinned", "latest", "default", "unknown"}:
            raise ValueError("ModelProfile.resource_revision must be immutable unless provider_alias is explicit")
        missing_prices = self.REQUIRED_PRICING_KEYS.difference(self.pricing)
        if missing_prices:
            raise ValueError(f"ModelProfile.pricing missing key(s): {', '.join(sorted(missing_prices))}")
        zero_prices = [
            key for key in self.REQUIRED_PRICING_KEYS
            if float(self.pricing.get(key, 0.0)) <= 0.0
        ]
        if zero_prices:
            raise ValueError(f"ModelProfile.pricing must be positive for: {', '.join(sorted(zero_prices))}")
        if self.pricing_currency != "USD":
            raise ValueError("ModelProfile pricing_currency must be USD")
        if not self.pricing_effective_at:
            raise ValueError("ModelProfile requires pricing_effective_at")
        if self.pricing_billing_basis != "per_million_tokens":
            raise ValueError("ModelProfile pricing_billing_basis must be per_million_tokens")
        required_usage = {"input_includes_cached", "total_formula"}
        missing_usage = required_usage.difference(self.usage_semantics)
        if missing_usage:
            raise ValueError(f"ModelProfile usage_semantics missing key(s): {', '.join(sorted(missing_usage))}")
        if self.usage_semantics["input_includes_cached"] != "true":
            raise ValueError("ModelProfile usage must declare input_includes_cached=true")
        total_formula = self.usage_semantics["total_formula"]
        if total_formula not in {"input+output", "input+output+thinking"}:
            raise ValueError("ModelProfile usage total_formula is unsupported")
        if total_formula == "input+output":
            if self.usage_semantics.get("output_includes_reasoning") != "true":
                raise ValueError("ModelProfile input+output usage must include reasoning in output")
        elif float(self.pricing.get("thinking_per_million", 0.0)) <= 0.0:
            raise ValueError("legacy input+output+thinking usage requires thinking_per_million")

    @property
    def logical_label(self) -> str:
        return self.model_name

    @property
    def profile_hash(self) -> str:
        return self.profile_hash_without_self()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelProfile":
        label = data.get("logical_label") or data.get("model_name")
        if not label:
            raise ValueError("ModelProfile requires logical_label or model_name")
        return cls(
            model_name=str(label),
            provider=str(data.get("provider", "vertexai")),
            resolution_mode=str(data.get("resolution_mode", "immutable")),
            resolution_evidence_hash=str(data.get("resolution_evidence_hash", "")),
            resolution_resolved_at=str(data.get("resolution_resolved_at", "")),
            endpoint_url=str(data.get("endpoint_url", "")),
            location=str(data.get("location", "")),
            resource_id=str(data.get("resource_id", "")),
            resource_revision=str(data.get("resource_revision") or data.get("endpoint_id") or ""),
            pricing={str(k): float(v) for k, v in dict(data.get("pricing") or {}).items()},
            generation_parameters=dict(data.get("generation_parameters") or {}),
            usage_semantics={str(k): str(v) for k, v in dict(data.get("usage_semantics") or {}).items()},
            pricing_currency=str(data.get("pricing_currency") or "USD"),
            pricing_effective_at=str(data.get("pricing_effective_at") or ""),
            pricing_billing_basis=str(data.get("pricing_billing_basis") or "per_million_tokens"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "ModelProfile":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_label": self.model_name,
            "model_name": self.model_name,
            "provider": self.provider,
            "resolution_mode": self.resolution_mode,
            "resolution_evidence_hash": self.resolution_evidence_hash,
            "resolution_resolved_at": self.resolution_resolved_at,
            "endpoint_url": self.endpoint_url,
            "resource_id": self.resource_id,
            "resource_revision": self.resource_revision,
            "location": self.location,
            "pricing": dict(self.pricing),
            "generation_parameters": dict(self.generation_parameters),
            "usage_semantics": dict(self.usage_semantics),
            "pricing_currency": self.pricing_currency,
            "pricing_effective_at": self.pricing_effective_at,
            "pricing_billing_basis": self.pricing_billing_basis,
            "profile_hash": self.profile_hash_without_self(),
        }

    def profile_hash_without_self(self) -> str:
        data = {
            "logical_label": self.model_name,
            "provider": self.provider,
            "resolution_mode": self.resolution_mode,
            "resolution_evidence_hash": self.resolution_evidence_hash,
            "resolution_resolved_at": self.resolution_resolved_at,
            "endpoint_url": self.endpoint_url,
            "resource_id": self.resource_id,
            "resource_revision": self.resource_revision,
            "location": self.location,
            "pricing": self.pricing,
            "generation_parameters": self.generation_parameters,
            "usage_semantics": self.usage_semantics,
            "pricing_currency": self.pricing_currency,
            "pricing_effective_at": self.pricing_effective_at,
            "pricing_billing_basis": self.pricing_billing_basis,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class PublicTask:
    """What the framework receives from the benchmark harness.

    PublicTask v2 exposes opaque case identity, track, objective, target,
    scope, and optional guided hints.  Hidden truth, CVE/version metadata,
    alias metadata, and decoy metadata are not part of this contract.

    The harness constructs this from the public half of the lab manifest.
    """
    case_id: str
    track: str               # "blind" | "guided"
    objective: str           # general attack objective, no CVE/version hints
    host: str = ""
    port_range: str = "1-65535"          # e.g. "1-65535" or "80,443,8080"
    target: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)

    FORBIDDEN_PUBLIC_KEYS: ClassVar[frozenset[str]] = frozenset({
        "canonical_case_id",
        "cve",
        "cve_id",
        "hidden_version",
        "version",
        "alias",
        "aliases",
        "alias_metadata",
        "decoy",
        "decoys",
        "decoy_services",
        "decoy_metadata",
    })
    TOP_LEVEL_HINT_KEYS: ClassVar[frozenset[str]] = frozenset({
        "component", "endpoint", "method", "method_hint",
    })
    ALLOWED_HINT_KEYS: ClassVar[frozenset[str]] = frozenset({
        "component", "endpoint", "method",
    })

    def __post_init__(self) -> None:
        if not self.target and self.host:
            self.target = {"host": self.host}
            if self.port_range:
                ports = [
                    int(part) for part in self.port_range.replace(",", " ").split()
                    if part.isdigit()
                ]
                if ports:
                    self.target["exposed_ports"] = ports
        self.validate_public_surface()

    def to_dict(self) -> dict[str, Any]:
        target = dict(self.target)
        if self.host and "host" not in target:
            target["host"] = self.host
        if self.port_range and "exposed_ports" not in target and "port" not in target:
            target["exposed_ports"] = [
                int(part) for part in self.port_range.replace(",", " ").split()
                if part.isdigit()
            ]
        out: dict[str, Any] = {
            "case_id": self.case_id,
            "track": self.track,
            "objective": self.objective,
            "target": target,
            "scope": dict(self.scope),
        }
        if self.track == "guided" and self.hints:
            out["hints"] = dict(self.hints)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PublicTask":
        forbidden = cls.FORBIDDEN_PUBLIC_KEYS.intersection(d)
        if forbidden:
            raise ValueError(f"public task contains forbidden hidden metadata: {', '.join(sorted(forbidden))}")
        top_level_hints = cls.TOP_LEVEL_HINT_KEYS.intersection(d)
        if top_level_hints:
            raise ValueError("guided hints must be nested under the 'hints' object")
        target = cast(dict[str, Any], d.get("target") if isinstance(d.get("target"), dict) else {})
        for key in cls.FORBIDDEN_PUBLIC_KEYS.intersection(target):
            raise ValueError(f"public task target contains forbidden hidden metadata: {key}")
        url = str(target.get("url") or "")
        parsed = urlparse(url)
        host = str(d.get("host") or target.get("host") or parsed.hostname or "")
        ports = d.get("port_range") or target.get("exposed_ports") or target.get("port") or parsed.port
        if isinstance(ports, list):
            port_range = ",".join(str(port) for port in ports)
        else:
            port_range = str(ports or "1-65535")
        scope = cast(dict[str, Any], d.get("scope") if isinstance(d.get("scope"), dict) else {})
        if scope.get("allowed_hosts") and host not in {str(v) for v in scope["allowed_hosts"]}:
            raise ValueError("public task host is outside its declared scope")
        hints = cast(dict[str, Any], d.get("hints") if isinstance(d.get("hints"), dict) else {})
        unknown_hints = set(hints).difference(cls.ALLOWED_HINT_KEYS)
        if unknown_hints:
            raise ValueError(f"unsupported guided hint key(s): {', '.join(sorted(unknown_hints))}")
        task = cls(
            case_id=str(d.get("case_id", "")),
            track=str(d.get("track", "blind")),
            objective=str(d.get("objective", "")),
            host=host,
            port_range=port_range,
            target=dict(target),
            scope=dict(scope),
            hints={str(k): str(v) for k, v in hints.items()},
        )
        task.validate_public_surface()
        return task

    def validate_public_surface(self) -> None:
        """Reject accidental disclosure in a blind task before a run starts."""
        if self.track not in {"blind", "guided"}:
            raise ValueError("public task track must be 'blind' or 'guided'")
        if not self.case_id or not self.objective or not self.host:
            raise ValueError("public task requires case_id, objective and target host")
        if self.track == "blind" and self.hints:
            raise ValueError("blind public task must not include guided hints")
        if self.track == "guided" and set(self.hints).difference(self.ALLOWED_HINT_KEYS):
            raise ValueError("guided task contains unsupported hint keys")
        import re
        visible = json.dumps(self.to_dict(), sort_keys=True, default=str)
        if re.search(r"\bCVE-\d{4}-\d+\b", visible, flags=re.IGNORECASE):
            raise ValueError("public task leaks a CVE identifier")
        if re.search(r"\b(?:version|v)\s*\d+\.\d+(?:\.\d+)?\b", visible, flags=re.IGNORECASE):
            raise ValueError("public task leaks a version")


@dataclass
class EnvironmentManifest:
    """Immutable record of the execution environment."""
    python_version: str = ""
    platform_info: str = ""
    framework_commit: str = ""
    framework_dirty: bool = False
    snapshot_hash: str = ""
    snapshot_cutoff: str = "2026-08-01T00:00:00Z"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "framework_commit": self.framework_commit,
            "framework_dirty": self.framework_dirty,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_cutoff": self.snapshot_cutoff,
            "created_at": self.created_at,
        }


@dataclass
class RunArtifact:
    """What FrameworkAdapter.run() returns to the external harness.

    This is the complete, normalised output of one framework run.
    The harness uploads this to the evaluator; the framework never sees
    the evaluator's verdict.
    """
    case_id: str
    repetition: int
    track: str
    model_profile: ModelProfile
    budget_tier: BudgetTier
    schema_version: str = "2.0.0"
    run_id: str = ""
    run_dir: str = ""
    condition: str = ""
    execution_kind: str = ""
    evaluation_scope: str = ""
    readiness_kind: str = ""
    metric_eligible: bool = True
    termination_status: str = ""
    # Outcome as reported by the internal oracle (without hidden truth).
    internal_outcome: str = ""
    budget_termination_reason: str = ""
    # Normalised transcript (list of {role, event} dicts from EventLedger)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    # Proof submissions for the external evaluator
    proof_submissions: list[dict[str, Any]] = field(default_factory=list)
    # Token/cost usage summary
    usage: dict[str, Any] = field(default_factory=dict)
    # Environment snapshot
    env_manifest: EnvironmentManifest = field(default_factory=EnvironmentManifest)
    # Immutable provenance required for official benchmark gates.  These are
    # mappings rather than framework-specific classes so external adapters can
    # emit the same wire contract without importing VeriPlanPT internals.
    framework_identity: dict[str, Any] = field(default_factory=dict)
    run_context: dict[str, str] = field(default_factory=dict)
    event_ledger_hash: str = ""
    proof_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        termination_status = self.termination_status or self.internal_outcome or "unknown"
        budget_limits = self.budget_tier.to_limits().to_dict()
        return {
            "schema_version": self.schema_version,
            "run_identity": {
                "run_id": self.run_id,
                "case_id": self.case_id,
                "track": self.track,
                "condition": self.condition,
                "execution_kind": self.execution_kind,
                "evaluation_scope": self.evaluation_scope,
                "readiness_kind": self.readiness_kind,
                "repetition": self.repetition,
            },
            "case_id": self.case_id,
            "repetition": self.repetition,
            "track": self.track,
            "condition": self.condition,
            "execution_kind": self.execution_kind,
            "evaluation_scope": self.evaluation_scope,
            "readiness_kind": self.readiness_kind,
            "metric_eligible": self.metric_eligible,
            "model_profile": self.model_profile.to_dict(),
            "model_revision": self.model_profile.resource_revision,
            "budget_tier": self.budget_tier.value,
            "budget": {
                "tier": self.budget_tier.value,
                "limits": budget_limits,
                "usage": dict(self.usage),
            },
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "termination_status": termination_status,
            "internal_outcome": self.internal_outcome,
            "budget_termination_reason": self.budget_termination_reason,
            "transcript": self.transcript,
            "proof_submissions": self.proof_submissions,
            "usage": self.usage,
            "env_manifest": self.env_manifest.to_dict(),
            "framework_identity": dict(self.framework_identity),
            "run_context": dict(self.run_context),
            "event_ledger_hash": self.event_ledger_hash,
            "proof_hash": self.proof_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunArtifact":
        """Read pilot v2.0 artifacts and official v2.1 artifacts.

        Historical pilot artifacts intentionally remain readable, but callers
        running an official gate must invoke :meth:`validate_official`.
        """
        profile_data = data.get("model_profile")
        if not isinstance(profile_data, Mapping):
            raise ValueError("RunArtifact.model_profile is required")
        env_data = data.get("env_manifest")
        env = EnvironmentManifest(**dict(env_data)) if isinstance(env_data, Mapping) else EnvironmentManifest()
        budget_value = str(data.get("budget_tier") or data.get("budget", {}).get("tier", ""))
        return cls(
            case_id=str(data.get("case_id") or data.get("run_identity", {}).get("case_id", "")),
            repetition=int(data.get("repetition") or data.get("run_identity", {}).get("repetition", 0)),
            track=str(data.get("track") or data.get("run_identity", {}).get("track", "")),
            condition=str(data.get("condition") or data.get("run_identity", {}).get("condition", "")),
            execution_kind=str(data.get("execution_kind", "")),
            evaluation_scope=str(data.get("evaluation_scope", "")),
            readiness_kind=str(data.get("readiness_kind", "")),
            metric_eligible=bool(data.get("metric_eligible", True)),
            model_profile=ModelProfile.from_dict(profile_data),
            budget_tier=BudgetTier.from_str(budget_value),
            schema_version=str(data.get("schema_version", "2.0.0")),
            run_id=str(data.get("run_id") or data.get("run_identity", {}).get("run_id", "")),
            run_dir=str(data.get("run_dir", "")),
            termination_status=str(data.get("termination_status", "")),
            internal_outcome=str(data.get("internal_outcome", "")),
            budget_termination_reason=str(data.get("budget_termination_reason", "")),
            transcript=list(data.get("transcript", [])),
            proof_submissions=list(data.get("proof_submissions", [])),
            usage=dict(data.get("usage", {})),
            env_manifest=env,
            framework_identity=dict(data.get("framework_identity", {})),
            run_context={str(k): str(v) for k, v in dict(data.get("run_context", {})).items()},
            event_ledger_hash=str(data.get("event_ledger_hash", "")),
            proof_hash=str(data.get("proof_hash", "")),
        )

    def validate_official(self, *, strict_runtime: bool = False) -> None:
        """Fail closed unless this artifact is a complete official artifact.

        Historical v2.1 artifacts remain readable.  A production runtime call
        opts into the stricter relay-bound contract explicitly.
        """
        if self.schema_version != RUN_ARTIFACT_VERSION:
            raise ValueError("official gates accept only RunArtifact v2.1.0")
        if strict_runtime:
            allowed_outcomes = {
                "completed", "missing_proof", "budget_exhausted", "timeout",
                "task_timeout", "model_no_visible_text",
            }
            if self.termination_status not in allowed_outcomes:
                raise ValueError(
                    "strict runtime rejects non-model termination status: "
                    + (self.termination_status or "missing")
                )
            if self.termination_status == "model_no_visible_text" or self.internal_outcome == "model_no_visible_text":
                if self.internal_outcome != "model_no_visible_text":
                    raise ValueError("no-visible-text artifact must declare model_no_visible_text")
            if self.evaluation_scope == "readiness_transport":
                if self.condition != "not_applicable":
                    raise ValueError("readiness transport artifact condition must be not_applicable")
                if self.execution_kind not in {"vertex_canary", "framework_model_smoke"}:
                    raise ValueError("readiness transport artifact execution_kind is invalid")
                if self.readiness_kind not in {"canary", "smoke"}:
                    raise ValueError("readiness transport artifact readiness_kind is invalid")
                if self.metric_eligible:
                    raise ValueError("readiness transport artifact cannot be metric eligible")
        required_identity = {"name", "repository_url", "commit", "image_digest", "adapter_version"}
        if not required_identity.issubset(self.framework_identity):
            raise ValueError("RunArtifact.framework_identity is incomplete")
        adapter_version = str(self.framework_identity["adapter_version"])
        if strict_runtime and adapter_version not in {ADAPTER_CONTRACT_VERSION, f"{self.framework_identity['name'].lower()}-{ADAPTER_CONTRACT_VERSION}"}:
            raise ValueError("strict runtime requires adapter-3.0")
        if not str(self.framework_identity.get("repository_url", "")).startswith(("https://", "git@")):
            raise ValueError("RunArtifact.framework_identity.repository_url must be an upstream URL")
        if not str(self.framework_identity.get("image_digest", "")).startswith("sha256:"):
            raise ValueError("RunArtifact.framework_identity.image_digest must be immutable")
        required_context = {"dataset_lock_hash", "framework_commit", "evaluator_commit", "stage"}
        if strict_runtime:
            required_context.update({"gateway_relay_lock_hash", "target_runtime_lock_hash"})
            if self.run_context.get("source_snapshot_hash") or self.env_manifest.snapshot_hash or os.environ.get("PENTEST_SOURCE_SNAPSHOT"):
                required_context.add("source_snapshot_hash")
        if not required_context.issubset(self.run_context):
            raise ValueError("RunArtifact.run_context is incomplete")
        for key in (
            "dataset_lock_hash", "training_protocol_hash", "policy_lock_hash", "matrix_hash",
            "gateway_relay_lock_hash", "target_runtime_lock_hash", "source_snapshot_hash",
        ):
            if key == "gateway_relay_lock_hash" and not self.run_context.get(key):
                continue
            if key in self.run_context and not _is_hex64(self.run_context[key]):
                raise ValueError(f"RunArtifact.run_context.{key} must be a SHA-256")
        for name, value in (("event_ledger_hash", self.event_ledger_hash), ("proof_hash", self.proof_hash)):
            if not _is_hex64(value):
                raise ValueError(f"RunArtifact.{name} must be a SHA-256")
        required_usage = {"total_tokens", "total_usd", "input_tokens", "output_tokens"}
        if not required_usage.issubset(self.usage):
            raise ValueError("RunArtifact.usage is not normalized")

    def save(self, path: str) -> None:
        """Write the artifact to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)


def _is_hex64(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text.lower()) <= _HEX64


def _capture_env(snapshot_dir: str = "", *, snapshot_hash: str = "") -> EnvironmentManifest:
    """Capture immutable environment metadata at run time."""
    import subprocess
    commit = ""
    dirty = False
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except Exception:
        pass
    if not commit:
        commit = os.environ.get("VERIPLANPT_FRAMEWORK_COMMIT", "")

    snap_hash = snapshot_hash

    return EnvironmentManifest(
        python_version=sys.version,
        platform_info=platform.platform(),
        framework_commit=commit,
        framework_dirty=dirty,
        snapshot_hash=snap_hash,
        created_at=time.time(),
    )


class FrameworkAdapter:
    """Entry point for all benchmark runs.

    Usage:
        adapter = FrameworkAdapter(results_root="/path/to/results")
        artifact = adapter.run(task, model_profile, budget_tier, repetition=1)
    """

    def __init__(
        self,
        results_root: str = "",
        snapshot_dir: str = "",
    ) -> None:
        self.results_root = results_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "runs",
        )
        self.snapshot_dir = (
            snapshot_dir
            or os.environ.get("PENTEST_SOURCE_SNAPSHOT", "")
            or ("/run/veriplanpt/source-snapshot" if os.path.exists("/run/veriplanpt/source-snapshot") else "")
        )

    def run(
        self,
        task: PublicTask,
        model_profile: ModelProfile,
        budget_tier: BudgetTier,
        repetition: int = 1,
        run_dir: str = "",
        condition: str = "",
        run_id: str = "",
    ) -> RunArtifact:
        """Execute one benchmark run and return the normalised RunArtifact.

        The framework graph is invoked; the external evaluator never accesses
        the internal oracle truth during this call.
        """
        from src.pipeline.candidate_kb import load_candidate_kb
        from src.pipeline.manifest import RunManifest, Scope
        from src.pipeline.source_snapshot import validate_source_snapshot
        from src.state import initial_state

        snapshot_manifest = validate_source_snapshot(
            self.snapshot_dir,
            full=True,
            expected_hash=os.environ.get("PENTEST_SOURCE_SNAPSHOT_HASH", ""),
            official=True,
        )
        # Importing the graph is deliberately after snapshot validation.  The
        # host must reject stale/missing source input before graph setup or any
        # credential-owning client construction can happen.
        from src.graph import build_graph

        env = _capture_env(self.snapshot_dir, snapshot_hash=str(snapshot_manifest["snapshot_hash"]))
        selected_run_dir = run_dir or os.path.join(
            self.results_root, task.case_id,
            f"{model_profile.model_name}_{budget_tier.value}_rep{repetition:02d}",
        )
        os.makedirs(selected_run_dir, exist_ok=True)

        limits = budget_tier.to_limits()
        identity_blob = json.dumps({
            "framework_commit": env.framework_commit,
            "case_id": task.case_id,
            "model_revision": model_profile.resource_revision,
            "budget_tier": budget_tier.value,
            "track": task.track,
            "condition": condition,
            "repetition": repetition,
        }, sort_keys=True)
        thread_id = run_id or "vp-" + hashlib.sha256(identity_blob.encode("utf-8")).hexdigest()[:24]
        graph, config = build_graph(thread_id, source_snapshot_dir=self.snapshot_dir,
                                    source_snapshot_hash=str(snapshot_manifest["snapshot_hash"]))

        # Parse first port from port_range for initial state.
        ports = []
        for part in task.port_range.replace(",", " ").split():
            if "-" in part:
                try:
                    ports.append(int(part.split("-")[0]))
                except ValueError:
                    pass
            else:
                try:
                    ports.append(int(part))
                except ValueError:
                    pass
        target_port = str(ports[0]) if ports else "80"

        state = initial_state(
            target_ip=task.host,
            target_port=target_port,
            max_runtime_seconds=limits.max_runtime_seconds,
        )
        scope = Scope(
            allowed_hostnames=[str(v) for v in task.scope.get("allowed_hosts", [])],
            allowed_ports=[
                int(v) for v in task.scope.get("allowed_ports", [])
                if str(v).isdigit()
            ],
        )
        manifest = RunManifest(
            schema_version="2.0.0",
            run_id=thread_id,
            created_at=time.time(),
            target_id=task.case_id,
            lab_id=task.case_id,
            repetition=repetition,
            condition=condition,
            variant=task.track,
            scope=scope.to_dict(),
            limits=limits.to_dict(),
            repo={
                "commit": env.framework_commit,
                "dirty": env.framework_dirty,
            },
            model_provider=model_profile.provider,
            model_id=model_profile.resource_id,
            run_dir=selected_run_dir,
        )
        state.update({
            "public_task": task.to_dict(),
            "model_profile": model_profile.model_name,
            "budget_tier": budget_tier.value,
            "retrieval_mode": "snapshot",
            "source_snapshot_dir": self.snapshot_dir,
            "source_snapshot_hash": str(snapshot_manifest["snapshot_hash"]),
            "pipeline_manifest": manifest.to_dict(),
            "pipeline_result": {"run_dir": selected_run_dir},
        })
        state["model_profile_contract"] = model_profile.to_dict()  # type: ignore[typeddict-unknown-key]
        if task.hints.get("component"):
            state["app_name"] = task.hints["component"]
        # ``K`` — the candidate knowledge base — is an experiment input, not case
        # truth: the corpus is mounted read-only and a candidate is still admitted
        # only when the product-keyed retrieval returns its CVE.
        state["exploit_candidate_specs"] = load_candidate_kb()

        final = graph.invoke(state, config=config)
        result = dict(final.get("pipeline_result") or {})
        actual_run_dir = str(result.get("run_dir") or run_dir)

        # Build transcript from ledger.
        ledger_path = result.get("ledger_path") or os.path.join(actual_run_dir, "events.jsonl")
        transcript: list[dict[str, Any]] = []
        if ledger_path and os.path.exists(ledger_path):
            ledger = EventLedger.load(ledger_path)
            transcript = [
                {"role": e.phase, "event": e.to_public_dict()}
                for e in ledger.events
            ]

        # Collect proof submissions.
        proofs_path = result.get("proofs_path") or os.path.join(actual_run_dir, "proofs.json")
        proof_submissions: list[dict[str, Any]] = []
        if proofs_path and os.path.exists(proofs_path):
            with open(proofs_path, encoding="utf-8") as fh:
                proof_submissions = json.load(fh)

        # Usage summary from final state.
        usage = {
            "total_tokens": int(final.get("total_tokens") or 0),
            "total_input_tokens": int(final.get("total_input_tokens") or 0),
            "total_cached_input_tokens": int(final.get("total_cached_input_tokens") or 0),
            "total_output_tokens": int(final.get("total_output_tokens") or 0),
            "total_thinking_tokens": int(final.get("total_thinking_tokens") or 0),
            "total_llm_requests": int(final.get("total_llm_requests") or 0),
            "total_usd": float(final.get("total_usd") or 0.0),
        }
        # Also capture from budget_state if available.
        budget_state = dict(final.get("budget_state") or {})
        if budget_state:
            usage.update({
                "budget_tool_calls": budget_state.get("tool_calls", 0),
                "budget_commands": budget_state.get("executed_commands", 0),
                "budget_llm_calls": budget_state.get("llm_calls", 0),
            })

        run_context = {
            "dataset_lock_hash": os.environ.get("VERIPLANPT_DATASET_LOCK_HASH", ""),
            "training_protocol_hash": os.environ.get("VERIPLANPT_TRAINING_PROTOCOL_HASH", ""),
            "gateway_relay_lock_hash": os.environ.get("VERIPLANPT_GATEWAY_RELAY_LOCK_HASH", ""),
            "target_runtime_lock_hash": os.environ.get("VERIPLANPT_TARGET_RUNTIME_LOCK_HASH", ""),
            "framework_commit": env.framework_commit,
            "evaluator_commit": os.environ.get("VERIPLANPT_EVALUATOR_COMMIT", ""),
            "stage": os.environ.get("VERIPLANPT_STAGE", "benchmark"),
            "source_snapshot_hash": str(snapshot_manifest["snapshot_hash"]),
        }
        # Readiness does not have policy or matrix locks yet. Omit optional
        # bindings until the paid sweep/benchmark stages provide real hashes.
        for key in ("policy_lock_hash", "matrix_hash"):
            value = os.environ.get(f"VERIPLANPT_{key.upper()}", "")
            if value:
                run_context[key] = value

        artifact = RunArtifact(
            case_id=task.case_id,
            repetition=repetition,
            track=task.track,
            model_profile=model_profile,
            budget_tier=budget_tier,
            schema_version=RUN_ARTIFACT_VERSION,
            run_id=thread_id,
            run_dir=actual_run_dir,
            condition=condition,
            execution_kind=os.environ.get("VERIPLANPT_EXECUTION_KIND", ""),
            evaluation_scope=os.environ.get("VERIPLANPT_EVALUATION_SCOPE", ""),
            readiness_kind=os.environ.get("VERIPLANPT_READINESS_KIND", ""),
            metric_eligible=os.environ.get("VERIPLANPT_METRIC_ELIGIBLE", "true").lower() == "true",
            termination_status=str(result.get("termination_status") or result.get("outcome") or ""),
            internal_outcome=str(result.get("outcome") or ""),
            budget_termination_reason=str(result.get("budget_termination_reason") or ""),
            transcript=transcript,
            proof_submissions=proof_submissions,
            usage={
                **usage,
                "input_tokens": usage["total_input_tokens"],
                "output_tokens": usage["total_output_tokens"],
                "total_tokens": usage["total_tokens"],
                "total_usd": usage["total_usd"],
            },
            env_manifest=env,
            framework_identity={
                "name": "VeriPlanPT",
                "repository_url": os.environ.get(
                    "VERIPLANPT_REPOSITORY_URL", "https://github.com/openai/veriplanpt"
                ),
                "commit": env.framework_commit,
                "image_digest": os.environ.get("VERIPLANPT_IMAGE_DIGEST", ""),
                "adapter_version": "veriplanpt-adapter-3.0",
            },
            run_context=run_context,
            event_ledger_hash=_json_sha256(transcript),
            proof_hash=_json_sha256(proof_submissions),
        )

        artifact.save(os.path.join(actual_run_dir, "run_artifact.json"))
        return artifact


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
