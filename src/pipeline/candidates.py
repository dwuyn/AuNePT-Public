"""
src/pipeline/candidates.py
──────────────────────────
Generalised candidate interface with eight first-class kinds.

Each candidate has identity, immutable provenance, applicability, structured
procedure, capability, safety, and supporting evidence. ``candidate_id`` is
derived deterministically from canonical identity and provenance so that IDs
stay identical across snapshot / replay runs.

Legacy ``poc_candidates`` are still readable so that old artefacts and tests
remain intact; new writers only produce ``exploit_candidates``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.pipeline.evidence import Fingerprint, VersionConstraint

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_KINDS = (
    "poc", "exploitdb", "metasploit", "nuclei", "nmap_nse",
    "vendor_recipe", "native_tool", "guided_procedure",
)

SUPPORTED_TRUST = ("trusted", "lab_approved", "discovery_only", "llm_provisional", "blocked")

SUPPORTED_CAPABILITIES = (
    "detection", "info_read", "file_write", "auth_bypass",
    "code_execution", "session",
)

# ``setup`` remains readable for snapshots produced before the exploit-skill
# compiler.  New compiled candidates use the five explicit lifecycle stages.
LIFECYCLE_STAGES = ("prepare", "check", "execute", "verify", "cleanup")
LEGACY_STAGE_ALIASES = {"setup": "prepare"}

PLACEHOLDER_RE = re.compile(
    r"(\$\{[A-Za-z_]\w*\})"
    r"|(\{[A-Za-z_][\w]*\})"
    r"|(<[A-Za-z_][\w]*>)"
    r"|(\$\([A-Za-z_]\w*\))"
    r"|(__[A-Z0-9_]+__)"
)


# ── Procedure steps ──────────────────────────────────────────────────────────


@dataclass
class ProcedureStep:
    """A structured, single-stage procedure step.

    ``args`` is a structured argument array — never a free-form shell string.
    The renderer substitutes declared placeholders before scope validation.
    """

    stage: str                       # setup | execute | verify | cleanup
    argv: list[str]
    timeout_seconds: int = 60
    capture_stdout: bool = True
    capture_stderr: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "argv": list(self.argv),
            "timeout_seconds": int(self.timeout_seconds),
            "capture_stdout": bool(self.capture_stdout),
            "capture_stderr": bool(self.capture_stderr),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcedureStep":
        return cls(
            stage=data.get("stage", "execute"),
            argv=list(data.get("argv", []) or []),
            timeout_seconds=int(data.get("timeout_seconds", 60) or 60),
            capture_stdout=bool(data.get("capture_stdout", True)),
            capture_stderr=bool(data.get("capture_stderr", True)),
        )


# ── Provenance ────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    revision: str = ""               # commit, template ID, or schema version
    sha256: str = ""
    retrieved_at: float = 0.0
    references: list[str] = field(default_factory=list)
    license: str = ""                # SPDX or "unknown"
    trust: str = "discovery_only"    # one of SUPPORTED_TRUST
    source_kind: str = ""            # github | exploitdb | metasploit | nuclei | nmap | vendor
    advisory_ref: str = ""           # advisory URL or DB id

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(**{k: data.get(k, getattr(cls(), k)) for k in (
            "revision", "sha256", "retrieved_at", "references", "license",
            "trust", "source_kind", "advisory_ref",
        )})


# ── ExploitCandidate ─────────────────────────────────────────────────────────


@dataclass
class ExploitCandidate:
    candidate_id: str
    cve_id: str
    kind: str                        # one of SUPPORTED_KINDS
    source: str                      # human-readable source name
    locator: str                     # repo URL, template path, module path, etc.

    provenance: Provenance = field(default_factory=Provenance)
    constraint: VersionConstraint = field(default_factory=VersionConstraint)
    platform: str = ""
    auth_required: str = "unknown"   # yes | no | unknown
    network_prereqs: list[str] = field(default_factory=list)
    endpoint_prereqs: list[str] = field(default_factory=list)

    procedure: list[ProcedureStep] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    working_dir: str = ""

    capability: str = "detection"    # one of SUPPORTED_CAPABILITIES
    side_effect_class: str = "read_only"
    requires_callback: bool = False

    product_evidence: str = ""
    version_evidence: str = ""
    cpe_evidence: str = ""

    artifact_hash: str = ""
    # Exploit-skill contract.  These fields intentionally live on the existing
    # candidate instead of introducing a competing skill/candidate model.
    runtime_kind: str = "stateless_process"  # stateless_process|metasploit_rpc|isolated_container
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    requirements: dict[str, list[str]] = field(default_factory=dict)
    expected_evidence: list[str] = field(default_factory=list)
    failure_predicates: list[str] = field(default_factory=list)
    produces_session: bool = False
    repair_lineage: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    #: membership in the closed procedure registry (paper §3.3).  Empty means
    #: the candidate names no template and cannot be compiled for dispatch.
    template_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cve_id": self.cve_id,
            "kind": self.kind,
            "source": self.source,
            "locator": self.locator,
            "provenance": self.provenance.to_dict(),
            "constraint": self.constraint.to_dict(),
            "platform": self.platform,
            "auth_required": self.auth_required,
            "network_prereqs": list(self.network_prereqs),
            "endpoint_prereqs": list(self.endpoint_prereqs),
            "procedure": [s.to_dict() for s in self.procedure],
            "placeholders": list(self.placeholders),
            "working_dir": self.working_dir,
            "capability": self.capability,
            "side_effect_class": self.side_effect_class,
            "requires_callback": bool(self.requires_callback),
            "product_evidence": self.product_evidence,
            "version_evidence": self.version_evidence,
            "cpe_evidence": self.cpe_evidence,
            "artifact_hash": self.artifact_hash,
            "runtime_kind": self.runtime_kind,
            "bindings": dict(self.bindings),
            "requirements": {k: list(v) for k, v in self.requirements.items()},
            "expected_evidence": list(self.expected_evidence),
            "failure_predicates": list(self.failure_predicates),
            "produces_session": bool(self.produces_session),
            "repair_lineage": dict(self.repair_lineage),
            "extra": dict(self.extra),
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExploitCandidate":
        return cls(
            candidate_id=data.get("candidate_id", ""),
            cve_id=data.get("cve_id", ""),
            kind=data.get("kind", ""),
            source=data.get("source", ""),
            locator=data.get("locator", ""),
            provenance=Provenance.from_dict(data.get("provenance") or {}),
            constraint=VersionConstraint.from_dict(data.get("constraint") or {}),
            platform=data.get("platform", ""),
            auth_required=data.get("auth_required", "unknown"),
            network_prereqs=list(data.get("network_prereqs", []) or []),
            endpoint_prereqs=list(data.get("endpoint_prereqs", []) or []),
            procedure=[ProcedureStep.from_dict(p) for p in (data.get("procedure") or [])],
            placeholders=list(data.get("placeholders", []) or []),
            working_dir=data.get("working_dir", ""),
            capability=data.get("capability", "detection"),
            side_effect_class=data.get("side_effect_class", "read_only"),
            requires_callback=bool(data.get("requires_callback", False)),
            product_evidence=data.get("product_evidence", ""),
            version_evidence=data.get("version_evidence", ""),
            cpe_evidence=data.get("cpe_evidence", ""),
            artifact_hash=data.get("artifact_hash", ""),
            runtime_kind=data.get("runtime_kind", "stateless_process"),
            bindings=dict(data.get("bindings", {}) or {}),
            requirements={str(k): list(v or []) for k, v in (data.get("requirements", {}) or {}).items()},
            expected_evidence=list(data.get("expected_evidence", []) or []),
            failure_predicates=list(data.get("failure_predicates", []) or []),
            produces_session=bool(data.get("produces_session", False)),
            repair_lineage=dict(data.get("repair_lineage", {}) or {}),
            extra=dict(data.get("extra", {}) or {}),
            template_id=str(data.get("template_id", "") or ""),
        )


# ── Deterministic ID derivation ──────────────────────────────────────────────


def _canonical(provenance: Provenance, kind: str, cve_id: str, locator: str) -> str:
    """Build a canonical string for ``candidate_id`` derivation."""
    parts = [
        kind.strip().lower(),
        cve_id.strip().upper(),
        locator.strip().lower(),
        provenance.revision.strip().lower(),
        provenance.sha256.strip().lower(),
        provenance.advisory_ref.strip().lower(),
    ]
    return "|".join(parts)


def derive_candidate_id(*, kind: str, cve_id: str, locator: str,
                         provenance: Provenance) -> str:
    """Stable SHA-256-derived candidate id.

    Excludes retrieval time and local paths so two runs over the same source
    produce identical ids.
    """
    canon = _canonical(provenance, kind, cve_id, locator)
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return f"cand-{kind}-{digest[:16]}"


def hash_artifact(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ── Placeholder substitution ─────────────────────────────────────────────────


def substitute_placeholders(argv: Iterable[str], values: Mapping[str, str],
                             *, strict: bool = True) -> tuple[list[str], list[str]]:
    """Substitute placeholders in *argv* using *values*.

    Returns ``(new_argv, unresolved_placeholders)``. When ``strict`` is True,
    any placeholder lacking a value is preserved verbatim and reported as
    unresolved.
    """
    out: list[str] = []
    unresolved: list[str] = []
    for tok in argv:
        new = tok
        for m in PLACEHOLDER_RE.finditer(tok):
            placeholder = m.group(0)
            key = placeholder.strip("${}<>()__")
            val = values.get(key)
            if val is None:
                if strict:
                    unresolved.append(placeholder)
                continue
            new = new.replace(placeholder, val)
        out.append(new)
    return out, unresolved


# ── Trust policy ─────────────────────────────────────────────────────────────


def evaluate_trust(candidate: ExploitCandidate, *,
                    manifest_approved_lab_ids: Iterable[str] | None = None) -> str:
    """Resolve a candidate's final trust state from provenance.

    Only ``trusted`` and ``lab_approved`` (with explicit manifest approval) may
    execute. ``llm_provisional`` is the lowest tier for LLM-generated
    procedures — executable only after explicit verifier approval.
    GitHub stars / repo popularity are *never* a trust signal.
    """
    base = candidate.provenance.trust
    if base == "blocked":
        return "blocked"
    if base == "llm_provisional":
        return "llm_provisional"
    if base == "trusted":
        return "trusted"
    if base == "lab_approved":
        if manifest_approved_lab_ids and candidate.cve_id in manifest_approved_lab_ids:
            return "lab_approved"
        return "blocked"
    if base == "discovery_only":
        return "discovery_only"
    return "blocked"


# ── Legacy reader ─────────────────────────────────────────────────────────────


@dataclass
class LegacyPocCandidate:
    """Minimal view of the legacy ``PocCandidate`` model."""

    cve_id: str
    repo_url: str = ""
    local_path: str = ""
    license: str = ""
    trust_score: float = 0.0
    entry_point: str = ""
    dependencies: list[str] = field(default_factory=list)
    command_template: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "LegacyPocCandidate":
        return cls(
            cve_id=str(getattr(obj, "cve_id", "") or ""),
            repo_url=str(getattr(obj, "repo_url", "") or ""),
            local_path=str(getattr(obj, "local_path", "") or ""),
            license=str(getattr(obj, "license", "") or ""),
            trust_score=float(getattr(obj, "trust_score", 0.0) or 0.0),
            entry_point=str(getattr(obj, "entry_point", "") or ""),
            dependencies=list(getattr(obj, "dependencies", []) or []),
            command_template=str(getattr(obj, "command_template", "") or ""),
        )


def legacy_poc_to_exploit(legacy: LegacyPocCandidate | Any,
                           *, fingerprint: Fingerprint | None = None,
                           ) -> ExploitCandidate:
    """Convert a legacy ``PocCandidate`` into a v1 ``ExploitCandidate``.

    The legacy ``local_path`` becomes a ``poc`` locator; the ``command_template``
    becomes a single ``execute`` step when present. Provenance is preserved.
    """
    if not isinstance(legacy, LegacyPocCandidate):
        legacy = LegacyPocCandidate.from_obj(legacy)
    proc: list[ProcedureStep] = []
    if legacy.command_template:
        proc.append(ProcedureStep(stage="execute", argv=["bash", "-c", legacy.command_template]))
    prov = Provenance(
        revision=legacy.repo_url,
        sha256="",
        references=[legacy.repo_url] if legacy.repo_url else [],
        license=legacy.license or "unknown",
        trust="trusted" if (legacy.license and legacy.license.lower() not in {"unknown", "private", ""}) else "discovery_only",
        source_kind="github",
        advisory_ref=legacy.repo_url,
    )
    cand = ExploitCandidate(
        candidate_id=derive_candidate_id(
            kind="poc", cve_id=legacy.cve_id, locator=legacy.local_path or legacy.repo_url,
            provenance=prov,
        ),
        cve_id=legacy.cve_id,
        kind="poc",
        source="legacy_poc",
        locator=legacy.local_path or legacy.repo_url,
        provenance=prov,
        procedure=proc,
        placeholders=[],
        capability="code_execution",
        side_effect_class="remote_exploit",
        product_evidence=fingerprint.product.parsed if fingerprint else "",
        version_evidence=fingerprint.version.parsed if fingerprint else "",
    )
    return cand


# ── Trust gating ─────────────────────────────────────────────────────────────


def is_executable(candidate: ExploitCandidate, *,
                   manifest_approved_lab_ids: Iterable[str] | None = None,
                   verifier_approved: bool = False,
                   ) -> bool:
    """A candidate is executable iff trust resolves to trusted|lab_approved.

    ``llm_provisional`` candidates are executable only when
    ``verifier_approved`` is True — the verifier must explicitly endorse
    the procedure before it may run.
    """
    state = evaluate_trust(candidate, manifest_approved_lab_ids=manifest_approved_lab_ids)
    if state == "llm_provisional":
        return verifier_approved
    return state in {"trusted", "lab_approved"}


# ── Save/load helpers ────────────────────────────────────────────────────────


def save_candidate(candidate: ExploitCandidate, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as fh:
        json.dump(candidate.to_dict(), fh, sort_keys=True, indent=2)


def load_candidate(path: str) -> ExploitCandidate:
    with open(path) as fh:
        return ExploitCandidate.from_dict(json.load(fh))
