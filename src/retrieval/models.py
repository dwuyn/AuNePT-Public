"""
Canonical retrieval models and JSON-safe serialization helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProductFingerprint:
    target_ip: str
    port: int
    protocol: str = "tcp"
    raw_service: str = ""
    raw_banner: str = ""
    raw_version: str = ""
    vendor: str = ""
    product: str = ""
    version: str = ""
    cpe_candidates: list[str] = field(default_factory=list)
    platform_hints: list[str] = field(default_factory=list)
    auth_hint: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductFingerprint":
        return cls(**data)


@dataclass
class AuthoritativeRecord:
    cve_id: str
    source: str
    title: str = ""
    description: str = ""
    cvss_score: float = 0.0
    epss_percentile: float = 0.0
    weaknesses: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    affected_ranges: list[dict[str, Any]] = field(default_factory=list)
    platform_hints: list[str] = field(default_factory=list)
    auth_hint: str = "unknown"
    exploit_maturity_hint: str = "unknown"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthoritativeRecord":
        return cls(**data)


@dataclass
class PocCandidate:
    candidate_id: str
    cve_id: str
    source: str
    path: str = ""
    locator: str = ""
    repo_name: str = ""
    entry_files: list[str] = field(default_factory=list)
    language: str = ""
    stars: int = 0
    forks: int = 0
    created_at: str = ""
    has_readme: bool = False
    has_usage: bool = False
    raw_confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PocCandidate":
        return cls(**data)


@dataclass
class ProcedureSnippet:
    candidate_id: str
    commands: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    required_placeholders: list[str] = field(default_factory=list)
    target_assumptions: list[str] = field(default_factory=list)
    usage_notes: list[str] = field(default_factory=list)
    working_directory: str = ""
    setup_commands: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcedureSnippet":
        return cls(**data)


@dataclass
class ApplicabilityAssessment:
    cve_id: str
    candidate_id: str
    version_match: str = "unknown"
    cpe_match: str = "unknown"
    platform_match: str = "unknown"
    auth_match: str = "unknown"
    network_match: str = "unknown"
    procedure_ready: bool = False
    trust_score: float = 0.0
    estimated_cost: float = 0.0
    score: float = 0.0
    verdict: str = "rejected"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplicabilityAssessment":
        return cls(**data)


@dataclass
class RetrievalBundle:
    fingerprints: list[dict[str, Any]] = field(default_factory=list)
    authoritative_records: list[dict[str, Any]] = field(default_factory=list)
    poc_candidates: list[dict[str, Any]] = field(default_factory=list)
    procedure_snippets: list[dict[str, Any]] = field(default_factory=list)
    assessments: list[dict[str, Any]] = field(default_factory=list)
    normalized_evidence: list[dict[str, Any]] = field(default_factory=list)
    shortlist: list[dict[str, Any]] = field(default_factory=list)
    critic_report: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalBundle":
        return cls(**data)
