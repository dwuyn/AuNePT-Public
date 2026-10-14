"""
src/pipeline/evaluator.py
─────────────────────────
External evaluator.

Baseline and improved variants share the same ground truth, oracle, budget,
ledger, and scope validation. The evaluator runs a variant runner, then
independently adjudicates every produced proof with the oracle and emits a
traceable ``ResultRow``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.pipeline.budget import ResourceBudget, ResourceLimits
from src.pipeline.ledger import EventLedger
from src.pipeline.manifest import RunManifest
from src.pipeline.oracle import (
    BenchmarkOracle,
    Oracle,
    OracleResult,
    ProofArtifact,
    TargetTruth,
)

# A variant runner executes the pipeline for one target and returns the proof
# artifacts it produced. It is expected to populate the ledger with structured
# events. It must not decide success itself.
VariantRunner = Callable[
    [RunManifest, EventLedger, ResourceBudget, TargetTruth],
    list[ProofArtifact],
]


@dataclass(frozen=True)
class EvaluatorResult:
    """Sealed evaluator output; this is the only source of final success."""
    run_id: str
    proof_accepted: bool
    hidden_truth_match: bool
    terminal_root_cause: str
    recon_match: bool = False
    infrastructure_failure: bool = False
    fixed_control: bool = False
    evaluator_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "proof_accepted": self.proof_accepted,
            "hidden_truth_match": self.hidden_truth_match,
            "terminal_root_cause": self.terminal_root_cause,
            "recon_match": self.recon_match,
            "infrastructure_failure": self.infrastructure_failure,
            "fixed_control": self.fixed_control,
            "evaluator_commit": self.evaluator_commit,
        }


@dataclass
class ResultRow:
    run_id: str
    target_id: str
    variant: str
    condition: str
    repetition: int
    outcome: str                       # primary: task_proof_obtained?
    success_at_1: bool
    vulnerability_confirmed: bool
    oracle_result: dict[str, Any]
    run_dir: str
    repo_commit: str
    model_id: str
    config_hash: str
    tool_versions: dict[str, str]
    source_snapshot_id: str
    candidate_hashes: list[str]
    proof_ref: str
    tokens_in: int
    tokens_out: int
    cost: float
    executed_commands: int
    invalid_commands: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


def _summarize(ledger: EventLedger) -> dict[str, Any]:
    tokens_in = tokens_out = 0
    cost = 0.0
    executed = invalid = 0
    for ev in ledger.events:
        tokens_in += ev.tokens_in
        tokens_out += ev.tokens_out
        cost += ev.cost
        if ev.payload.get("executed_command"):
            executed += 1
        if ev.failure_class == "command_invalid":
            invalid += 1
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": round(cost, 6),
        "executed_commands": executed,
        "invalid_commands": invalid,
    }


class Evaluator:
    """Drives one variant run through independent oracle adjudication."""

    def __init__(self, oracle: Oracle | None = None) -> None:
        self.oracle = oracle or BenchmarkOracle()

    def evaluate(
        self,
        *,
        manifest: RunManifest,
        truth: TargetTruth,
        runner: VariantRunner,
        limits: ResourceLimits | None = None,
    ) -> ResultRow:
        budget = ResourceBudget(limits)
        ledger = EventLedger(manifest.run_id)
        start = time.time()

        proofs = runner(manifest, ledger, budget, truth)
        elapsed = time.time() - start

        oracle_result = self._adjudicate(manifest, truth, proofs)
        success_at_1 = self._first_attempt_succeeded(ledger)
        summary = _summarize(ledger)

        row = ResultRow(
            run_id=manifest.run_id,
            target_id=manifest.target_id,
            variant=manifest.variant,
            condition=manifest.condition,
            repetition=manifest.repetition,
            outcome=oracle_result.outcome,
            success_at_1=success_at_1,
            vulnerability_confirmed=oracle_result.outcome == "vulnerability_confirmed",
            oracle_result=oracle_result.to_dict(),
            run_dir=manifest.run_dir,
            repo_commit=manifest.repo.get("commit", ""),
            model_id=manifest.model_id,
            config_hash=manifest.config_hash,
            tool_versions=dict(manifest.tool_versions),
            source_snapshot_id=(manifest.source_snapshot_ids[0]
                                if manifest.source_snapshot_ids else ""),
            candidate_hashes=list(manifest.artifact_hashes.values()),
            proof_ref=(oracle_result.proof_artifact.content_hash
                       if oracle_result.proof_artifact else ""),
            tokens_in=summary["tokens_in"],
            tokens_out=summary["tokens_out"],
            cost=summary["cost"],
            executed_commands=summary["executed_commands"],
            invalid_commands=summary["invalid_commands"],
            elapsed_seconds=round(elapsed, 3),
        )
        manifest.oracle_result = oracle_result.to_dict()
        return row

    def _adjudicate(self, manifest: RunManifest, truth: TargetTruth,
                    proofs: list[ProofArtifact]) -> OracleResult:
        target_cve = manifest.oracle_spec.get("cve_id", "")
        if target_cve and target_cve in truth.applicable_cves:
            for proof in proofs:
                res = self.oracle.evaluate_proof(target_cve, proof, truth)
                if res.task_proof:
                    return res
            if proofs:
                return self.oracle.evaluate_proof(target_cve, proofs[0], truth)
            return self.oracle.evaluate_proof(target_cve, ProofArtifact(kind="none"), truth)

        for cve in truth.applicable_cves:
            for proof in proofs:
                res = self.oracle.evaluate_proof(cve, proof, truth)
                if res.task_proof:
                    return res
        return OracleResult(outcome="execution_failed",
                            reason="No proof artifacts produced.")

    @staticmethod
    def _first_attempt_succeeded(ledger: EventLedger) -> bool:
        # success_at_1: the first executed candidate yielded task proof with no
        # prior classified failure for an earlier candidate on the same CVE.
        first_candidate: str | None = None
        failed_before_proof = False
        for ev in ledger.events:
            if ev.candidate_id and first_candidate is None:
                first_candidate = ev.candidate_id
            if ev.outcome in {"execution_failed", "not_executable"}:
                failed_before_proof = True
            if ev.outcome == "task_proof_obtained":
                return not failed_before_proof
        return False
