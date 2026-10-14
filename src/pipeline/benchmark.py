"""
src/pipeline/benchmark.py
─────────────────────────
Benchmark manifests and the metric/reporting layer.

Metrics are computed solely from the event ledger + benchmark truth; never from
agent/executor state. The preregistered paired endpoint is success within three
repetitions for each target-condition pair. Wilson 95% intervals are reported
for success and false-positive proportions.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.pipeline.evaluator import ResultRow
from src.pipeline.ledger import ALLOWED_OUTCOMES, EventLedger
from src.pipeline.oracle import ProofSpec, TargetTruth

VARIANT_SETTINGS = {
    "1": {"automatic_exploit_compilation": False, "automatic_metasploit_discovery": False,
          "allow_llm_fallback": False},
    "2": {"automatic_exploit_compilation": False, "automatic_metasploit_discovery": False,
          "allow_llm_fallback": False},
    "3": {"automatic_exploit_compilation": True, "automatic_metasploit_discovery": True,
          "allow_llm_fallback": False},
    "4": {"automatic_exploit_compilation": True, "automatic_metasploit_discovery": True,
          "allow_llm_fallback": True},
}


def variant_settings(variant: str) -> dict[str, bool]:
    """Controls for the four preregistered paired evaluation variants."""
    return dict(VARIANT_SETTINGS.get(str(variant), VARIANT_SETTINGS["2"]))

# ── Benchmark manifests ──────────────────────────────────────────────────────


@dataclass
class BenchmarkTarget:
    target_id: str
    image: str
    image_hash: str = ""
    source_available_at: float = 0.0
    applicable_cves: list[str] = field(default_factory=list)
    version_constraints: dict[str, Any] = field(default_factory=dict)
    proof_specs: dict[str, ProofSpec] = field(default_factory=dict)
    is_patched_control: bool = False
    cleanup_verifier: str = ""
    network_services: list[str] = field(default_factory=list)
    noisy_mirrors: list[str] = field(default_factory=list)

    def to_truth(self) -> TargetTruth:
        return TargetTruth(
            target_id=self.target_id,
            applicable_cves=list(self.applicable_cves),
            version_constraints=dict(self.version_constraints),
            proof_specs=dict(self.proof_specs),
            is_patched_control=self.is_patched_control,
            cleanup_verifier=self.cleanup_verifier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "image": self.image,
            "image_hash": self.image_hash,
            "source_available_at": self.source_available_at,
            "applicable_cves": list(self.applicable_cves),
            "version_constraints": dict(self.version_constraints),
            "proof_specs": {k: v.to_dict() for k, v in self.proof_specs.items()},
            "is_patched_control": self.is_patched_control,
            "cleanup_verifier": self.cleanup_verifier,
            "network_services": list(self.network_services),
            "noisy_mirrors": list(self.noisy_mirrors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkTarget":
        specs = {
            cve: ProofSpec(**spec) if isinstance(spec, dict) else spec
            for cve, spec in (data.get("proof_specs") or {}).items()
        }
        return cls(
            target_id=data.get("target_id", ""),
            image=data.get("image", ""),
            image_hash=data.get("image_hash", ""),
            source_available_at=float(data.get("source_available_at", 0.0) or 0.0),
            applicable_cves=list(data.get("applicable_cves", []) or []),
            version_constraints=dict(data.get("version_constraints") or {}),
            proof_specs=specs,
            is_patched_control=bool(data.get("is_patched_control", False)),
            cleanup_verifier=data.get("cleanup_verifier", ""),
            network_services=list(data.get("network_services", []) or []),
            noisy_mirrors=list(data.get("noisy_mirrors", []) or []),
        )


@dataclass
class BenchmarkManifest:
    name: str
    schema_version: str = "1.0.0"
    targets: list[BenchmarkTarget] = field(default_factory=list)
    legacy_targets: int = 0
    recent_targets: int = 0
    patched_controls: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "targets": [t.to_dict() for t in self.targets],
            "legacy_targets": self.legacy_targets,
            "recent_targets": self.recent_targets,
            "patched_controls": self.patched_controls,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkManifest":
        return cls(
            name=data.get("name", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            targets=[BenchmarkTarget.from_dict(t) for t in (data.get("targets") or [])],
            legacy_targets=int(data.get("legacy_targets", 0) or 0),
            recent_targets=int(data.get("recent_targets", 0) or 0),
            patched_controls=int(data.get("patched_controls", 0) or 0),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str) -> "BenchmarkManifest":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, indent=2)

    def condition_pairs(self, *, conditions: Iterable[str] = ("clean", "noisy"),
                          repetitions: int = 3) -> list[tuple[str, str, int]]:
        out: list[tuple[str, str, int]] = []
        for tgt in self.targets:
            for cond in conditions:
                if cond == "noisy" and not tgt.noisy_mirrors:
                    continue
                for rep in range(1, repetitions + 1):
                    out.append((tgt.target_id, cond, rep))
        return out


# ── Metric computation (events-only) ──────────────────────────────────────────


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def metrics_from_ledger(ledger: EventLedger, *, truth: TargetTruth) -> dict[str, Any]:
    """Compute the per-run metric block from the ledger.

    Source-of-truth: events only. Never reads agent/executor state.
    """
    events = ledger.events
    outcomes = [ev.outcome for ev in events]
    proof = [ev for ev in events if ev.outcome == "task_proof_obtained"]
    confirmed = [ev for ev in events if ev.outcome == "vulnerability_confirmed"]
    failures = [ev for ev in events if ev.outcome in {"execution_failed", "not_executable", "blocked_by_policy"}]
    executed = [ev for ev in events if ev.payload.get("executed_command")]
    invalid_cmds = [ev for ev in events if ev.failure_class == "command_invalid"]
    rescue = ledger.alternate_method_rescue(truth.applicable_cves[0]) if truth.applicable_cves else False

    tokens_in = sum(ev.tokens_in for ev in events)
    tokens_out = sum(ev.tokens_out for ev in events)
    cost = sum(ev.cost for ev in events)

    # Success metrics.
    task_proof = bool(proof)
    vulnerability_confirmed = bool(confirmed)
    primary_success = task_proof and (not truth.is_patched_control)
    success_at_1 = task_proof and not failures
    methods = [ev.method for ev in events if ev.method]
    unique_methods = set(methods)
    compiled = [ev for ev in events if ev.phase == "candidates" and ev.detail == "compiled"]
    source_counts: dict[str, int] = {}
    for event in compiled:
        source = str(event.payload.get("source_kind") or event.method or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    failure_counts: dict[str, int] = {}
    for event in events:
        if event.failure_class:
            failure_counts[event.failure_class] = failure_counts.get(event.failure_class, 0) + 1

    return {
        "task_proof_obtained": task_proof,
        "vulnerability_confirmed": vulnerability_confirmed,
        "primary_success": primary_success,
        "success_at_1": success_at_1,
        "executed_commands": len(executed),
        "invalid_command_rate": (len(invalid_cmds) / len(executed)) if executed else 0.0,
        "validated_vulnerability_discovery": vulnerability_confirmed or task_proof,
        "method_diversity": len(unique_methods),
        "repeated_method_rate": (1.0 - (len(unique_methods) / len(methods))) if methods else 0.0,
        "alternate_method_rescue": rescue,
        "fallback_rescue_rate": 1.0 if rescue else 0.0,
        "compiled_method_count": len(compiled),
        "compiled_method_count_by_source": source_counts,
        "failure_count_by_class": failure_counts,
        "preflight_rejection_rate": (
            sum(1 for event in events if event.failure_class in {"syntax_invalid", "option_invalid", "dependency_missing"})
            / len(compiled)
        ) if compiled else 0.0,
        "oracle_confirmed_proof": task_proof,
        "false_positive_on_patched_control": truth.is_patched_control and task_proof,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": round(cost, 6),
        "outcome_counts": {k: outcomes.count(k) for k in ALLOWED_OUTCOMES},
    }


def aggregate_metrics(rows: list[ResultRow]) -> dict[str, Any]:
    """Aggregate per-run rows into the preregistered endpoint metrics."""
    vulnerable = [r for r in rows if not _is_control_row(r)]
    controls = [r for r in rows if _is_control_row(r)]
    # Group vulnerable rows by (target_id, condition).
    by_pair: dict[tuple[str, str], list[ResultRow]] = {}
    for r in vulnerable:
        by_pair.setdefault((r.target_id, r.condition), []).append(r)

    full_vs_poc: list[tuple[bool, bool]] = []   # (full, poc) per pair across all reps
    success_at_1_pairs = 0
    vulnerable_total = 0
    vulnerable_success_any = 0
    vulnerable_success_reps: list[int] = []
    for rows_for_pair in by_pair.values():
        full_rows = [r for r in rows_for_pair if r.variant == "4"]
        poc_rows = [r for r in rows_for_pair if r.variant == "1"]
        full_success = any(r.outcome == "task_proof_obtained" for r in full_rows)
        poc_success = any(r.outcome == "task_proof_obtained" for r in poc_rows)
        full_vs_poc.append((full_success, poc_success))
        if any(r.success_at_1 and r.outcome == "task_proof_obtained" for r in rows_for_pair):
            success_at_1_pairs += 1
        vulnerable_total += 1
        if any(r.outcome == "task_proof_obtained" for r in rows_for_pair):
            vulnerable_success_any += 1
        vulnerable_success_reps.append(sum(1 for r in rows_for_pair
                                            if r.outcome == "task_proof_obtained"))

    discordant = sum(1 for f, p in full_vs_poc if f != p)
    # Report status without claiming significance when discordants are too few.
    discordant_for_mcnemar = sum(1 for f, p in full_vs_poc if f and not p)
    discordant_for_mcnemar_opposite = sum(1 for f, p in full_vs_poc if p and not f)

    false_positives = sum(1 for r in controls if r.outcome == "task_proof_obtained")

    avg_cost = (sum(r.cost for r in rows) / len(rows)) if rows else 0.0
    cost_per_success = (sum(r.cost for r in rows) / vulnerable_success_any) if vulnerable_success_any else 0.0

    wilson_total = _wilson_interval(vulnerable_success_any, vulnerable_total)

    return {
        "vulnerable_targets": vulnerable_total,
        "success_pairs": vulnerable_success_any,
        "wilson_95_success": list(wilson_total),
        "success_at_1_pairs": success_at_1_pairs,
        "discordant_pairs": discordant,
        "discordant_full_minus_poc": discordant_for_mcnemar,
        "discordant_poc_minus_full": discordant_for_mcnemar_opposite,
        "power_adequate": (discordant_for_mcnemar + discordant_for_mcnemar_opposite) >= 10,
        "controls": len(controls),
        "false_positive_count": false_positives,
        "wilson_95_false_positive": list(_wilson_interval(false_positives, len(controls))),
        "avg_cost_per_target": round(avg_cost, 6),
        "cost_per_success": round(cost_per_success, 6),
        "total_cost": round(sum(r.cost for r in rows), 6),
        "invalid_command_count": sum(r.invalid_commands for r in rows),
        "executed_command_count": sum(r.executed_commands for r in rows),
    }


def _is_control_row(r: ResultRow) -> bool:
    return r.target_id.startswith("control-") or "patched" in r.target_id.lower()


# ── Reporting ─────────────────────────────────────────────────────────────────


def write_per_target_table(rows: list[ResultRow], path: str) -> None:
    """Emit a CSV with one row per (target, condition, repetition)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "target_id", "condition", "repetition", "variant",
            "outcome", "success_at_1", "executed_commands",
            "invalid_commands", "tokens_in", "tokens_out",
            "cost", "elapsed_seconds", "repo_commit", "model_id",
            "source_snapshot", "proof_ref",
        ])
        for r in rows:
            writer.writerow([
                r.target_id, r.condition, r.repetition, r.variant,
                r.outcome, int(r.success_at_1), r.executed_commands,
                r.invalid_commands, r.tokens_in, r.tokens_out,
                f"{r.cost:.6f}", f"{r.elapsed_seconds:.3f}",
                r.repo_commit, r.model_id, r.source_snapshot_id, r.proof_ref,
            ])


def write_summary(rows: list[ResultRow], path: str) -> None:
    """Write aggregate metrics + per-target table to a single report file."""
    summary = aggregate_metrics(rows)
    lines = ["# Benchmark summary",
              "",
              f"- vulnerable targets (unique target+condition pairs): {summary['vulnerable_targets']}",
              f"- success pairs: {summary['success_pairs']}",
              f"- success @1 pairs: {summary['success_at_1_pairs']}",
              f"- Wilson 95% interval for success: "
              f"{summary['wilson_95_success'][0]:.3f}–{summary['wilson_95_success'][1]:.3f}",
              f"- Discordant pairs (full vs PoC-only): {summary['discordant_pairs']}",
              f"- McNemar discordant (full\\poc): {summary['discordant_full_minus_poc']}",
              f"- McNemar discordant (poc\\full): {summary['discordant_poc_minus_full']}",
              f"- Adequate power for McNemar: {summary['power_adequate']}",
              f"- Controls: {summary['controls']}",
              f"- False positives on controls: {summary['false_positive_count']}",
              f"- Wilson 95% interval for false positive: "
              f"{summary['wilson_95_false_positive'][0]:.3f}–"
              f"{summary['wilson_95_false_positive'][1]:.3f}",
              f"- Avg cost per target: {summary['avg_cost_per_target']}",
              f"- Cost per success (vulnerable): {summary['cost_per_success']}",
              f"- Total cost: {summary['total_cost']}",
              f"- Executed commands: {summary['executed_command_count']}",
              f"- Invalid commands: {summary['invalid_command_count']}",
              ""]
    if not summary["power_adequate"]:
        lines.append(
            "> Result is reported as inconclusive because the number of "
            "discordant pairs is too small to power the preregistered McNemar "
            "test. Hypotheses are not expanded post-hoc.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
