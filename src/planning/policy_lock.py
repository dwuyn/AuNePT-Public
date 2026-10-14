"""Deterministic policy CV and lock-file helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.pipeline.protocol import write_json_atomically
from src.planning.policy import PolicyWeights

POLICY_SEED = 20260801
GRID_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class TrainCase:
    case_id: str
    severity: str
    capability: str


def stable_hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tree_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return stable_hash_text(blob)


def weight_grid() -> list[PolicyWeights]:
    out: list[PolicyWeights] = []
    for values in itertools.product(GRID_VALUES, repeat=4):
        if abs(sum(values) - 1.0) < 1e-9:
            out.append(PolicyWeights(*values))
    return sorted(out, key=lambda w: (w.w_success, w.w_evidence_gain, w.w_cost, w.w_risk))


def stratified_folds(cases: Sequence[Mapping[str, Any]], *, seed: int = POLICY_SEED) -> list[list[str]]:
    parsed = [
        TrainCase(
            case_id=str(case["case_id"]),
            severity=str(case.get("severity", "unknown")),
            capability=str(case.get("capability", "unknown")),
        )
        for case in cases
    ]
    if len(parsed) != 40:
        raise ValueError(f"policy CV requires exactly 40 train cases, got {len(parsed)}")
    folds: list[list[str]] = [[] for _ in range(5)]
    strata: dict[tuple[str, str], list[TrainCase]] = {}
    for case in parsed:
        if case.case_id.startswith("CVE-"):
            raise ValueError("policy CV train case IDs must be opaque")
        strata.setdefault((case.severity, case.capability), []).append(case)
    # Allocate every stratum independently while keeping fold sizes balanced.
    # This is deliberately not replaced by a global fallback: a global shuffle
    # erases the severity×capability protocol when a stratum is small.
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda c: stable_hash_text(f"{seed}:{c.case_id}:{c.severity}:{c.capability}"),
        )
        start = int(stable_hash_text(f"{seed}:{key[0]}:{key[1]}")[:8], 16) % 5
        for index, case in enumerate(ordered):
            candidates = [(len(folds[(start + offset) % 5]), (start + offset) % 5) for offset in range(5)]
            _, destination = min(candidates)
            folds[destination].append(case.case_id)
    for fold in folds:
        fold.sort()
    if sorted(len(fold) for fold in folds) != [8, 8, 8, 8, 8]:
        raise ValueError("stratified allocation did not yield five folds of eight cases")
    return folds


def select_weights(scores: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select by OSR, precision, token/success, HFR, then lexicographic weights."""
    scored = list(scores)
    if not scored:
        raise ValueError("no policy weight scores supplied")

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        weights = row["weights"]
        weight_tuple = (
            float(weights["w_success"]),
            float(weights["w_evidence_gain"]),
            float(weights["w_cost"]),
            float(weights["w_risk"]),
        )
        return (
            -float(row.get("osr", 0.0)),
            -float(row.get("exploit_applicability_precision", 0.0)),
            float(row.get("tokens_per_success", float("inf"))),
            float(row.get("hfr", 1.0)),
            weight_tuple,
        )

    return sorted(scored, key=key)[0]


def current_commit(repo_path: str = ".") -> str:
    try:
        return subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def build_policy_lock(
    train_cases: Sequence[Mapping[str, Any]],
    feature_schema: Mapping[str, Any],
    fold_scores: Sequence[Mapping[str, Any]],
    *,
    train_tree: Mapping[str, Any] | Sequence[Any],
    repo_path: str = ".",
    seed: int = POLICY_SEED,
) -> dict[str, Any]:
    folds = stratified_folds(train_cases, seed=seed)
    selected = select_weights(fold_scores)
    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "folds": folds,
        "grid": list(GRID_VALUES),
        "train_tree_hash": tree_hash(train_tree),
        "feature_schema_hash": tree_hash(feature_schema),
        "selected_weights": dict(selected["weights"]),
        "fold_scores": list(fold_scores),
        "code_commit": current_commit(repo_path),
    }


def validate_policy_lock(lock: Mapping[str, Any], *, dataset_train_hash: str, feature_schema_hash: str) -> None:
    if not lock:
        raise ValueError("policy lock is missing")
    if lock.get("seed") != POLICY_SEED:
        raise ValueError("policy lock seed mismatch")
    if lock.get("train_tree_hash") != dataset_train_hash:
        raise ValueError("policy lock train hash does not match dataset lock")
    if lock.get("feature_schema_hash") != feature_schema_hash:
        raise ValueError("policy lock feature schema hash mismatch")
    weights = dict(lock.get("selected_weights") or {})
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-9:
        raise ValueError("policy lock weights must sum to 1")


def freeze_policy_lock(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    valid_artifact_count: int,
    training_protocol_hash: str,
    dataset_lock_hash: str,
    profile_hashes: Sequence[str],
    evaluator_commit: str,
    framework_commit: str,
) -> None:
    """Atomically freeze a complete policy result; an existing lock is final."""
    if valid_artifact_count != 4560:
        raise ValueError("policy lock requires exactly 4,560 valid run artifacts")
    if not training_protocol_hash or not dataset_lock_hash or len(profile_hashes) != 3:
        raise ValueError("policy lock requires all upstream protocol and profile hashes")
    payload = dict(lock)
    payload.update({
        "schema_version": "2.0.0",
        "training_protocol_hash": training_protocol_hash,
        "dataset_lock_hash": dataset_lock_hash,
        "profile_hashes": sorted(profile_hashes),
        "evaluator_commit": evaluator_commit,
        "framework_commit": framework_commit,
        "valid_artifact_count": valid_artifact_count,
    })
    payload["policy_hash"] = tree_hash(payload)
    write_json_atomically(path, payload, refuse_existing=True)
