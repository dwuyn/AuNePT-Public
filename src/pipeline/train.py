"""Fail-closed training planner for the 4,560-cell policy protocol.

Dry run is the default.  The module intentionally does not contain a Vertex
client: execution is injected by the experiment service after explicit cost
approval and canary validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.pipeline.dataset_lock import load_dataset_lock, lock_hash, validate_dataset_lock
from src.pipeline.protocol import git_state, hash_lock_file, load_json, validate_training_protocol
from src.planning.policy_lock import POLICY_SEED, stratified_folds, weight_grid


@dataclass(frozen=True)
class TrainingCell:
    phase: str
    case_id: str
    fold: int
    model_label: str
    model_profile_hash: str
    budget_tier: str
    track: str
    repetition: int
    weights: dict[str, float] | None
    run_id: str


def _run_id(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def _require_run_context(run_context: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable identities that make a training run resumable."""
    required = {
        "dataset_commit",
        "dataset_lock_hash",
        "training_protocol_hash",
        "framework_commit",
        "evaluator_commit",
    }
    missing = sorted(key for key in required if not str(run_context.get(key, "")).strip())
    if missing:
        raise ValueError(f"training run context missing immutable identity: {', '.join(missing)}")
    return {key: str(run_context[key]) for key in required}


def training_cells(
    cases: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    run_context: Mapping[str, Any],
) -> list[TrainingCell]:
    """Materialize only the 4,200 approved sweep cells.

    Confirmation is deliberately absent: it is illegal to create a confirmation
    run with ``weights=None`` before the winner is selected.
    """
    context = _require_run_context(run_context)
    folds = stratified_folds(cases)
    fold_index = {case_id: index for index, fold in enumerate(folds) for case_id in fold}
    if len(fold_index) != 40 or len(profiles) != 3:
        raise ValueError("training protocol requires 40 cases and 3 model profiles")
    cells: list[TrainingCell] = []
    for weights in weight_grid():
        weight_data = weights.to_dict()
        for case in cases:
            for profile in profiles:
                payload = {
                    "phase": "sweep", "case_id": case["case_id"], "fold": fold_index[case["case_id"]],
                    "model_label": profile["logical_label"], "model_profile_hash": profile["profile_hash"],
                    "budget_tier": "medium", "track": "blind", "repetition": 1, "weights": weight_data,
                }
                cells.append(TrainingCell(**payload, run_id=_run_id({**context, **payload})))
    if len(cells) != 4200 or len({cell.run_id for cell in cells}) != 4200:
        raise ValueError("training sweep must contain 4,200 unique cells")
    return cells


def confirmation_cells(
    cases: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    selected_weights: Mapping[str, float],
    run_context: Mapping[str, Any],
) -> list[TrainingCell]:
    """Materialize the 360 confirmation cells only after weight selection."""
    context = _require_run_context(run_context)
    required_weights = {"w_success", "w_evidence_gain", "w_cost", "w_risk"}
    if set(selected_weights) != required_weights:
        raise ValueError("confirmation requires the complete selected weight vector")
    weights = {name: float(selected_weights[name]) for name in sorted(required_weights)}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("confirmation weights must sum to 1")
    folds = stratified_folds(cases)
    fold_index = {case_id: index for index, fold in enumerate(folds) for case_id in fold}
    if len(fold_index) != 40 or len(profiles) != 3:
        raise ValueError("training protocol requires 40 cases and 3 model profiles")
    cells: list[TrainingCell] = []
    for case in cases:
        for profile in profiles:
            for repetition in (1, 2, 3):
                payload = {
                    "phase": "confirmation", "case_id": case["case_id"], "fold": fold_index[case["case_id"]],
                    "model_label": profile["logical_label"], "model_profile_hash": profile["profile_hash"],
                    "budget_tier": "medium", "track": "blind", "repetition": repetition, "weights": weights,
                }
                cells.append(TrainingCell(**payload, run_id=_run_id({**context, **payload})))
    if len(cells) != 360 or len({cell.run_id for cell in cells}) != 360:
        raise ValueError("confirmation must contain 360 unique cells")
    return cells


def _read_train_cases(dataset_root: Path, lock: Mapping[str, Any]) -> list[dict[str, str]]:
    metadata = dataset_root / "sealed-train-metadata.json"
    if not metadata.exists():
        raise ValueError("sealed train metadata is missing")
    data = load_json(metadata)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("sealed train metadata cases is invalid")
    expected = set(str(value) for value in lock["train_cases"])
    selected = [dict(case) for case in cases if str(case.get("case_id")) in expected]
    if {str(case.get("case_id")) for case in selected} != expected:
        raise ValueError("sealed train metadata does not match dataset train split")
    for case in selected:
        if not str(case.get("severity", "")).strip() or not str(case.get("capability", "")).strip():
            raise ValueError(f"sealed train metadata record {case.get('case_id')} missing severity or capability")
    if any("test" in str(case) for case in selected):
        raise ValueError("training metadata contains a test path")
    return selected


def plan_training(*, dataset_root: str | Path, protocol_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    parts = tuple(part.lower() for part in root.parts)
    is_test_split = root.name.lower() == "test" or any(
        parts[index:index + 2] == ("cases", "test") for index in range(len(parts) - 1)
    )
    if is_test_split or not root.exists():
        raise ValueError("training requires an explicit dataset root, never a test path")
    lock_path = root / "dataset.lock.json"
    lock = load_dataset_lock(lock_path)
    validate_dataset_lock(lock, dataset_root=root)
    protocol = load_json(protocol_path)
    state = git_state(Path(__file__).resolve().parents[2])
    if state["dirty"]:
        raise ValueError("training refuses a dirty framework repository")
    evaluator_commit = str(protocol.get("evaluator_commit") or protocol.get("evaluator_source_hash") or "")
    validate_training_protocol(protocol, dataset_hash=lock_hash(lock), framework_commit=str(state["commit"]),
                               evaluator_commit=evaluator_commit)
    cases = _read_train_cases(root, lock)
    protocol_hash = hash_lock_file(protocol_path)
    context = {
        "dataset_commit": str(protocol["dataset_repository_commit"]),
        "dataset_lock_hash": lock_hash(lock),
        "training_protocol_hash": protocol_hash,
        "framework_commit": str(state["commit"]),
        "evaluator_commit": evaluator_commit,
    }
    cells = training_cells(cases, protocol["model_profiles"], run_context=context)
    plan = {
        "schema_version": "3.0.0",
        "seed": POLICY_SEED,
        "run_context": context,
        "dataset_lock_hash": lock_hash(lock),
        "training_protocol_hash": protocol_hash,
        "cell_count": len(cells),
        "sweep_cell_count": 4200,
        "confirmation_cell_count": 360,
        "confirmation_materialized": False,
        "estimated_cost_usd": protocol.get("cost_estimate_usd"),
        "cells": [asdict(cell) for cell in cells],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute VeriPlanPT policy training.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-protocol", required=True)
    parser.add_argument("--output", default="training_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-cost", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_training(dataset_root=args.dataset_root, protocol_path=args.training_protocol, output_path=args.output)
    if args.execute and not args.approve_cost:
        raise SystemExit("--execute requires --approve-cost before any Vertex call")
    if args.execute:
        raise SystemExit("execution service is intentionally external; use the approved runner with this plan")
    print(json.dumps({key: plan[key] for key in ("cell_count", "sweep_cell_count", "confirmation_cell_count", "estimated_cost_usd")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
