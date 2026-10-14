"""Deterministic, post-policy benchmark matrix generation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.pipeline.vertex_runtime import VertexContractError, validate_resolution_fields


@dataclass(frozen=True)
class MatrixCell:
    case_id: str
    framework: str
    framework_commit: str
    framework_image_digest: str
    model_label: str
    model_resource_id: str
    model_revision: str
    model_resolution_mode: str
    model_resolution_evidence_hash: str
    model_resolution_resolved_at: str
    dataset_lock_hash: str
    policy_lock_hash: str
    evaluator_commit: str
    budget_tier: str
    track: str
    condition: str
    repetition: int
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def stable_run_id(payload: Mapping[str, Any]) -> str:
    """Run identity covers the complete cell, including each baseline revision."""
    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def generate_matrix(
    *,
    test_cases: Sequence[str],
    robustness_cases: Sequence[str],
    frameworks: Sequence[Mapping[str, str]],
    models: Sequence[Mapping[str, str]],
    dataset_lock_hash: str,
    policy_lock_hash: str,
    evaluator_commit: str,
    repetitions: Iterable[int] = (1, 2, 3),
    strict: bool = False,
) -> list[MatrixCell]:
    """Generate exactly 3,807 post-policy cells, refusing ambiguous inputs."""
    if not policy_lock_hash:
        raise ValueError("final benchmark matrix requires a policy lock")
    if len(test_cases) != 27 or len(robustness_cases) != 9:
        raise ValueError("matrix requires 27 clean test and 9 robustness cases")
    if len(frameworks) != 5 or len(models) != 3:
        raise ValueError("matrix requires five frameworks and three model profiles")
    if strict:
        if not re.fullmatch(r"[0-9a-f]{64}", dataset_lock_hash) or not re.fullmatch(r"[0-9a-f]{64}", policy_lock_hash):
            raise ValueError("matrix requires real dataset and policy SHA-256 locks")
        if not re.fullmatch(r"[0-9a-f]{40}", evaluator_commit):
            raise ValueError("matrix requires a pinned evaluator commit")
        for item in frameworks:
            if not re.fullmatch(r"[0-9a-f]{40}", str(item.get("commit", ""))):
                raise ValueError("matrix framework commits must be full Git SHAs")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("image_digest", ""))):
                raise ValueError("matrix framework images must be immutable digests")
        for item in models:
            if not str(item.get("resource_id", "")).strip() or not str(item.get("resource_revision", "")).strip():
                raise ValueError("matrix models require resolved resource IDs and revisions")
            if "latest" in str(item.get("resource_revision", "")).lower():
                raise ValueError("matrix model revisions cannot use latest")
            try:
                validate_resolution_fields(
                    str(item.get("logical_label", "")),
                    str(item.get("resource_revision", "")),
                    str(item.get("resolution_mode", "immutable")),
                    str(item.get("resolution_evidence_hash", "")),
                    str(item.get("resolution_resolved_at", "")),
                )
            except VertexContractError as exc:
                raise ValueError(str(exc)) from exc

    cells: list[MatrixCell] = []

    def add(case_id: str, framework: Mapping[str, str], model: Mapping[str, str], budget: str,
            track: str, condition: str, repetition: int) -> None:
        identity: dict[str, Any] = {
            "case_id": case_id, "framework": str(framework["name"]),
            "framework_commit": str(framework["commit"]),
            "framework_image_digest": str(framework["image_digest"]),
            "model_label": str(model["logical_label"]),
            "model_resource_id": str(model["resource_id"]),
            "model_revision": str(model["resource_revision"]),
            "model_resolution_mode": str(model.get("resolution_mode", "immutable")),
            "model_resolution_evidence_hash": str(model.get("resolution_evidence_hash", "")),
            "model_resolution_resolved_at": str(model.get("resolution_resolved_at", "")),
            "dataset_lock_hash": dataset_lock_hash, "policy_lock_hash": policy_lock_hash,
            "evaluator_commit": evaluator_commit, "budget_tier": budget, "track": track,
            "condition": condition, "repetition": repetition,
        }
        cells.append(MatrixCell(
            case_id=case_id, framework=identity["framework"], framework_commit=identity["framework_commit"],
            framework_image_digest=identity["framework_image_digest"], model_label=identity["model_label"],
            model_resource_id=identity["model_resource_id"], model_revision=identity["model_revision"],
            model_resolution_mode=identity["model_resolution_mode"],
            model_resolution_evidence_hash=identity["model_resolution_evidence_hash"],
            model_resolution_resolved_at=identity["model_resolution_resolved_at"],
            dataset_lock_hash=dataset_lock_hash, policy_lock_hash=policy_lock_hash,
            evaluator_commit=evaluator_commit, budget_tier=budget, track=track, condition=condition,
            repetition=repetition, run_id=stable_run_id(identity),
        ))

    main_frameworks = list(frameworks)
    veriplan = next((item for item in frameworks if item["name"] == "VeriPlanPT"), None)
    if veriplan is None:
        raise ValueError("matrix frameworks must include VeriPlanPT")
    for case_id in test_cases:
        for framework in main_frameworks:
            for model in models:
                for track in ("blind", "guided"):
                    for repetition in repetitions:
                        add(case_id, framework, model, "medium", track, "main", repetition)
    for case_id in test_cases:
        for model in models:
            for track in ("blind", "guided"):
                for budget in ("low", "high"):
                    for repetition in repetitions:
                        add(case_id, veriplan, model, budget, track, "budget_ablation", repetition)
    for case_id in robustness_cases:
        for framework in main_frameworks:
            for model in models:
                for repetition in repetitions:
                    add(case_id, framework, model, "medium", "blind", "robustness", repetition)
    if len({cell.run_id for cell in cells}) != len(cells) or len(cells) != 3807:
        raise ValueError(f"matrix must contain 3,807 unique cells, got {len(cells)}")
    return cells


def matrix_hash(cells: Sequence[MatrixCell]) -> str:
    blob = json.dumps([cell.to_dict() for cell in cells], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_matrix(cells: Sequence[MatrixCell], json_path: str | Path, csv_path: str | Path) -> None:
    rows = [cell.to_dict() for cell in cells]
    with Path(json_path).open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def validate_matrix_files(json_path: str | Path, csv_path: str | Path) -> None:
    """Ensure JSON and CSV are projections of the same canonical rows."""
    rows = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("matrix JSON must contain non-empty rows")
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected = [{str(key): str(value) for key, value in row.items()} for row in rows]
    if csv_rows != expected:
        raise ValueError("matrix JSON/CSV canonical rows differ")
