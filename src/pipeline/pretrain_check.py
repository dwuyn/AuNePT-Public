"""Generate an auditable, fail-closed pretraining readiness report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from src.pipeline.dataset_lock import (
    load_dataset_lock,
    lock_hash,
    sha256_file,
    validate_dataset_lock,
)
from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.model_resolution import validate_resolution_lock
from src.pipeline.protocol import (
    git_state,
    hash_lock_file,
    load_json,
    validate_baseline_lock,
    validate_training_protocol,
)
from src.pipeline.readiness_evidence import load_smoke_evidence, validate_smoke_evidence
from src.pipeline.runtime_contract import (
    validate_gateway_relay_lock,
    validate_lock_reference,
    verify_alias_exception,
    verify_impersonated_adc,
)
from src.pipeline.runtime_readiness import _canonical_hash, validate_canary_smoke_plan
from src.pipeline.train import plan_training


def _check(checks: dict[str, dict[str, Any]], name: str, func: Callable[[], Any]) -> None:
    try:
        detail = func()
        checks[name] = {"passed": True, "detail": detail}
    except Exception as exc:  # Report every gate rather than stopping at the first failure.
        checks[name] = {"passed": False, "detail": str(exc)}


def _command(root: Path, args: list[str]) -> dict[str, str]:
    result = subprocess.run(args, cwd=root, check=False, capture_output=True, text=True, timeout=180)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2_000:]
        raise ValueError(f"{' '.join(args)} failed: {detail}")
    return {"command": " ".join(args)}


def pretrain_check(*, dataset_root: str | Path, baseline_lock: str | Path | None = None,
                   training_protocol: str | Path | None = None, artifact_root: str | Path | None = None,
                   output: str | Path = "pretrain_readiness.json", framework_root: str | Path = ".",
                   evaluator_commit: str | None = None,
                   approver_public_key: str | None = None,
                   project: str | None = None) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    framework = Path(framework_root).resolve()
    if artifact_root is not None:
        art_path = Path(artifact_root).resolve()
        if baseline_lock is None:
            baseline_lock = art_path / "baseline.lock.json"
        if training_protocol is None:
            training_protocol = art_path / "training_protocol.json"
    if baseline_lock is None:
        baseline_lock = root / "baseline.lock.json"
    if training_protocol is None:
        training_protocol = root / "training_protocol.json"

    checks: dict[str, dict[str, Any]] = {}
    _check(checks, "framework_clean", lambda: git_state(framework) if not git_state(framework)["dirty"]
           else (_ for _ in ()).throw(ValueError("framework repository is dirty")))
    _check(checks, "poetry_lock", lambda: _command(framework, ["poetry", "check", "--lock"]))
    _check(checks, "ruff", lambda: _command(framework, ["poetry", "run", "ruff", "check", "src"]))
    _check(checks, "mypy", lambda: _command(framework, [
        "poetry", "run", "mypy", "src/pipeline", "src/planning", "src/scoring",
        "--ignore-missing-imports", "--follow-imports=skip",
    ]))
    _check(checks, "dependency_security", lambda: _command(framework, ["poetry", "run", "pip", "check"]))
    _check(checks, "dataset_clean", lambda: git_state(root) if not git_state(root)["dirty"]
           else (_ for _ in ()).throw(ValueError("dataset repository is dirty")))
    lock_path = root / "dataset.lock.json"
    dataset_lock: dict[str, Any] = {}

    def dataset_gate() -> dict[str, Any]:
        nonlocal dataset_lock
        dataset_lock = load_dataset_lock(lock_path)
        validate_dataset_lock(dataset_lock, dataset_root=root, strict=True)
        return {"lock_hash": lock_hash(dataset_lock)}

    _check(checks, "dataset_lock", dataset_gate)

    def baseline_gate() -> dict[str, str]:
        validate_baseline_lock(load_json(baseline_lock), strict=True)
        return {"lock_hash": hash_lock_file(baseline_lock)}

    _check(checks, "baseline_lock", baseline_gate)

    def protocol_gate() -> dict[str, Any]:
        if not dataset_lock:
            raise ValueError("dataset lock did not validate")
        protocol = load_json(training_protocol)
        state = git_state(framework)
        eval_commit = evaluator_commit or state["commit"]
        validate_training_protocol(
            protocol,
            dataset_hash=lock_hash(dataset_lock),
            framework_commit=state["commit"],
            evaluator_commit=eval_commit,
            artifact_root=artifact_root,
            strict_runtime=artifact_root is not None,
        )
        b_hash = hash_lock_file(baseline_lock)
        if str(protocol.get("baseline_lock_hash")) != b_hash:
            raise ValueError("training protocol baseline_lock_hash mismatch with baseline lock")
        profiles = [ModelProfile.from_dict(item) for item in protocol["model_profiles"]]
        if any(profile.resolution_mode == "provider_alias" for profile in profiles):
            if artifact_root is None:
                raise ValueError("provider alias profiles require --artifact-root")
            resolution_ref = protocol["model_resolution_lock"]
            if not isinstance(resolution_ref, dict):
                raise ValueError("model_resolution_lock is invalid")
            relative = Path(str(resolution_ref.get("artifact_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("model_resolution_lock artifact path is unsafe")
            resolution_path = Path(artifact_root).resolve() / relative
            if not resolution_path.is_file() or sha256_file(resolution_path) != resolution_ref.get("sha256"):
                raise ValueError("model_resolution_lock artifact hash mismatch")
            resolution = load_json(resolution_path)
            if str(resolution.get("dataset_lock_hash", lock_hash(dataset_lock))) != lock_hash(dataset_lock):
                raise ValueError("model resolution lock dataset hash mismatch")
            validate_resolution_lock(
                resolution, profiles=profiles, artifact_root=artifact_root,
                strict=artifact_root is not None,
            )
        return {"protocol_hash": hash_lock_file(training_protocol)}

    _check(checks, "training_protocol", protocol_gate)

    def runtime_approval_gate() -> dict[str, Any]:
        if artifact_root is None:
            raise ValueError("runtime artifact root is required for live model readiness")
        if not dataset_lock:
            raise ValueError("dataset lock did not validate")
        protocol = load_json(training_protocol)
        art_root = Path(artifact_root).resolve()
        profiles = [ModelProfile.from_dict(item) for item in protocol["model_profiles"]]
        for profile in profiles:
            if profile.resolution_mode == "provider_alias":
                alias_ref = protocol["alias_exception"]
                validate_lock_reference(alias_ref, name="alias_exception", artifact_root=art_root)
                alias_path = art_root / Path(str(alias_ref["artifact_path"]))
                signature_value = alias_ref.get("signature_path")
                if not isinstance(signature_value, str) or Path(signature_value).is_absolute() or ".." in Path(signature_value).parts:
                    raise ValueError("alias exception requires a safe detached signature path")
                verify_alias_exception(
                    load_json(alias_path), signature_path=art_root / signature_value,
                    public_key=str(approver_public_key or ""),
                    project=str(project or protocol["vertex_project"]),
                    dataset_lock_hash=lock_hash(dataset_lock),
                )
                break

        plan_ref = protocol["canary_smoke_plan"]
        approval_ref = protocol["approval_canary_smoke"]
        validate_lock_reference(plan_ref, name="canary_smoke_plan", artifact_root=art_root)
        validate_lock_reference(approval_ref, name="approval_canary_smoke", artifact_root=art_root)
        plan_path = art_root / Path(str(plan_ref["artifact_path"]))
        approval_path = art_root / Path(str(approval_ref["artifact_path"]))
        plan = load_json(plan_path)
        validate_canary_smoke_plan(plan, profiles=profiles, strict=True)
        baseline = load_json(baseline_lock)
        baseline_images = {
            str(entry["name"]): str(entry["image_digest"])
            for entry in baseline.get("baselines", [])
            if isinstance(entry, dict)
        }
        native_ref = protocol.get("native_identity")
        if not isinstance(native_ref, dict):
            raise ValueError("training protocol native_identity reference is invalid")
        native_path = art_root / Path(str(native_ref["artifact_path"]))
        native = load_json(native_path)
        native_image = str(native.get("image", {}).get("image_digest", ""))
        native_hash = sha256_file(native_path)
        relay_ref = protocol.get("gateway_relay_lock")
        if not isinstance(relay_ref, dict):
            raise ValueError("training protocol gateway_relay_lock reference is invalid")
        validate_lock_reference(relay_ref, name="gateway_relay_lock", artifact_root=art_root)
        relay_path = art_root / Path(str(relay_ref["artifact_path"]))
        relay_lock = load_json(relay_path)
        validate_gateway_relay_lock(relay_lock, artifact_root=art_root, strict=True)
        relay_hash = sha256_file(relay_path)
        if str(protocol.get("gateway_relay_lock_hash")) != relay_hash or str(relay_ref.get("sha256")) != relay_hash:
            raise ValueError("gateway relay lock hash is not pinned consistently")
        expected_common = {
            "dataset_lock_hash": lock_hash(dataset_lock),
            "baseline_identity_hash": hash_lock_file(baseline_lock),
            "native_identity_hash": native_hash,
            "model_resolution_lock_hash": str(protocol["model_resolution_lock"]["sha256"]),
            "evaluator_hash": str(protocol["evaluator_bundle_hash"]),
            "oracle_hash": str(protocol["oracle_bundle_hash"]),
            "gateway_relay_lock_hash": relay_hash,
        }
        profile_by_label = {profile.logical_label: profile for profile in profiles}
        for cell in plan["cells"]:
            for field, expected in expected_common.items():
                if str(cell.get(field, "")) != expected:
                    raise ValueError(f"canary smoke {field} is not pinned to the verified protocol")
            profile = profile_by_label[str(cell["model_label"])]
            if str(cell["model_profile_hash"]) != profile.profile_hash:
                raise ValueError("canary smoke model profile hash mismatch")
            framework = str(cell.get("framework", ""))
            expected_image = native_image if cell["kind"] == "vertex_canary" else baseline_images.get(framework, "")
            if str(cell["image_digest"]) != expected_image:
                raise ValueError(f"canary smoke image digest mismatch for {cell['run_id']}")
            if str(cell.get("gateway_relay_lock_hash")) != relay_hash:
                raise ValueError("canary smoke gateway relay lock hash mismatch")
            budgets = protocol["runtime_budgets"]
            if int(cell["max_input_tokens"]) != int(budgets["max_input_tokens"]):
                raise ValueError("canary smoke input-token cap mismatch")
            if int(cell["max_output_tokens"]) != int(budgets["max_output_tokens"]):
                raise ValueError("canary smoke output-token cap mismatch")
        approval = load_json(approval_path)
        signature_value = approval_ref.get("signature_path")
        if not isinstance(signature_value, str) or Path(signature_value).is_absolute() or ".." in Path(signature_value).parts:
            raise ValueError("canary approval requires a safe detached signature path")
        reserved = sum(float(cell["cell_worst_case_cost_usd"]) for cell in plan["cells"])
        from src.pipeline.approval import verify_approval
        verified = verify_approval(
            approval, scope="canary_smoke", plan_hash=_canonical_hash(plan),
            cell_count=18, cost_ceiling_usd=reserved,
            signature_path=art_root / signature_value, public_key=str(approver_public_key or ""),
        )
        adc = verify_impersonated_adc(
            str(protocol["impersonate_service_account"]),
            project=str(project or protocol["vertex_project"]),
        )
        return {
            "plan_hash": _canonical_hash(plan), "approval_hash": sha256_file(approval_path),
            "expires_at": verified["expires_at"], "reserved_cost_usd": reserved,
            "adc": adc,
        }

    _check(checks, "runtime_approval", runtime_approval_gate)

    def smoke_gate() -> dict[str, Any]:
        if not dataset_lock:
            raise ValueError("dataset lock did not validate")
        protocol = load_json(training_protocol)
        profiles = protocol.get("model_profiles")
        if not isinstance(profiles, list):
            raise ValueError("training protocol model_profiles is invalid")
        evidence_root = Path(artifact_root).resolve() if artifact_root is not None else root
        art_root = evidence_root
        runtime_path = evidence_root / "readiness" / "runtime-smoke-evidence.json"
        dataset_path = root / "readiness" / "dataset-freeze-evidence.json"
        if artifact_root is not None and (not runtime_path.is_file() or not dataset_path.is_file()):
            raise ValueError("runtime pretrain requires completed dataset and 18-cell runtime smoke evidence")
        if runtime_path.exists() and dataset_path.exists():
            dataset_summary = validate_smoke_evidence(
                load_smoke_evidence(dataset_path),
                base_case_ids=[
                    *dataset_lock["train_cases"], *dataset_lock["validation_cases"], *dataset_lock["test_cases"]
                ],
                robustness_base_case_ids=dataset_lock["test_cases"], mode="dataset-freeze", artifact_root=root,
            )
            runtime_summary = validate_smoke_evidence(
                load_smoke_evidence(runtime_path),
                base_case_ids=[],
                model_labels=[
                    str(profile.get("logical_label") or profile.get("model_name") or "")
                    for profile in profiles if isinstance(profile, dict)
                ],
                mode="runtime-smoke", artifact_root=evidence_root,
            )
            runtime = load_smoke_evidence(runtime_path)
            if str(runtime.get("dataset_lock_hash")) != lock_hash(dataset_lock):
                raise ValueError("runtime smoke dataset lock hash mismatch")
            if str(runtime.get("dataset_evidence_hash")) != sha256_file(dataset_path):
                raise ValueError("runtime smoke dataset evidence hash mismatch")
            if runtime.get("schema_version") in {"2.1.0", "2.2.0"}:
                protocol_hash = hash_lock_file(training_protocol)
                protocol = load_json(training_protocol)
                if str(runtime.get("training_protocol_hash")) != protocol_hash:
                    raise ValueError("runtime smoke training protocol hash mismatch")
                if str(runtime.get("baseline_lock_hash")) != hash_lock_file(baseline_lock):
                    raise ValueError("runtime smoke baseline lock hash mismatch")
                for field, ref_name in (
                    ("model_resolution_lock_hash", "model_resolution_lock"),
                    ("pricing_snapshot_hash", "pricing_snapshot"),
                    ("plan_hash", "canary_smoke_plan"),
                    ("approval_hash", "approval_canary_smoke"),
                ):
                    expected = (
                        str(protocol[ref_name].get("sha256", ""))
                        if field != "plan_hash" else str(load_json(art_root / Path(str(protocol[ref_name]["artifact_path"]))).get("plan_hash", ""))
                    )
                    if str(runtime.get(field)) != expected:
                        raise ValueError(f"runtime smoke {field} mismatch")
                if runtime.get("schema_version") == "2.2.0":
                    relay_hash = str(protocol.get("gateway_relay_lock_hash", ""))
                    if str(runtime.get("gateway_relay_lock_hash")) != relay_hash:
                        raise ValueError("runtime smoke gateway relay lock hash mismatch")
            return {**dataset_summary, **runtime_summary}
        evidence_path = root / "readiness" / "smoke-evidence.json"
        return validate_smoke_evidence(
            load_smoke_evidence(evidence_path),
            base_case_ids=[
                *dataset_lock["train_cases"],
                *dataset_lock["validation_cases"],
                *dataset_lock["test_cases"],
            ],
            model_labels=[
                str(profile.get("logical_label") or profile.get("model_name") or "")
                for profile in profiles
                if isinstance(profile, dict)
            ],
            robustness_base_case_ids=dataset_lock["test_cases"],
            mode="pretrain",
            artifact_root=None,
        )

    _check(checks, "smoke_evidence", smoke_gate)

    def no_test_artifacts() -> dict[str, Any]:
        test_root = root / "cases" / "test"
        if any(test_root.rglob("run_artifact.json")):
            raise ValueError("test split has run artifacts before policy freeze")
        if (root / "policy.lock.json").exists() or (root / "matrix.json").exists() or (root / "matrix.csv").exists():
            raise ValueError("policy lock or final matrix exists before training")
        return {"test_root": str(test_root)}

    _check(checks, "test_sealed", no_test_artifacts)
    _check(checks, "train_dry_run", lambda: {
        "cells": plan_training(dataset_root=root, protocol_path=training_protocol,
                               output_path=Path(output).with_name("training_plan.preview.json"))["cell_count"]
    })
    report = {
        "schema_version": "2.0.0",
        "ready": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "framework_commit": git_state(framework)["commit"],
        "dataset_lock_hash": lock_hash(dataset_lock) if dataset_lock else "",
        "baseline_lock_hash": hash_lock_file(baseline_lock) if Path(baseline_lock).exists() else "",
        "training_protocol_hash": hash_lock_file(training_protocol) if Path(training_protocol).exists() else "",
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate every gate before Vertex policy training.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--artifact-root", help="Directory containing downstream artifacts (baseline.lock.json, training_protocol.json)")
    parser.add_argument("--baseline-lock")
    parser.add_argument("--training-protocol")
    parser.add_argument("--evaluator-commit")
    parser.add_argument("--approver-public-key")
    parser.add_argument("--project")
    parser.add_argument("--output", default="pretrain_readiness.json")
    args = parser.parse_args(argv)
    report = pretrain_check(
        dataset_root=args.dataset_root,
        baseline_lock=args.baseline_lock,
        training_protocol=args.training_protocol,
        artifact_root=args.artifact_root,
        evaluator_commit=args.evaluator_commit,
        approver_public_key=args.approver_public_key,
        project=args.project,
        output=args.output,
    )
    print(json.dumps({"ready": report["ready"], "output": args.output}, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
