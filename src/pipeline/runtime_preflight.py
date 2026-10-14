"""Verify live-runtime authority without reading smoke evidence or invoking Vertex."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.pipeline.approval import verify_approval
from src.pipeline.dataset_lock import load_dataset_lock, lock_hash, validate_dataset_lock
from src.pipeline.framework_adapter import ModelProfile
from src.pipeline.model_resolution import validate_resolution_lock
from src.pipeline.protocol import (
    git_state,
    hash_lock_file,
    load_json,
    validate_baseline_lock,
    validate_training_protocol,
)
from src.pipeline.runtime_contract import (
    sha256_file,
    validate_gateway_relay_lock,
    validate_lock_reference,
    verify_alias_exception,
    verify_impersonated_adc,
)
from src.pipeline.runtime_readiness import _canonical_hash, validate_canary_smoke_plan


def runtime_preflight(*, dataset_root: str | Path, artifact_root: str | Path,
                      approver_public_key: str, evaluator_commit: str, evaluator_bundle_hash: str,
                      oracle_bundle_hash: str, project: str | None = None) -> dict[str, Any]:
    root, artifacts = Path(dataset_root).resolve(), Path(artifact_root).resolve()
    lock = load_dataset_lock(root / "dataset.lock.json")
    validate_dataset_lock(lock, dataset_root=root, strict=True)
    protocol = load_json(artifacts / "training_protocol.json")
    state = git_state(Path(__file__).resolve().parents[2])
    if state["dirty"]:
        raise ValueError("runtime preflight requires a clean framework worktree")
    validate_training_protocol(
        protocol, dataset_hash=lock_hash(lock), framework_commit=str(state["commit"]),
        evaluator_commit=evaluator_commit, artifact_root=artifacts, strict_runtime=True,
    )
    if str(protocol["evaluator_bundle_hash"]) != evaluator_bundle_hash or str(protocol["oracle_bundle_hash"]) != oracle_bundle_hash:
        raise ValueError("observed evaluator or oracle bundle hash does not match training protocol")
    baseline_path = artifacts / "baseline.lock.json"
    validate_baseline_lock(load_json(baseline_path), strict=True)
    if hash_lock_file(baseline_path) != str(protocol["baseline_lock_hash"]):
        raise ValueError("baseline lock does not match training protocol")
    profiles = [ModelProfile.from_dict(item) for item in protocol["model_profiles"]]
    resolution_ref = protocol["model_resolution_lock"]
    resolution_path = artifacts / Path(str(resolution_ref["artifact_path"]))
    validate_resolution_lock(load_json(resolution_path), profiles=profiles, artifact_root=artifacts, strict=True)
    alias_ref = protocol["alias_exception"]
    validate_lock_reference(alias_ref, name="alias_exception", artifact_root=artifacts)
    verify_alias_exception(
        load_json(artifacts / Path(str(alias_ref["artifact_path"]))),
        signature_path=artifacts / str(alias_ref["signature_path"]), public_key=approver_public_key,
        project=project or str(protocol["vertex_project"]), dataset_lock_hash=lock_hash(lock),
    )
    plan_ref, approval_ref = protocol["canary_smoke_plan"], protocol["approval_canary_smoke"]
    validate_lock_reference(plan_ref, name="canary_smoke_plan", artifact_root=artifacts)
    validate_lock_reference(approval_ref, name="approval_canary_smoke", artifact_root=artifacts)
    plan = load_json(artifacts / Path(str(plan_ref["artifact_path"])))
    validate_canary_smoke_plan(plan, profiles=profiles, strict=True)
    relay_ref = protocol["gateway_relay_lock"]
    validate_lock_reference(relay_ref, name="gateway_relay_lock", artifact_root=artifacts)
    relay_path = artifacts / Path(str(relay_ref["artifact_path"]))
    relay_lock = load_json(relay_path)
    validate_gateway_relay_lock(relay_lock, artifact_root=artifacts, strict=True)
    relay_hash = sha256_file(relay_path)
    if str(protocol.get("gateway_relay_lock_hash")) != relay_hash or str(relay_ref.get("sha256")) != relay_hash:
        raise ValueError("runtime protocol gateway relay lock hash mismatch")
    baseline = load_json(baseline_path)
    baseline_images = {
        str(entry["name"]): str(entry["image_digest"])
        for entry in baseline.get("baselines", []) if isinstance(entry, Mapping)
    }
    native_ref = protocol["native_identity"]
    validate_lock_reference(native_ref, name="native_identity", artifact_root=artifacts)
    native = load_json(artifacts / Path(str(native_ref["artifact_path"])))
    native_hash = sha256_file(artifacts / Path(str(native_ref["artifact_path"])))
    native_image = str(native.get("image", {}).get("image_digest", ""))
    expected_common = {
        "dataset_lock_hash": lock_hash(lock), "baseline_identity_hash": hash_lock_file(baseline_path),
        "native_identity_hash": native_hash, "model_resolution_lock_hash": str(protocol["model_resolution_lock"]["sha256"]),
        "evaluator_hash": str(protocol["evaluator_bundle_hash"]), "oracle_hash": str(protocol["oracle_bundle_hash"]),
        "gateway_relay_lock_hash": relay_hash,
    }
    profile_by_label = {profile.logical_label: profile for profile in profiles}
    reservation = sum(float(cell["cell_worst_case_cost_usd"]) for cell in plan["cells"])
    verified = verify_approval(
        load_json(artifacts / Path(str(approval_ref["artifact_path"]))), scope="canary_smoke",
        plan_hash=_canonical_hash(plan), cell_count=18, cost_ceiling_usd=reservation,
        signature_path=artifacts / str(approval_ref["signature_path"]), public_key=approver_public_key,
    )
    for cell in plan["cells"]:
        for field, expected in expected_common.items():
            if str(cell.get(field, "")) != expected:
                raise ValueError(f"runtime cell {field} binding mismatch")
        profile = profile_by_label.get(str(cell.get("model_label", "")))
        if profile is None or str(cell.get("model_profile_hash")) != profile.profile_hash:
            raise ValueError("runtime cell model profile binding mismatch")
        expected_image = native_image if cell.get("kind") == "vertex_canary" else baseline_images.get(str(cell.get("framework", "")), "")
        if str(cell.get("image_digest")) != expected_image:
            raise ValueError(f"runtime cell framework image binding mismatch: {cell.get('run_id', '')}")
        if int(cell.get("max_input_tokens", 0)) != int(protocol["runtime_budgets"]["max_input_tokens"]):
            raise ValueError("runtime cell input-token binding mismatch")
        if int(cell.get("max_output_tokens", 0)) != int(protocol["runtime_budgets"]["max_output_tokens"]):
            raise ValueError("runtime cell output-token binding mismatch")
        if str(cell.get("gateway_relay_lock_hash")) != relay_hash:
            raise ValueError(f"runtime cell gateway relay lock binding mismatch: {cell.get('run_id', '')}")
        image = str(cell["image_digest"])
        inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, check=False)
        if inspected.returncode or inspected.stdout.strip() != image:
            raise ValueError(f"runtime image digest unavailable: {image}")
    relay_image = str(relay_lock["relay"].get("image", ""))
    if not relay_image:
        raise ValueError("gateway relay lock does not name an image")
    inspected_relay = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", relay_image],
        capture_output=True, text=True, check=False,
    )
    if inspected_relay.returncode or inspected_relay.stdout.strip() != str(relay_lock["relay"]["image_digest"]):
        raise ValueError("gateway relay image digest unavailable or drifted")
    adc = verify_impersonated_adc(str(protocol["impersonate_service_account"]), project=project or str(protocol["vertex_project"]))
    return {"ready": True, "plan_hash": _canonical_hash(plan), "reservation_ceiling_usd": reservation,
            "approval_expires_at": verified["expires_at"], "adc": adc, "vertex_calls": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approver-public-key", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--evaluator-bundle-hash", required=True)
    parser.add_argument("--oracle-bundle-hash", required=True)
    parser.add_argument("--project")
    args = parser.parse_args(argv)
    try:
        report = runtime_preflight(**vars(args))
    except Exception as exc:
        print(json.dumps({"ready": False, "vertex_calls": 0, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
