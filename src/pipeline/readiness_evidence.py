"""Canonical VeriPlanPT readiness-evidence contract shared with the dataset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.pipeline.vertex_runtime import VertexContractError, validate_resolution_fields

SCHEMA_VERSION = "2.0.0"
LEGACY_RUNTIME_SCHEMA_VERSION = "2.1.0"
RUNTIME_SCHEMA_VERSION = "2.2.0"
R10_5_RUNTIME_SCHEMA_VERSION = "2.3.0"
R10_6_RUNTIME_SCHEMA_VERSION = "2.4.0"
R10_7_RUNTIME_SCHEMA_VERSION = "2.6.0"
R10_4_RUNTIME_CONTRACT = "veriplanpt-runtime-v0.4.0-r10.4"
R10_5_RUNTIME_CONTRACT = "veriplanpt-runtime-v0.4.0-r10.5"
R10_6_RUNTIME_CONTRACT = "veriplanpt-runtime-v0.4.0-r10.6"
R10_7_RUNTIME_CONTRACT = "veriplanpt-runtime-v0.4.0-r10.8"
R10_10_RUNTIME_CONTRACT = "veriplanpt-runtime-v0.4.0-r10.10"
BASE_CASE_COUNT = 94
ROBUSTNESS_COUNT = 9
VERTEX_CANARY_COUNT = 3
FRAMEWORK_MODEL_SMOKE_COUNT = 15
ROBUSTNESS_STRATA = frozenset({"semantic_preserving", "environmental", "deceptive_noise"})
FRAMEWORKS = frozenset({"VeriPlanPT", "PentestGPT", "VulnBot", "HackSynth", "PentestAgent"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_owned_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": evidence.get("schema_version"),
        "base_case_fixed_controls": evidence.get("base_case_fixed_controls"),
        "robustness": evidence.get("robustness"),
    }


def dataset_owned_evidence_hash(evidence: Mapping[str, Any]) -> str:
    return canonical_hash(dataset_owned_evidence(evidence))


def load_smoke_evidence(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    if not isinstance(evidence, dict):
        raise ValueError("smoke evidence must be a JSON object")
    return evidence


def _relative_artifact_path(value: Any, name: str) -> Path:
    """Return a safe relative artifact path.

    A digest copied into a JSON file is not evidence by itself.  Official
    readiness gates therefore require a path that can be opened and hashed.
    ``Path`` normalisation is deliberately strict so an evidence file cannot
    escape its artifact root.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.artifact_path must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name}.artifact_path must stay relative to the artifact root")
    return path


def rehash_artifact(record: Mapping[str, Any], *, artifact_root: str | Path, name: str) -> str:
    """Open and SHA-256 an evidence artifact, refusing hash-only records."""
    path = _relative_artifact_path(record.get("artifact_path"), name)
    expected = record.get("artifact_sha256")
    _digest(expected, f"{name}.artifact_sha256")
    full_path = Path(artifact_root).resolve() / path
    if not full_path.is_file():
        raise ValueError(f"{name}.artifact_path does not reference an existing file")
    digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
    if digest != str(expected):
        raise ValueError(f"{name}.artifact_sha256 does not match artifact contents")
    return digest


def _records(value: Any, name: str, expected: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"smoke evidence requires exactly {expected} {name} records")
    if not all(isinstance(record, Mapping) for record in value):
        raise ValueError(f"smoke evidence {name} records must be objects")
    return value


def _digest(value: Any, name: str) -> None:
    if not SHA256.fullmatch(str(value)):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _passed(
    record: Mapping[str, Any], name: str, *, artifact_root: str | Path | None = None,
    production: bool = False,
) -> None:
    if record.get("status") != "passed":
        raise ValueError(f"{name} must have status='passed'")
    digest_value = record.get("evidence_sha256", record.get("artifact_sha256"))
    _digest(digest_value, f"{name}.evidence_sha256")
    if artifact_root is not None:
        path = _relative_artifact_path(record.get("artifact_path"), name)
        rehash_artifact(record, artifact_root=artifact_root, name=name)
        if record.get("artifact_type") == "run_artifact":
            try:
                payload = json.loads((Path(artifact_root).resolve() / path).read_text(encoding="utf-8"))
                from src.pipeline.protocol import validate_run_artifact
                validate_run_artifact(payload, official=True, strict_runtime=production)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{name}.artifact is not a valid official RunArtifact") from exc


def _runtime_record(
    record: Mapping[str, Any], name: str, *, artifact_root: str | Path,
    production: bool = False,
) -> None:
    """Verify the complete source-backed v2 runtime cell evidence."""
    r10_4 = str(record.get("runtime_contract", "")) in {
        R10_4_RUNTIME_CONTRACT, R10_5_RUNTIME_CONTRACT, R10_6_RUNTIME_CONTRACT,
        R10_7_RUNTIME_CONTRACT, R10_10_RUNTIME_CONTRACT,
    } or str(record.get("evaluation_scope", "")) == "readiness_transport"
    r10_5 = str(record.get("runtime_contract", "")) in {
        R10_5_RUNTIME_CONTRACT, R10_6_RUNTIME_CONTRACT, R10_7_RUNTIME_CONTRACT,
        R10_10_RUNTIME_CONTRACT,
    }
    required = {
        "status", "run_id", "model_label", "plan_hash", "dataset_lock_hash",
        "baseline_identity_hash", "native_identity_hash", "model_profile_hash",
        "model_resolution_lock_hash", "evaluator_hash", "oracle_hash", "image_digest",
        "max_input_tokens", "max_output_tokens", "retry_policy", "artifact_path",
        "artifact_sha256", "artifact_type", "event_ledger_path", "event_ledger_sha256",
        "proof_path", "proof_sha256",
        "usage_path", "usage_sha256", "cost_path", "cost_sha256", "evaluator_path",
        "evaluator_sha256", "cleanup_path", "cleanup_sha256", "billing_status",
        "oracle_status",
    }
    if r10_4:
        required.update({
            "execution_kind", "condition", "evaluation_scope", "readiness_kind",
            "metric_eligible", "oracle_applicability_path", "oracle_applicability_sha256",
        })
    if production:
        required.add("gateway_relay_lock_hash")
        required.update({"invocation_ledger_path", "invocation_ledger_sha256"})
        if r10_5:
            required.update({
                "invocation_ledger_hash", "invocation_call_indices",
                "invocation_replay_count", "invocation_response_count",
            })
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"{name} is missing runtime evidence field(s): {', '.join(missing)}")
    _passed(record, name, artifact_root=artifact_root, production=production)
    if record.get("artifact_type") != "run_artifact":
        raise ValueError(f"{name}.artifact_type must be run_artifact")
    if not str(record.get("image_digest", "")).startswith("sha256:"):
        raise ValueError(f"{name}.image_digest must be immutable")
    for key in (
        "plan_hash", "dataset_lock_hash", "baseline_identity_hash", "native_identity_hash",
        "model_profile_hash", "model_resolution_lock_hash", "evaluator_hash", "oracle_hash",
    ):
        _digest(record[key], f"{name}.{key}")
    if production:
        _digest(record["gateway_relay_lock_hash"], f"{name}.gateway_relay_lock_hash")
    if int(record["max_input_tokens"]) <= 0 or int(record["max_output_tokens"]) <= 0:
        raise ValueError(f"{name} token caps must be positive")
    expected_oracle_status = "not_applicable" if r10_4 else "passed"
    if record.get("billing_status") != "known" or record.get("oracle_status") != expected_oracle_status:
        raise ValueError(f"{name} billing and oracle status do not match its runtime contract")
    if r10_4:
        if str(record.get("condition")) != "not_applicable":
            raise ValueError(f"{name} readiness condition must be not_applicable")
        if str(record.get("evaluation_scope")) != "readiness_transport":
            raise ValueError(f"{name} readiness evaluation scope is invalid")
        if bool(record.get("metric_eligible")):
            raise ValueError(f"{name} readiness artifact cannot be metric eligible")
    policy = record["retry_policy"]
    if not isinstance(policy, Mapping) or int(policy.get("max_attempts", 0)) < 1:
        raise ValueError(f"{name}.retry_policy is invalid")
    root = Path(artifact_root).resolve()
    paths = {
        "event_ledger": (record["event_ledger_path"], record["event_ledger_sha256"]),
        "proof": (record["proof_path"], record["proof_sha256"]),
        "usage": (record["usage_path"], record["usage_sha256"]),
        "cost": (record["cost_path"], record["cost_sha256"]),
        "evaluator": (record["evaluator_path"], record["evaluator_sha256"]),
        "cleanup": (record["cleanup_path"], record["cleanup_sha256"]),
    }
    if production:
        paths["invocation_ledger"] = (record["invocation_ledger_path"], record["invocation_ledger_sha256"])
    if r10_4:
        paths["oracle_applicability"] = (record["oracle_applicability_path"], record["oracle_applicability_sha256"])
    for label, (path, digest) in paths.items():
        rehash_artifact({"artifact_path": path, "artifact_sha256": digest}, artifact_root=root, name=f"{name}.{label}")
    if production:
        try:
            ledger = json.loads((root / _relative_artifact_path(record["invocation_ledger_path"], name)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} invocation ledger is not valid JSON") from exc
        if not isinstance(ledger, Mapping) or ledger.get("gateway_relay_lock_hash") != record["gateway_relay_lock_hash"]:
            raise ValueError(f"{name} invocation ledger relay binding mismatch")
        rows = [item for item in ledger.get("invocations", []) if isinstance(item, Mapping) and item.get("run_id") == record["run_id"]]
        if not rows:
            raise ValueError(f"{name} invocation ledger has no observed request")
        if r10_5:
            indices = [int(item.get("call_index", -1)) for item in rows]
            if record.get("invocation_ledger_hash") != record.get("invocation_ledger_sha256"):
                raise ValueError(f"{name} invocation ledger hash alias mismatch")
            if record.get("invocation_call_indices") != indices:
                raise ValueError(f"{name} invocation call indices differ from the ledger")
            if int(record.get("invocation_replay_count", -1)) != sum(int(item.get("replay_count", 0)) for item in rows):
                raise ValueError(f"{name} invocation replay count differs from the ledger")
            if int(record.get("invocation_response_count", -1)) != len(rows):
                raise ValueError(f"{name} invocation response count differs from the ledger")
            if len(rows) != 1 or indices != [0]:
                raise ValueError(f"{name} readiness must contain exactly call index 0")
    try:
        usage = json.loads((root / _relative_artifact_path(record["usage_path"], name)).read_text(encoding="utf-8"))
        cost = json.loads((root / _relative_artifact_path(record["cost_path"], name)).read_text(encoding="utf-8"))
        evaluator = json.loads((root / _relative_artifact_path(record["evaluator_path"], name)).read_text(encoding="utf-8"))
        cleanup = json.loads((root / _relative_artifact_path(record["cleanup_path"], name)).read_text(encoding="utf-8"))
        event_ledger = json.loads((root / _relative_artifact_path(record["event_ledger_path"], name)).read_text(encoding="utf-8"))
        proof = json.loads((root / _relative_artifact_path(record["proof_path"], name)).read_text(encoding="utf-8"))
        artifact = json.loads((root / _relative_artifact_path(record["artifact_path"], name)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} runtime evidence contains invalid JSON") from exc
    if not isinstance(usage, Mapping) or not all(str(key) in usage for key in ("input_tokens", "output_tokens", "total_tokens", "usd")):
        raise ValueError(f"{name} usage record lacks complete token/cost data")
    counts = [usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")]
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError(f"{name} usage token counts are abnormal")
    if not isinstance(usage["usd"], (int, float)) or not math.isfinite(float(usage["usd"])) or float(usage["usd"]) < 0:
        raise ValueError(f"{name} usage cost is abnormal")
    if not isinstance(cost, Mapping) or cost.get("billing_status") != "known":
        raise ValueError(f"{name} cost record is not known")
    if abs(float(cost.get("cost_usd", -1)) - float(usage["usd"])) > 1e-8:
        raise ValueError(f"{name} usage/cost records do not match")
    if not isinstance(evaluator, Mapping) or evaluator.get("status") != "passed":
        raise ValueError(f"{name} evaluator did not pass")
    if r10_4:
        try:
            applicability = json.loads((root / _relative_artifact_path(record["oracle_applicability_path"], name)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} oracle applicability is not valid JSON") from exc
        expected_applicability = {
            "schema_version": "1.0.0",
            "status": "not_applicable",
            "reason": "no_target_condition_in_readiness",
            "osr": None,
            "plan_hash": str(record["plan_hash"]),
            "run_id": str(record["run_id"]),
            "readiness_kind": str(record["readiness_kind"]),
        }
        if applicability != expected_applicability:
            raise ValueError(f"{name} oracle applicability is not the signed-plan deterministic N/A record")
    if not isinstance(cleanup, Mapping) or cleanup.get("success") is not True:
        raise ValueError(f"{name} cleanup did not pass")
    resources = cleanup.get("resources", {})
    if isinstance(resources, Mapping) and any(record.get("ids") for record in resources.values() if isinstance(record, Mapping)):
        raise ValueError(f"{name} cleanup left Docker resources")
    try:
        from src.pipeline.framework_adapter import RunArtifact
        run_artifact = RunArtifact.from_dict(artifact)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot load RunArtifact for trust binding") from exc
    if run_artifact.event_ledger_hash != canonical_hash(event_ledger):
        raise ValueError(f"{name} RunArtifact ledger hash mismatch")
    if run_artifact.proof_hash != canonical_hash(proof):
        raise ValueError(f"{name} RunArtifact proof hash mismatch")
    if any(float(run_artifact.usage[key]) != float(usage["usd"] if key == "total_usd" else usage[key])
           for key in ("input_tokens", "output_tokens", "total_tokens", "total_usd")):
        raise ValueError(f"{name} RunArtifact usage mismatch")
    if str(run_artifact.framework_identity.get("image_digest", "")) != str(record["image_digest"]):
        raise ValueError(f"{name} RunArtifact framework image mismatch")
    if production and str(run_artifact.run_context.get("gateway_relay_lock_hash")) != str(record["gateway_relay_lock_hash"]):
        raise ValueError(f"{name} RunArtifact relay lock mismatch")


def _validate_control(
    case: Mapping[str, Any], name: str, *, oracle_status: str,
    artifact_root: str | Path | None = None,
) -> None:
    docker = case.get("docker")
    oracle = case.get("oracle")
    if not isinstance(docker, Mapping) or not isinstance(oracle, Mapping):
        raise ValueError(f"{name} requires docker and oracle records")
    _passed(docker, f"{name}.docker", artifact_root=artifact_root)
    if not str(docker.get("image_digest", "")).startswith("sha256:"):
        raise ValueError(f"{name}.docker.image_digest must be an immutable image digest")
    if oracle.get("status") != oracle_status:
        raise ValueError(f"{name}.oracle must have status={oracle_status!r}")
    _digest(
        oracle.get("evidence_sha256", oracle.get("artifact_sha256")),
        f"{name}.oracle.evidence_sha256",
    )
    if artifact_root is not None:
        rehash_artifact(oracle, artifact_root=artifact_root, name=f"{name}.oracle")


def validate_smoke_evidence(
    evidence: Mapping[str, Any],
    *,
    base_case_ids: Sequence[str],
    model_labels: Sequence[str] = (),
    robustness_base_case_ids: Sequence[str] | None = None,
    mode: str,
    artifact_root: str | Path | None = None,
) -> dict[str, int]:
    """Validate dataset-freeze or pretrain evidence without fabricating results."""
    if mode not in {"dataset-freeze", "pretrain", "runtime-smoke"}:
        raise ValueError("smoke evidence mode must be dataset-freeze, runtime-smoke, or pretrain")
    if mode == "runtime-smoke":
        required_runtime = {
            "schema_version", "generated_at", "dataset_lock_hash", "dataset_evidence_hash",
            "vertex_canaries", "framework_model_smokes",
        }
        missing_runtime = sorted(required_runtime.difference(evidence))
        if missing_runtime:
            raise ValueError(f"runtime smoke evidence missing field(s): {', '.join(missing_runtime)}")
        strict_runtime_fields = {
            "plan_hash", "training_protocol_hash", "baseline_lock_hash",
            "model_resolution_lock_hash", "pricing_snapshot_hash", "approval_hash",
            "gateway_relay_lock_hash", "runtime_topology_evidence_path",
            "runtime_topology_evidence_sha256",
        }
        counter_fields = {
            "preflight_provider_calls", "preflight_vertex_calls",
            "paid_provider_responses", "paid_vertex_responses",
        }
        unexpected_runtime = sorted(set(evidence).difference(required_runtime | strict_runtime_fields | counter_fields))
        if unexpected_runtime:
            raise ValueError(f"runtime smoke evidence has unexpected field(s): {', '.join(unexpected_runtime)}")
        runtime_strict = evidence["schema_version"] in {LEGACY_RUNTIME_SCHEMA_VERSION, RUNTIME_SCHEMA_VERSION, R10_5_RUNTIME_SCHEMA_VERSION, R10_6_RUNTIME_SCHEMA_VERSION, R10_7_RUNTIME_SCHEMA_VERSION}
        production_runtime = evidence["schema_version"] in {RUNTIME_SCHEMA_VERSION, R10_5_RUNTIME_SCHEMA_VERSION, R10_6_RUNTIME_SCHEMA_VERSION, R10_7_RUNTIME_SCHEMA_VERSION}
        if evidence["schema_version"] not in {SCHEMA_VERSION, LEGACY_RUNTIME_SCHEMA_VERSION, RUNTIME_SCHEMA_VERSION, R10_5_RUNTIME_SCHEMA_VERSION, R10_6_RUNTIME_SCHEMA_VERSION, R10_7_RUNTIME_SCHEMA_VERSION} or not str(evidence["generated_at"]).endswith("Z"):
            raise ValueError("runtime smoke evidence schema or timestamp is invalid")
        if runtime_strict:
            strict_required = {
                "plan_hash", "training_protocol_hash", "baseline_lock_hash",
                "model_resolution_lock_hash", "pricing_snapshot_hash", "approval_hash",
            }
            missing_strict = sorted(strict_required.difference(evidence))
            if missing_strict:
                raise ValueError(f"runtime smoke evidence missing strict field(s): {', '.join(missing_strict)}")
            for key in strict_required:
                _digest(evidence[key], f"runtime.{key}")
        if production_runtime:
            for key in ("gateway_relay_lock_hash", "runtime_topology_evidence_sha256"):
                _digest(evidence.get(key), f"runtime.{key}")
            if artifact_root is None:
                raise ValueError("production runtime evidence requires artifact_root")
            topology_path = _relative_artifact_path(
                evidence.get("runtime_topology_evidence_path"), "runtime_topology_evidence"
            )
            topology_file = Path(artifact_root).resolve() / topology_path
            if not topology_file.is_file():
                raise ValueError("production runtime evidence requires runtime-topology-evidence.json")
            actual_topology_hash = hashlib.sha256(topology_file.read_bytes()).hexdigest()
            if actual_topology_hash != str(evidence["runtime_topology_evidence_sha256"]):
                raise ValueError("runtime topology evidence hash mismatch")
            try:
                topology = json.loads(topology_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("runtime topology evidence is not valid JSON") from exc
            if isinstance(topology, Mapping):
                from src.pipeline.runtime_topology import validate_runtime_topology_evidence
                try:
                    validate_runtime_topology_evidence(topology)
                except ValueError as exc:
                    raise ValueError("runtime topology lifecycle evidence is invalid") from exc
            if not isinstance(topology, Mapping) or topology.get("success") is not True:
                raise ValueError("runtime topology lifecycle did not succeed")
            if topology.get("gateway_relay_lock_hash") != evidence.get("gateway_relay_lock_hash"):
                raise ValueError("runtime topology relay lock binding mismatch")
        _digest(evidence["dataset_lock_hash"], "runtime.dataset_lock_hash")
        _digest(evidence["dataset_evidence_hash"], "runtime.dataset_evidence_hash")
        expected_models = set(model_labels)
        if len(expected_models) != VERTEX_CANARY_COUNT:
            raise ValueError("readiness requires exactly three locked model labels")
        canary_records = _records(evidence["vertex_canaries"], "Vertex canary", VERTEX_CANARY_COUNT)
        if {str(record.get("model_label", "")) for record in canary_records} != expected_models:
            raise ValueError("Vertex canaries must cover exactly the locked model labels")
        for record in canary_records:
            name = f"Vertex canary {record.get('model_label', '')}"
            if production_runtime and str(record.get("gateway_relay_lock_hash")) != str(evidence["gateway_relay_lock_hash"]):
                raise ValueError(f"{name} gateway relay lock binding mismatch")
            if runtime_strict:
                if artifact_root is None:
                    raise ValueError("strict runtime evidence requires artifact_root")
                _runtime_record(record, name, artifact_root=artifact_root, production=production_runtime)
            else:
                _passed(record, name, artifact_root=artifact_root)
            if not str(record.get("resource_id", "")).strip() or not str(record.get("resource_revision", "")).strip():
                raise ValueError("Vertex canary requires a pinned resource_id and resource_revision")
            try:
                validate_resolution_fields(
                    str(record.get("model_label", "")),
                    str(record.get("resource_revision", "")),
                    str(record.get("resolution_mode", "immutable")),
                    str(record.get("resolution_evidence_hash", "")),
                    str(record.get("resolution_resolved_at", "")),
                )
            except VertexContractError as exc:
                raise ValueError(str(exc)) from exc
        smoke_records = _records(evidence["framework_model_smokes"], "framework-model smoke", FRAMEWORK_MODEL_SMOKE_COUNT)
        expected_pairs = {(framework, model) for framework in FRAMEWORKS for model in expected_models}
        runtime_pairs: set[tuple[str, str]] = set()
        for record in smoke_records:
            framework = str(record.get("framework", ""))
            model = str(record.get("model_label", ""))
            pair = (framework, model)
            if not str(record.get("smoke_id", "")).strip() or pair in runtime_pairs:
                raise ValueError("framework-model smoke pairs must be unique and named")
            runtime_pairs.add(pair)
            name = f"framework-model smoke {framework}/{model}"
            if production_runtime and str(record.get("gateway_relay_lock_hash")) != str(evidence["gateway_relay_lock_hash"]):
                raise ValueError(f"{name} gateway relay lock binding mismatch")
            if runtime_strict:
                if artifact_root is None:
                    raise ValueError("strict runtime evidence requires artifact_root")
                _runtime_record(record, name, artifact_root=artifact_root, production=production_runtime)
            else:
                _passed(record, name, artifact_root=artifact_root)
        if runtime_pairs != expected_pairs:
            raise ValueError("framework-model smokes must cover exactly every framework/model pair")
        if evidence["schema_version"] in {R10_5_RUNTIME_SCHEMA_VERSION, R10_6_RUNTIME_SCHEMA_VERSION, R10_7_RUNTIME_SCHEMA_VERSION}:
            response_count = sum(int(record.get("invocation_response_count", -1)) for record in (*canary_records, *smoke_records))
            if response_count != 18 or int(evidence.get("paid_provider_responses", -1)) != response_count:
                raise ValueError("r10.5 readiness response count must be 18 observed ledger responses")
            if int(evidence.get("paid_vertex_responses", -1)) != response_count:
                raise ValueError("r10.5 readiness Vertex response count is not ledger-backed")
        return {
            "base_case_fixed_controls": 0, "robustness": 0,
            "vertex_canaries": len(canary_records), "framework_model_smokes": len(smoke_records),
        }
    required = {
        "schema_version", "generated_at", "base_case_fixed_controls", "robustness",
        "vertex_canaries", "framework_model_smokes",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError(f"smoke evidence missing field(s): {', '.join(missing)}")
    unexpected = sorted(set(evidence).difference(required))
    if unexpected:
        raise ValueError(f"smoke evidence has unexpected field(s): {', '.join(unexpected)}")
    if evidence["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"smoke evidence schema_version must be {SCHEMA_VERSION}")
    if not str(evidence["generated_at"]).endswith("Z"):
        raise ValueError("smoke evidence generated_at must be UTC")

    expected_cases = set(base_case_ids)
    if len(expected_cases) != BASE_CASE_COUNT:
        raise ValueError(f"readiness requires exactly {BASE_CASE_COUNT} locked base cases")
    controls = _records(evidence["base_case_fixed_controls"], "base-case fixed-control", BASE_CASE_COUNT)
    if {str(record.get("case_id", "")) for record in controls} != expected_cases:
        raise ValueError("base-case fixed-control evidence must cover exactly the locked base cases")
    for record in controls:
        case_id = str(record["case_id"])
        _validate_control(
            record.get("vulnerable", {}), f"{case_id}.vulnerable", oracle_status="passed",
            artifact_root=artifact_root,
        )
        _validate_control(
            record.get("fixed", {}), f"{case_id}.fixed", oracle_status="expected_failure",
            artifact_root=artifact_root,
        )

    robustness = _records(evidence["robustness"], "robustness", ROBUSTNESS_COUNT)
    allowed_bases = set(robustness_base_case_ids or base_case_ids)
    variant_ids: set[str] = set()
    bases: set[str] = set()
    strata: list[str] = []
    for record in robustness:
        variant_id = str(record.get("variant_id", ""))
        base_case_id = str(record.get("base_case_id", ""))
        if not variant_id or variant_id in variant_ids:
            raise ValueError("robustness variant_id must be present and unique")
        if not base_case_id or base_case_id in bases:
            raise ValueError("robustness base_case_id must be present and unique")
        if base_case_id not in allowed_bases:
            raise ValueError(f"robustness {variant_id} references an unlocked base case")
        if not str(record.get("transformation", "")).strip():
            raise ValueError(f"robustness {variant_id} requires a transformation")
        variant_ids.add(variant_id)
        bases.add(base_case_id)
        strata.append(str(record.get("stratum", "")))
        _passed(
            record.get("semantic_validity", {}), f"{variant_id}.semantic_validity",
            artifact_root=artifact_root,
        )
        _passed(record.get("smoke", {}), f"{variant_id}.smoke", artifact_root=artifact_root)
    if set(strata) != ROBUSTNESS_STRATA or any(strata.count(stratum) != 3 for stratum in ROBUSTNESS_STRATA):
        raise ValueError("robustness evidence requires three records per required stratum")

    canaries = evidence["vertex_canaries"]
    smokes = evidence["framework_model_smokes"]
    if mode == "dataset-freeze":
        if canaries != [] or smokes != []:
            raise ValueError("dataset-freeze evidence requires empty runtime evidence sections")
        return {
            "base_case_fixed_controls": len(controls),
            "robustness": len(robustness),
            "vertex_canaries": 0,
            "framework_model_smokes": 0,
        }

    expected_models = set(model_labels)
    if len(expected_models) != VERTEX_CANARY_COUNT:
        raise ValueError("readiness requires exactly three locked model labels")
    canary_records = _records(canaries, "Vertex canary", VERTEX_CANARY_COUNT)
    if {str(record.get("model_label", "")) for record in canary_records} != expected_models:
        raise ValueError("Vertex canaries must cover exactly the locked model labels")
    for record in canary_records:
        _passed(record, f"Vertex canary {record.get('model_label', '')}", artifact_root=artifact_root)
        if not str(record.get("resource_id", "")).strip() or not str(record.get("resource_revision", "")).strip():
            raise ValueError("Vertex canary requires a pinned resource_id and resource_revision")
        try:
            validate_resolution_fields(
                str(record.get("model_label", "")),
                str(record.get("resource_revision", "")),
                str(record.get("resolution_mode", "immutable")),
                str(record.get("resolution_evidence_hash", "")),
                str(record.get("resolution_resolved_at", "")),
            )
        except VertexContractError as exc:
            raise ValueError(str(exc)) from exc

    smoke_records = _records(smokes, "framework-model smoke", FRAMEWORK_MODEL_SMOKE_COUNT)
    expected_pairs = {(framework, model) for framework in FRAMEWORKS for model in expected_models}
    pairs: set[tuple[str, str]] = set()
    for record in smoke_records:
        framework = str(record.get("framework", ""))
        model = str(record.get("model_label", ""))
        if not str(record.get("smoke_id", "")).strip():
            raise ValueError("framework-model smoke requires smoke_id")
        pair = (framework, model)
        if pair in pairs:
            raise ValueError("framework-model smoke pairs must be unique")
        pairs.add(pair)
        _passed(record, f"framework-model smoke {framework}/{model}", artifact_root=artifact_root)
    if pairs != expected_pairs:
        raise ValueError("framework-model smokes must cover exactly every framework/model pair")
    return {
        "base_case_fixed_controls": len(controls),
        "robustness": len(robustness),
        "vertex_canaries": len(canary_records),
        "framework_model_smokes": len(smoke_records),
    }
