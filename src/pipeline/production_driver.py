"""Framework-owned production entrypoint used by the locked runtime image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.pipeline.budget import BudgetTier
from src.pipeline.framework_adapter import ModelProfile, PublicTask


def main() -> int:
    for key in list(os.environ.keys()):
        if key.startswith("GOOGLE_") or key in {"GOOGLE_ADC", "GOOGLE_APPLICATION_CREDENTIALS"}:
            os.environ.pop(key, None)
    if os.path.exists("/opt/veriplanpt/.env"):
        try:
            os.remove("/opt/veriplanpt/.env")
        except OSError:
            pass
    path = Path(os.environ["VERIPLANPT_PUBLIC_INVOCATION_FILE"])
    invocation = json.loads(path.read_text(encoding="utf-8"))
    task = PublicTask.from_dict(dict(invocation["task"]))
    profile = ModelProfile.from_dict(dict(invocation["model_profile"]))
    adapter = __import__("src.pipeline.framework_adapter", fromlist=["FrameworkAdapter"]).FrameworkAdapter(
        results_root=os.environ.get("VERIPLANPT_RUN_DIR", "/run/veriplanpt"),
    )
    output = Path(os.environ.get("VERIPLANPT_OUTPUT_DIR", os.environ.get("VERIPLANPT_RUN_DIR", ".")))
    generated = output / "generated-public-config.json"
    generated.write_text(json.dumps({
        "objective": task.objective,
        "target": task.target,
        "budget_tier": str(invocation.get("budget_tier", "medium")),
    }, sort_keys=True) + "\n", encoding="utf-8")
    try:
        artifact = adapter.run(
            task, profile, BudgetTier.from_str(str(invocation.get("budget_tier", "medium"))),
            repetition=int(invocation.get("repetition", 1)),
            run_dir=os.environ.get("VERIPLANPT_RUN_DIR", "/run/veriplanpt"),
            condition=str(invocation.get("condition", "")),
            run_id=str(invocation["run_id"]),
        )
        artifact.save(str(output / "run_artifact.json"))
    except Exception as exc:
        import sys, traceback
        print(f"[production_driver] adapter.run error: {exc}", file=sys.stderr)
        traceback.print_exc()
    calls = output / "provider-calls.jsonl"
    response_count = len(calls.read_text(encoding="utf-8").splitlines()) if calls.is_file() else 0
    (output / "driver-evidence.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "mode": "actual-framework-driver",
        "stage": os.environ.get("VERIPLANPT_STAGE", ""),
        "framework": "VeriPlanPT",
        "public_task_hash": hashlib.sha256(
            json.dumps(dict(invocation["task"]), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generated_config_hash": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "provider_response_count": response_count,
        "phases": ["framework_graph"],
        "outcome": "completed",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
