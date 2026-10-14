#!/usr/bin/env python3
"""
main.py — AuNePT CLI entry point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage examples
──────────────
# Full automated run against one authorized target
python main.py run --target 10.0.0.1 --port 8080 [--attacker 10.0.0.2]

# Resume a previous run
python main.py run --target 10.0.0.1 --port 8080 --thread-id my-run-1

# Reconnaissance only (bounded, LLM-free)
python main.py recon --target 10.0.0.1 --port 8080

# Dataset evaluation run
python main.py run-dataset --tasks data/dataset/tasks.json --case-id case-001 \\
    --benchmark-cve-cache data/dataset/benchmark_cve.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import dotenv

dotenv.load_dotenv()

# ── Path setup (keep working from project root) ───────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.graph import build_graph  # noqa: E402
from src.state import PentestState, initial_state  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.metrics_collector import MetricsCollector  # noqa: E402
from src.utils.structured_logger import get_structured_logger  # noqa: E402

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = RuntimeError

logger = setup_logging("aunept")
slog = get_structured_logger()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_banner():
    print("\n" + "═" * 60)
    print("  🛡  AuNePT  —  LangGraph Orchestrated")
    print("═" * 60 + "\n")


def _apply_runtime_args(state: PentestState, args) -> None:
    minutes = int(getattr(args, "max_runtime_minutes", 0) or 0)
    state["max_runtime_seconds"] = max(minutes, 0) * 60


def _final_stop_reason(state: dict) -> str:
    summary = str(state.get("execution_summary") or "").strip()
    if summary:
        return summary
    vlog = list(state.get("verification_log", []) or [])
    if not vlog:
        return ""
    return str(vlog[-1].get("reason", "") or "").strip()


def _print_metrics_summary(metrics_path: str, metrics: dict, final_state: dict) -> None:
    print(f"\n  📊 Metrics exported → {metrics_path}")
    print(f"     Tokens: {metrics.get('M8_tokens_total', 0)} | "
          f"LLM calls: {metrics.get('M7_total_llm_requests', 0)} | "
          f"Success: {'✅' if metrics.get('M1_osr') else '❌'}")
    if not metrics.get("M1_osr"):
        reason = _final_stop_reason(final_state)
        if reason:
            print(f"     Stop reason: {reason}")


def _stream_and_print(graph, state: PentestState, config: dict, skip_phases: list[str] | None = None):
    """
    Stream graph events, pretty-print each node output.
    """
    skip_phases = skip_phases or []

    for event in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            step_r = node_output.get("recon_step_count")
            step_e = node_output.get("execution_step_count")

            step_info = ""
            if step_r is not None:
                step_info = f" [recon step {step_r}]"
            if step_e is not None:
                step_info = f" [exec step {step_e}]"

            print(f"\n{'─'*50}")
            print(f"  Node: {node_name}{step_info}")
            print(f"{'─'*50}")

            if node_output.get("recon_complete"):
                print("  ✓ Recon complete")
                svc = node_output.get("port_services", {})
                for port, info in svc.items():
                    print(f"    Port {port}: {info.get('name','?')} {info.get('version','')}")

            if node_output.get("planning_complete"):
                print("  ✓ Planning complete")
                plan = node_output.get("exploit_plan", [])
                print(f"  Exploit plan ({len(plan)} entries):")
                for i, ex in enumerate(plan[:5], 1):
                    print(f"    {i}. {ex.get('name','?')} (score={ex.get('score','?')})")

            if node_output.get("execution_summary"):
                success = node_output.get("execution_success", False)
                icon = "✅" if success else "❌"
                print(f"  {icon} {node_output['execution_summary']}")


# ── Sub-commands ──────────────────────────────────────────────────────────────

def _initial_state_from_args(args) -> PentestState:
    """Build the initial state for a ``run``/``recon`` CLI invocation.

    The target port is threaded into the state so the runtime contract
    (``build_contract`` / the equivalent resolved at pipeline entry) carries an
    authorized port; without it ``assert_port`` would raise a usage ``ValueError``
    inside reconnaissance instead of the parser rejecting the call.
    """
    port = getattr(args, "port", None)
    return initial_state(
        target_ip=args.target,
        target_port=str(port) if port not in (None, "") else "",
        attacker_ip=getattr(args, "attacker", "") or "",
        recon_max_steps=getattr(args, "recon_steps", 12),
        execution_max_steps=getattr(args, "exec_steps", 30),
    )


def cmd_run(args):
    """Full Recon → Planning → Execution pipeline."""
    _print_banner()
    thread_id = args.thread_id or str(uuid.uuid4())
    graph, config = build_graph(thread_id=thread_id)

    print(f"  Thread ID  : {thread_id}  (use --thread-id to resume)")
    print(f"  Target     : {args.target}:{args.port}")
    print(f"  Attacker   : {args.attacker or 'auto-detect'}\n")

    t_start = time.time()
    state = _initial_state_from_args(args)
    state["run_start_time"] = t_start
    state["phase_timestamps"] = {"run_start": t_start}
    _apply_runtime_args(state, args)

    try:
        _stream_and_print(graph, state, config)
    except GraphRecursionError:
        logger.error("Graph hit recursion limit (possible infinite loop).")
        try:
            final = graph.get_state(config)
            if final and final.values:
                vlog = list(final.values.get("verification_log", []))
                last_reason = vlog[-1].get("reason", "unknown") if vlog else "no verification log"
                print(f"\n  ERROR: Recursion limit reached. Last reason: {last_reason}")
                final_state = dict(final.values)
                final_state["run_end_time"] = time.time()
                runs_dir = os.path.join(_ROOT, "data", "runs")
                os.makedirs(runs_dir, exist_ok=True)
                metrics_path = os.path.join(runs_dir, f"{thread_id}-metrics.json")
                gt_path = getattr(args, "ground_truth", None)
                collector = MetricsCollector(final_state, ground_truth_path=gt_path)
                collector.export(metrics_path)
        except Exception as export_exc:
            logger.warning("Post-crash metrics export failed: %s", export_exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Run failed: %s", exc)
        sys.exit(1)

    # ── Metrics export ────────────────────────────────────────────────────────
    try:
        final = graph.get_state(config)
        if final and final.values:
            final_state = dict(final.values)
            final_state["run_end_time"] = time.time()
            runs_dir = os.path.join(_ROOT, "data", "runs")
            os.makedirs(runs_dir, exist_ok=True)
            metrics_path = os.path.join(runs_dir, f"{thread_id}-metrics.json")
            gt_path = getattr(args, "ground_truth", None)
            collector = MetricsCollector(final_state, ground_truth_path=gt_path)
            metrics = collector.export(metrics_path)
            slog.run_summary(final_state)
            _print_metrics_summary(metrics_path, metrics, final_state)
    except Exception as exc:
        logger.warning("Metrics export failed (non-fatal): %s", exc)


def cmd_recon(args):
    """Recon only — stops before planning."""
    _print_banner()
    from src.agents.recon import recon_node

    state = _initial_state_from_args(args)
    # Run recon loop manually (no graph needed for single-phase)
    from src.agents.recon import route_recon
    while True:
        updates = recon_node(state)
        state.update(updates)
        if route_recon(state) == "planning":
            break

    print("\n  Final recon output:")
    print("  Observations:")
    print(json.dumps(state.get("recon_observations", []), indent=2))
    print("  World state:")
    print(json.dumps(state.get("world_state", {}), indent=2))


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_dataset_task(tasks_path: str, case_id: str) -> dict:
    """Load a single case from a dataset tasks JSON generated by the adapter."""
    with open(tasks_path, encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data if isinstance(data, list) else data.get("tasks", [])
    task = next((t for t in tasks if t.get("case_id") == case_id), None)
    if task is None:
        raise SystemExit(f"Case {case_id} not found in {tasks_path}")
    return task


def cmd_run_dataset(args):
    """Run against a curated dataset case."""
    _print_banner()
    task = load_dataset_task(args.tasks, args.case_id)

    thread_id = f"dataset-{args.case_id}"
    graph, config = build_graph(thread_id=thread_id)

    print(f"  Case ID    : {args.case_id}")
    print(f"  Target     : {task.get('target', '?')}")
    print(f"  Objective  : {task.get('objective', task.get('name', '?'))}\n")

    t_start = time.time()
    state = initial_state(
        target_ip=task["target"],
        target_port=str(task.get("port", "")),
        attacker_ip=args.attacker or task.get("attacker_ip", ""),
        recon_max_steps=args.recon_steps,
        execution_max_steps=args.exec_steps,
    )
    state["run_start_time"] = t_start
    state["phase_timestamps"] = {"run_start": t_start, "case_id": args.case_id}
    state["dataset_mode"] = "curated"
    state["dataset_case_id"] = args.case_id
    state["benchmark_cve_cache_path"] = args.benchmark_cve_cache
    if task.get("service_hint"):
        state["app_name"] = task["service_hint"]
    if task.get("version_hint"):
        state["app_version"] = task["version_hint"]
    _apply_runtime_args(state, args)

    try:
        _stream_and_print(graph, state, config)
    except GraphRecursionError:
        logger.error("Graph hit recursion limit (possible infinite loop).")
        try:
            final = graph.get_state(config)
            if final and final.values:
                vlog = list(final.values.get("verification_log", []))
                last_reason = vlog[-1].get("reason", "unknown") if vlog else "no verification log"
                print(f"\n  ERROR: Recursion limit reached. Last reason: {last_reason}")
                final_state = dict(final.values)
                final_state["run_end_time"] = time.time()
                runs_dir = os.path.join(_ROOT, "data", "runs")
                os.makedirs(runs_dir, exist_ok=True)
                metrics_path = os.path.join(runs_dir, f"{thread_id}-metrics.json")
                gt_path = getattr(args, "ground_truth", None)
                collector = MetricsCollector(final_state, ground_truth_path=gt_path)
                collector.export(metrics_path)
        except Exception as export_exc:
            logger.warning("Post-crash metrics export failed: %s", export_exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Run failed: %s", exc)
        sys.exit(1)

    try:
        final = graph.get_state(config)
        if final and final.values:
            final_state = dict(final.values)
            final_state["run_end_time"] = time.time()
            runs_dir = os.path.join(_ROOT, "data", "runs")
            os.makedirs(runs_dir, exist_ok=True)
            metrics_path = os.path.join(runs_dir, f"{thread_id}-metrics.json")
            gt_path = getattr(args, "ground_truth", None)
            collector = MetricsCollector(final_state, ground_truth_path=gt_path)
            metrics = collector.export(metrics_path)
            slog.run_summary(final_state)
            _print_metrics_summary(metrics_path, metrics, final_state)
    except Exception as exc:
        logger.warning("Metrics export failed (non-fatal): %s", exc)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aunept",
        description="LangGraph-orchestrated LLM penetration testing agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Full pipeline: recon → planning → execution")
    p_run.add_argument("--target", required=True, help="Target IP address")
    p_run.add_argument("--port", type=int, required=True, help="Authorized target port")
    p_run.add_argument("--attacker", default="", help="Attacker IP (optional)")
    p_run.add_argument("--thread-id", default="", help="Resume an existing run by thread ID")
    p_run.add_argument("--recon-steps", type=int, default=12)
    p_run.add_argument("--exec-steps", type=int, default=30)
    p_run.add_argument(
        "--max-runtime-minutes",
        type=int,
        default=60,
        help="Structural runtime limit in minutes; 0 disables the limit",
    )
    p_run.add_argument(
        "--ground-truth",
        default="",
        help="Path to ground-truth JSON for M3/M4 metric calculation",
    )
    p_run.set_defaults(func=cmd_run)

    # ── recon ─────────────────────────────────────────────────────────────────
    p_rec = sub.add_parser("recon", help="Reconnaissance only")
    p_rec.add_argument("--target", required=True)
    p_rec.add_argument("--port", type=int, required=True, help="Authorized target port")
    p_rec.add_argument("--recon-steps", type=int, default=12)
    p_rec.set_defaults(func=cmd_recon)

    # ── run-dataset ───────────────────────────────────────────────────────────
    p_ds = sub.add_parser("run-dataset", help="Evaluate on curated dataset case")
    p_ds.add_argument("--tasks", required=True, help="Path to tasks JSON from dataset adapter")
    p_ds.add_argument("--case-id", required=True, help="Dataset case identifier")
    p_ds.add_argument("--benchmark-cve-cache", required=True, help="Path to curated benchmark CVE JSON")
    p_ds.add_argument("--attacker", default="", help="Attacker IP (optional)")
    p_ds.add_argument("--recon-steps", type=int, default=12)
    p_ds.add_argument("--exec-steps", type=int, default=30)
    p_ds.add_argument(
        "--max-runtime-minutes",
        type=int,
        default=60,
        help="Structural runtime limit in minutes; 0 disables the limit",
    )
    p_ds.add_argument(
        "--ground-truth",
        default="",
        help="Path to ground-truth JSON for M3/M4 metric calculation",
    )
    p_ds.set_defaults(func=cmd_run_dataset)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
