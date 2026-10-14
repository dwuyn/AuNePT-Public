"""
src/utils/metrics_collector.py
────────────────────────────────
Metrics aggregation and export.

Reads the final PentestState after a run and computes all 15 evaluation
metrics defined in improve.txt section 6.

Single-run export:
    collector = MetricsCollector(state, ground_truth_path="gt.json")
    collector.export("runs/run-001-metrics.json")

Aggregate across multiple runs:
    MetricsCollector.aggregate_runs(["run-001-metrics.json", ...], "summary.json")

Ground-truth JSON schema (optional):
{
  "target_ip": "10.0.0.1",
  "correct_services": {
    "80": {"name": "apache", "version": "2.4.49"},
    "22": {"name": "openssh", "version": "8.2"}
  },
  "correct_cves": ["CVE-2021-41773", "CVE-2021-42013"],
  "flag": "HTB{some_flag}",
  "notes": "VulHub lab: Apache path traversal"
}
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from src.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


# ── Ground-truth schema default ───────────────────────────────────────────────

GROUND_TRUTH_SCHEMA: dict = {
    "target_ip": "",
    "correct_services": {},   # {port: {name, version}}
    "correct_cves": [],       # list of correct CVE IDs
    "flag": "",               # CTF flag string (optional)
    "notes": "",
}


class MetricsCollector:
    """
    Computes the 15 evaluation metrics from a finished PentestState dict.

    Parameters
    ----------
    state : dict
        The final PentestState (serialised via dict()).
    ground_truth_path : str, optional
        Path to a JSON file with ground-truth data for M3/M4 computation.
    token_price_per_1k : float
        USD cost per 1 000 tokens (input+output combined). Defaults to
        DeepSeek V3 pricing (~$0.0014/1k as reference).
    """

    def __init__(
        self,
        state: dict,
        ground_truth_path: Optional[str] = None,
        token_price_per_1k: float = 0.0014,
    ) -> None:
        self._state = state
        self._gt: dict = self._load_gt(ground_truth_path)
        self._price_per_1k = token_price_per_1k
        self._em = EpisodicMemory.from_list(state.get("episodic_memory", []))

    # ── Ground truth ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_gt(path: Optional[str]) -> dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load ground truth from %s: %s", path, exc)
            return {}

    @staticmethod
    def write_ground_truth_template(path: str, target_ip: str = "") -> None:
        """Write an empty ground-truth template for a new target."""
        template = dict(GROUND_TRUTH_SCHEMA)
        template["target_ip"] = target_ip
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(template, f, indent=2)
        logger.info("Ground-truth template written to %s", path)

    # ── Individual metric computations ────────────────────────────────────────

    def m1_osr(self) -> int:
        """M1: Overall Success Rate — 1 if execution succeeded, 0 otherwise."""
        return 1 if self._state.get("execution_success", False) else 0

    def m2_ssr(self) -> dict[str, int]:
        """
        M2: Step-wise Success Rate flags.
        Returns a dict of phase → 1/0.
        """
        return {
            "recon": 1 if self._state.get("recon_complete", False) else 0,
            "hypothesis": 1 if self._state.get("hypothesis_complete", False) else 0,
            "planning": 1 if self._state.get("planning_complete", False) else 0,
            "execution": 1 if self._state.get("execution_success", False) else 0,
        }

    def m3_service_id_accuracy(self) -> Optional[float]:
        """
        M3: Service Identification Accuracy.
        Requires ground truth.  Returns None if no ground truth provided.
        """
        correct_services = self._gt.get("correct_services", {})
        if not correct_services:
            return None
        port_services = self._state.get("port_services", {})
        correct = 0
        total = len(correct_services)
        for port, gt_info in correct_services.items():
            found = port_services.get(str(port), port_services.get(int(port), {}))
            if not found:
                continue
            gt_name = gt_info.get("name", "").lower()
            gt_ver = gt_info.get("version", "").lower()
            found_name = found.get("name", "").lower()
            found_ver = found.get("version", "").lower()
            name_ok = gt_name and gt_name in found_name
            # Version: accept if ground truth version is a prefix of found version
            ver_ok = (not gt_ver) or found_ver.startswith(gt_ver) or gt_ver in found_ver
            if name_ok and ver_ok:
                correct += 1
        return round(correct / max(total, 1), 4)

    def m4_correct_cve_at_k(self, k: int = 5) -> Optional[float]:
        """
        M4: Correct-CVE@k.
        Fraction of targets where correct CVE appears in top-k proposed CVEs.
        Returns None if no ground truth.
        """
        correct_cves = self._gt.get("correct_cves", [])
        if not correct_cves:
            return None
        proposed = self._state.get("cve_list", [])[:k]
        hit = any(c in proposed for c in correct_cves)
        return 1.0 if hit else 0.0

    def m5_exploit_applicability_precision(self) -> Optional[float]:
        """
        M5: Exploit Applicability Precision.
        Uses scoring data: exploits marked 'easy' exploitability vs total proposed.
        Returns None if no exploit plan generated.
        """
        exploit_plan = self._state.get("exploit_plan", [])
        if not exploit_plan:
            return None
        applicable = sum(
            1 for ex in exploit_plan
            if ex.get("exploitability", "") in ("easy", "medium")
        )
        return round(applicable / max(len(exploit_plan), 1), 4)

    def m6_attack_path_efficiency(self) -> Optional[float]:
        """
        M6: Attack Path Efficiency = useful_steps / total_steps.
        Useful steps: steps that produced non-empty output and outcome != 'error'/'blocked'.
        """
        episodes = self._em._episodes
        if not episodes:
            return None
        useful = sum(
            1 for ep in episodes
            if ep.outcome in ("success",) and ep.output_summary
        )
        return round(useful / max(len(episodes), 1), 4)

    def m7_mean_llm_requests(self) -> int:
        """M7: Total LLM requests in this run."""
        return self._state.get("total_llm_requests", 0)

    def m8_token_consumption(self) -> dict[str, int]:
        """M8: Token consumption breakdown."""
        total_in = self._state.get("total_tokens_in", 0)
        total_out = self._state.get("total_tokens_out", 0)
        # Backward compat: if only total_tokens is set, split equally
        total = self._state.get("total_tokens", 0)
        if not total_in and not total_out and total:
            total_in = total // 2
            total_out = total - total_in
        return {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
        }

    def m9_cost_per_target(self) -> float:
        """M9: Cost per target in USD."""
        tokens = self.m8_token_consumption()
        total_tokens = tokens["total_tokens"]
        return round(total_tokens / 1000 * self._price_per_1k, 6)

    def m10_cost_per_success(self) -> Optional[float]:
        """M10: Cost per successful target. None if this run failed."""
        if not self.m1_osr():
            return None
        return self.m9_cost_per_target()

    def m11_time_to_access(self) -> Optional[float]:
        """M11: Time-to-initial-access in seconds. None if not tracked."""
        timestamps = self._state.get("phase_timestamps", {})
        start = self._state.get("run_start_time") or timestamps.get("run_start")
        # Time to first execution success
        end = timestamps.get("execution_success_time") or self._state.get("run_end_time")
        if start and end and end > start:
            return round(end - start, 2)
        return None

    def m12_invalid_command_rate(self) -> Optional[float]:
        """M12: Invalid Command Rate = blocked_commands / total_commands."""
        total_cmds = self._em.total_steps()
        if not total_cmds:
            return None
        invalid = self._state.get("total_invalid_commands", self._em.count_invalid_commands())
        return round(invalid / total_cmds, 4)

    def m13_repeated_action_rate(self) -> Optional[float]:
        """M13: Repeated Action Rate = repeated_actions / total_actions."""
        total = self._em.total_steps()
        if not total:
            return None
        repeats = self._state.get("total_repeated_actions", self._em.count_repeats())
        return round(repeats / total, 4)

    def m14_hallucination_failure_rate(self) -> Optional[float]:
        """
        M14: Hallucination-induced Failure Rate (heuristic).
        Estimate: (invalid commands + repeat-induced failures) / total failures.
        """
        errors = self._em.count_errors()
        if not errors:
            return None
        invalid = self._state.get("total_invalid_commands", self._em.count_invalid_commands())
        repeats = self._em.count_repeats()
        hallucination_estimate = invalid + (repeats // 2)  # repeats partially due to hallucination
        return round(min(hallucination_estimate / max(errors, 1), 1.0), 4)

    def m15_recovery_rate(self) -> Optional[float]:
        """M15: Recovery Rate = recoveries / total_errors."""
        errors = self._em.count_errors()
        if not errors:
            return None
        recoveries = self._em.count_recoveries()
        return round(recoveries / errors, 4)

    def resource_usage(self) -> dict[str, Any]:
        """Plain resource usage metrics report for this run."""
        start = float(self._state.get("run_start_time", 0.0) or 0.0)
        end = float(self._state.get("run_end_time", 0.0) or 0.0)
        elapsed = round(max(end - start, 0.0), 2) if start > 0 and end > 0 else 0.0
        tokens = self.m8_token_consumption()
        return {
            "tokens": {
                "input": tokens["input_tokens"],
                "output": tokens["output_tokens"],
                "total": tokens["total_tokens"],
            },
            "llm_requests": int(self._state.get("total_llm_requests", 0) or 0),
            "retries_attempted": int(self._state.get("retry_spent", 0) or 0),
            "repeated_actions": int(self._state.get("total_repeated_actions", 0) or 0),
            "invalid_commands": int(self._state.get("total_invalid_commands", 0) or 0),
            "elapsed_time_sec": elapsed,
        }

    # ── Full metrics dict ─────────────────────────────────────────────────────

    def compute_all(self, k: int = 5) -> dict[str, Any]:
        """Compute all 15 metrics and return as a single dict."""
        tokens = self.m8_token_consumption()
        retrieval_bundle = self._state.get("retrieval_bundle", {}) or {}
        shortlist = list(retrieval_bundle.get("shortlist", []))
        assessments = list(retrieval_bundle.get("assessments", []))
        poc_candidates = list(retrieval_bundle.get("poc_candidates", []))
        resource_usage = self.resource_usage()
        return {
            "target_ip": self._state.get("target_ip", ""),
            "timestamp": time.time(),
            # Efficacy
            "M1_osr": self.m1_osr(),
            "M2_ssr": self.m2_ssr(),
            # Decision quality
            "M3_service_id_accuracy": self.m3_service_id_accuracy(),
            f"M4_correct_cve_at_{k}": self.m4_correct_cve_at_k(k),
            "M5_exploit_applicability_precision": self.m5_exploit_applicability_precision(),
            "M6_attack_path_efficiency": self.m6_attack_path_efficiency(),
            # Efficiency
            "M7_total_llm_requests": self.m7_mean_llm_requests(),
            "M8_tokens_in": tokens["input_tokens"],
            "M8_tokens_out": tokens["output_tokens"],
            "M8_tokens_total": tokens["total_tokens"],
            "M9_cost_usd": self.m9_cost_per_target(),
            "M10_cost_per_success_usd": self.m10_cost_per_success(),
            "M11_time_to_access_sec": self.m11_time_to_access(),
            # Reliability
            "M12_invalid_command_rate": self.m12_invalid_command_rate(),
            "M13_repeated_action_rate": self.m13_repeated_action_rate(),
            "M14_hallucination_failure_rate": self.m14_hallucination_failure_rate(),
            "M15_recovery_rate": self.m15_recovery_rate(),
            "retrieval_authoritative_candidates": len(retrieval_bundle.get("authoritative_records", [])),
            "retrieval_poc_candidates": len(poc_candidates),
            "retrieval_google_fallback_used": 1 if any(item.get("source") == "google" for item in poc_candidates) else 0,
            "retrieval_confirmed_candidates": sum(1 for item in assessments if item.get("verdict") == "confirmed"),
            "resource_usage": resource_usage,
            # Raw counts for debugging
            "_raw": {
                "recon_steps": self._state.get("recon_step_count", 0),
                "exec_steps": self._state.get("execution_step_count", 0),
                "verifier_blocks": self._state.get("verifier_blocks", 0),
                "debate_rounds": self._state.get("debate_round", 0),
                "retrieval_shortlist": len(shortlist),
                "total_episodes": self._em.total_steps(),
                "total_errors": self._em.count_errors(),
                "total_recoveries": self._em.count_recoveries(),
            },
        }

    def export(self, path: str, k: int = 5) -> dict[str, Any]:
        """Compute all metrics and write to a JSON file. Returns the metrics dict."""
        metrics = self.compute_all(k=k)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics exported to %s", path)
        return metrics

    # ── Aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def aggregate_runs(metric_paths: list[str], output_path: str) -> dict[str, Any]:
        """
        Aggregate metrics across multiple runs and export a summary.

        Computes mean ± stddev for each numeric metric across all runs.
        Counts success/failure for binary metrics.
        """
        import statistics

        runs = []
        for p in metric_paths:
            try:
                with open(p) as f:
                    runs.append(json.load(f))
            except Exception as exc:
                logger.warning("Could not load metrics from %s: %s", p, exc)

        if not runs:
            logger.error("No valid metric files found.")
            return {}

        numeric_keys = [
            "M1_osr", "M7_total_llm_requests",
            "M8_tokens_in", "M8_tokens_out", "M8_tokens_total",
            "M9_cost_usd", "M11_time_to_access_sec",
            "M12_invalid_command_rate", "M13_repeated_action_rate",
            "M14_hallucination_failure_rate", "M15_recovery_rate",
            "M3_service_id_accuracy", "M5_exploit_applicability_precision",
            "M6_attack_path_efficiency",
        ]

        summary: dict[str, Any] = {
            "n_runs": len(runs),
            "timestamp": time.time(),
        }

        for key in numeric_keys:
            values = [r[key] for r in runs if r.get(key) is not None]
            if not values:
                summary[key] = {"n": 0, "mean": None, "std": None}
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[key] = {
                "n": len(values),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        # M2 SSR aggregation
        for phase in ("recon", "hypothesis", "planning", "execution"):
            vals = [r.get("M2_ssr", {}).get(phase, 0) for r in runs]
            summary[f"M2_ssr_{phase}"] = round(sum(vals) / len(vals), 4) if vals else None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Aggregated metrics for %d runs → %s", len(runs), output_path)
        return summary
