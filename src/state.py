"""
src/state.py
────────────
Shared PentestState that flows through every node of the LangGraph graph.
All inter-agent communication happens via this TypedDict — no files, no
ad-hoc imports of one agent inside another.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Mapping, Optional, TypedDict

from langgraph.graph.message import add_messages


class PentestState(TypedDict, total=False):
    # ── Target ────────────────────────────────────────────────────────────────
    target_ip: str
    target_port: Optional[str]
    attacker_ip: Optional[str]

    # ── Phase control ─────────────────────────────────────────────────────────
    # "recon" | "hypothesis" | "planning" | "execution" | "done"
    current_phase: str
    phase2_route: str
    error_count: int
    last_error: Optional[str]

    # ── Recon outputs ─────────────────────────────────────────────────────────
    recon_complete: bool
    # {port: {"name": str, "version": str, "accessibility": str}}
    port_services: dict
    os_info: Optional[str]
    recon_step_count: int
    recon_max_steps: int
    target_services: list
    current_service_index: int

    # ── Planning inputs (derived from recon) ──────────────────────────────────
    keyword: str        # CVE ID or app keyword to search
    app_name: str
    app_version: str
    vuln_type: str

    # ── Planning outputs ──────────────────────────────────────────────────────
    cve_list: list        # list[str] of CVE IDs
    # list[dict] — each has: file_path, name, score, source
    exploit_plan: list
    selected_exploit: Optional[dict]
    planning_output_dir: Optional[str]
    planning_complete: bool

    # ── Debate Tracking ───────────────────────────────────────────────────────
    current_proposal: Optional[dict]
    debate_history: list
    debate_round: int

    # ── Execution outputs ─────────────────────────────────────────────────────
    doc_dir: Optional[str]
    execution_step_count: int
    execution_max_steps: int
    execution_success: bool
    execution_summary: Optional[str]
    execution_tracker: dict

    # ── Structured Memory (Layer 1) ───────────────────────────────────────────
    world_state: dict          # serialized WorldState (JSON-safe dict)
    episodic_memory: list      # serialized list[Episode]
    decision_memory: list      # serialized list[Decision]

    # ── Hypothesis outputs (Layer 3) ──────────────────────────────────────────
    retrieval_bundle: dict     # serialized RetrievalBundle
    vuln_hypotheses: list      # list[dict] — serialized VulnHypothesis
    hypothesis_complete: bool
    hypothesis_rework_count: int
    replan_count: int
    replan_max: int
    attempted_cves: list
    attempted_services: list

    # ── Verification tracking (Layer 2) ───────────────────────────────────────
    verification_log: list     # list of verifier verdicts
    verifier_blocks: int       # count of times verifier sent agent back (legacy; prefer per-phase counters)
    recon_verifier_blocks: int
    hypothesis_verifier_blocks: int

    # ── Phase 2 targeted recon control ──────────────────────────────────────
    phase2_followup_count: int
    phase2_followup_max: int
    phase2_target_service_key: str
    phase2_target_port: int
    phase2_target_product: str
    retrieval_status: str          # "ok" | "no_match" | "query_invalid" | "backend_failed" | "dataset_missing" | "empty"
    retrieval_errors: list
    service_exhausted: bool        # True when current service has exhausted follow-ups
    phase2_exhausted_service_keys: list  # list[str] — services whose Phase 2 follow-up budget is spent
    phase2_loop_count: int         # increments each time hypothesis_phase → recon → hypothesis_phase cycles
    phase2_loop_max: int           # hard cap on the above (default 6); forces planning when exceeded

    # ── Dataset mode ───────────────────────────────────────────────────────
    dataset_mode: str           # "" | "curated" | "hybrid" | "live"
    dataset_case_id: str        # dataset case identifier
    benchmark_cve_cache_path: str  # path to curated CVE cache (runtime override)

    # ── Active M0-M5 pipeline state ─────────────────────────────────────────
    pipeline_manifest: dict
    pipeline_ledger_path: str
    pipeline_result: dict
    #: Agent-visible runtime contract ``C_A`` (paper §3.1). An object in-process,
    #: projected onto a dict only at the persistence boundary.
    contract: Any
    #: Phase budgets K1/K2/K3/B_max carried by the run (paper §3.3).
    phase_budgets: Any
    #: Terminal outcome of the v7 episode: verified_success | budgeted_failure.
    terminal_outcome: str
    #: Decision units (``B_dec``) consumed by the v7 episode.
    decision_units: int
    #: Per-iteration trace of the v7 episode (Algorithm 2).
    episode_iterations: list
    #: Number of admissible hypotheses built by ``pipeline_hypotheses``.
    hypothesis_count: int
    retrieval_mode: str          # "live" | "snapshot" | "replay"
    source_snapshot_dir: str
    source_snapshot_hash: str
    #: Optional mounted vendor-advisory set (an additional ``K`` source).
    vendor_advisories_path: str
    exploit_candidates: list     # list[dict] — serialized ExploitCandidate
    oracle_truth: dict           # evaluator input; cleared at pipeline entry
    evaluator_truth_path: str    # private run artifact; agent roles never read it
    public_task: dict
    model_profile: str
    automatic_exploit_compilation: bool
    automatic_metasploit_discovery: bool
    allow_llm_fallback: bool
    lifecycle_progress: dict
    session_artifacts: list

    # ── Budget tier and singleton budget state (B1/B8) ───────────────────────
    # BudgetTier string: "low" | "medium" | "high"
    budget_tier: str
    # Serialized BudgetState dict — singleton persisted across all graph nodes
    budget_state: dict
    policy_state: dict          # service budgets, failure streaks and rotation state

    # ── Active fingerprint key (B2 multi-service targeting) ──────────────────
    # Identifies which host:port:protocol:service fingerprint is planned.
    active_fp_key: str

    # ── LLM Agent Pipeline State (v6 evidence-gated) ─────────────────────────
    planner_proposals: list      # list[dict] — serialized planner proposals
    critic_verdicts: list        # list[dict] — serialized critic verdicts
    gate_verdicts: list          # list[dict] — Figure 1 block F: approve | recover | reject
    subgoals: list               # list[dict] — Figure 1 block G: pending evidence-acquisition subgoals
    verifier_decisions: list     # list[dict] — serialized VerifierDecision
    guided_procedures: list      # list[dict] — serialized guided_procedure ExploitCandidates
    catalog_exhausted: bool      # True when all catalog kinds exhausted without task_proof
    planner_loop_count: int      # increments each planner→critic→verifier→execute cycle
    planner_loop_max: int        # hard cap (default 5)
    current_verifier_action: str # "collect_evidence" | "replan" | "execute" | "stop"
    last_executed_candidate_id: str
    last_execution_proof: dict   # ProofArtifact dict
    proof_artifacts: list        # append-only ProofArtifact dicts
    last_execution_result: dict
    execution_intent: dict
    active_plan: dict
    role_usage: list
    pending_evidence_request: dict

    # Optional pipeline overrides populated by benchmark/runtime callers.
    # Keeping these keys explicit preserves type checking without changing the
    # state contract or runtime defaults.
    oracle_spec: dict[str, Any]
    approved_lab_cves: list[str]
    variant: str
    condition: str
    pipeline_recon_observations: list[Any]
    recon_observations: list[Any]
    exploit_candidate_specs: list[Any]
    candidate_specs: list[Any]

    # ── Recon budget controls ────────────────────────────────────────────────
    recon_followup_step_budget: int
    recon_command_dedupe_window: int
    live_retrieval_retry_max: int

    # ── Metrics accumulation ──────────────────────────────────────────────────
    # Detailed token breakdown (mirrors BudgetState fields)
    total_input_tokens: int        # LLM prompt/input tokens
    total_cached_input_tokens: int # Cached/prefix input tokens (billed at reduced rate)
    total_output_tokens: int       # LLM completion/output tokens
    total_thinking_tokens: int     # Thinking/reasoning tokens
    total_tokens: int              # Sum of all four above (1:1 budget cap)
    total_llm_requests: int        # Alias for llm_calls in BudgetState
    total_usd: float               # Accumulated USD cost from billing
    total_invalid_commands: int
    total_repeated_actions: int
    retry_spent: int

    # ── Timing (for M11 Time-to-Access) ──────────────────────────────────────
    run_start_time: float          # time.time() at pipeline entry
    run_end_time: float            # time.time() at pipeline exit
    # {"recon_start": t, "recon_end": t, "execution_success_time": t, ...}
    phase_timestamps: dict
    max_runtime_seconds: int
    timeout_exceeded: bool

    # ── Session state (Phase 5: Maintaining Access) ───────────────────────────
    session_verified: bool         # True if session confirmed alive post-exploit
    session_privilege_level: Optional[str]  # "user" | "root" | "www-data" | None
    session_alive: bool            # False if session dropped during Phase 5
    session_artifact: Optional[dict]  # verification command + proof for Phase 5
    lport: Optional[str]

    # ── Conversation (append-only) ────────────────────────────────────────────
    # Uses LangGraph's add_messages reducer so updates are always appended
    messages: Annotated[list, add_messages]


def state_int(state: Mapping[str, Any], key: str, default: int = 0) -> int:
    """Read an integer state value with explicit runtime narrowing.

    TypedDict ``get`` returns ``object`` for optional/dynamic keys under the
    strict mypy configuration used by the framework.  State is JSON-shaped at
    runtime, so accepting numeric strings keeps the existing behavior while
    making arithmetic sites type-safe.
    """
    value = state.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def initial_state(
    target_ip: str,
    target_port: str = "",
    attacker_ip: str = "",
    keyword: str = "",
    app_name: str = "",
    app_version: str = "",
    recon_max_steps: int = 12,
    execution_max_steps: int = 30,
    replan_max: int = 3,
    max_runtime_seconds: int = 3600,
) -> PentestState:
    """Build a fresh PentestState with sensible defaults."""
    return PentestState(
        target_ip=target_ip,
        target_port=target_port or None,
        attacker_ip=attacker_ip or None,
        current_phase="recon",
        phase2_route="",
        error_count=0,
        last_error=None,
        recon_complete=False,
        port_services={},
        os_info=None,
        recon_step_count=0,
        recon_max_steps=recon_max_steps,
        target_services=[],
        current_service_index=0,
        keyword=keyword,
        app_name=app_name,
        app_version=app_version,
        vuln_type="",
        cve_list=[],
        exploit_plan=[],
        selected_exploit=None,
        planning_output_dir=None,
        planning_complete=False,
        # Debate
        current_proposal=None,
        debate_history=[],
        debate_round=0,
        doc_dir=None,
        execution_step_count=0,
        execution_max_steps=execution_max_steps,
        execution_success=False,
        execution_summary=None,
        execution_tracker={},
        # Layer 1: Structured memory
        world_state={},
        episodic_memory=[],
        decision_memory=[],
        # Layer 3: Hypothesis
        retrieval_bundle={},
        vuln_hypotheses=[],
        hypothesis_complete=False,
        hypothesis_rework_count=0,
        replan_count=0,
        replan_max=replan_max,
        attempted_cves=[],
        attempted_services=[],
        # Layer 2: Verification
        verification_log=[],
        verifier_blocks=0,
        recon_verifier_blocks=0,
        hypothesis_verifier_blocks=0,
        # Phase 2 targeted recon control
        phase2_followup_count=0,
        phase2_followup_max=2,
        phase2_target_service_key="",
        phase2_target_port=0,
        phase2_target_product="",
        retrieval_status="",
        retrieval_errors=[],
        service_exhausted=False,
        phase2_exhausted_service_keys=[],
        phase2_loop_count=0,
        phase2_loop_max=6,
        # Dataset mode
        dataset_mode="",
        dataset_case_id="",
        benchmark_cve_cache_path="",
        # Active M0-M5 pipeline
        pipeline_manifest={},
        pipeline_ledger_path="",
        pipeline_result={},
        retrieval_mode="snapshot",
        source_snapshot_dir="",
        source_snapshot_hash="",
        exploit_candidates=[],
        oracle_truth={},
        evaluator_truth_path="",
        public_task={},
        model_profile="",
        automatic_exploit_compilation=False,
        automatic_metasploit_discovery=False,
        allow_llm_fallback=True,
        lifecycle_progress={},
        session_artifacts=[],
        # Budget tier and singleton state (B1/B8)
        budget_tier="medium",
        budget_state={},
        policy_state={},
        # Active fingerprint key (B2)
        active_fp_key="",
        # LLM Agent Pipeline (v6 evidence-gated)
        planner_proposals=[],
        critic_verdicts=[],
        gate_verdicts=[],
        subgoals=[],
        verifier_decisions=[],
        guided_procedures=[],
        catalog_exhausted=False,
        planner_loop_count=0,
        planner_loop_max=5,
        current_verifier_action="",
        last_executed_candidate_id="",
        last_execution_proof={},
        proof_artifacts=[],
        last_execution_result={},
        execution_intent={},
        active_plan={},
        role_usage=[],
        pending_evidence_request={},
        # Recon budget controls
        recon_followup_step_budget=3,
        recon_command_dedupe_window=10,
        live_retrieval_retry_max=1,
        # Metrics — detailed token breakdown
        total_input_tokens=0,
        total_cached_input_tokens=0,
        total_output_tokens=0,
        total_thinking_tokens=0,
        total_tokens=0,
        total_llm_requests=0,
        total_usd=0.0,
        total_invalid_commands=0,
        total_repeated_actions=0,
        retry_spent=0,
        # Timing
        run_start_time=0.0,
        run_end_time=0.0,
        phase_timestamps={},
        max_runtime_seconds=max_runtime_seconds,
        timeout_exceeded=False,
        # Session
        session_verified=False,
        session_privilege_level=None,
        session_alive=False,
        session_artifact=None,
        lport=None,
        messages=[],
    )


def service_target_key(target_ip: str, port: int | str, service_name: str = "") -> str:
    name = str(service_name or "").strip().lower()
    return f"{str(target_ip or '').strip()}:{int(port or 0)}:{name}"


def runtime_exceeded(state: Mapping[str, Any], now: float | None = None) -> tuple[bool, str]:
    max_runtime = int(state.get("max_runtime_seconds", 0) or 0)
    start = float(state.get("run_start_time", 0.0) or 0.0)
    if max_runtime <= 0 or start <= 0:
        return False, ""
    current = now if now is not None else time.time()
    elapsed = current - start
    if elapsed < max_runtime:
        return False, ""
    minutes = max_runtime // 60 if max_runtime % 60 == 0 else round(max_runtime / 60.0, 1)
    return True, f"Run exceeded max runtime ({minutes} minute limit)."
