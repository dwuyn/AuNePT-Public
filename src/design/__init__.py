"""Design layer mirroring the contract-driven workflow (design v75).

The design describes the system in blocks A--G on two planes; this package
carries that vocabulary into the code so each block has a named, testable
counterpart:

    Design block                            module / function
    ----------------------------------------------------------------------
    A. Input: contract C = (port,            RuntimeContract (contract.py);
       launch_mode, env, health, relay),     corpus retrieval in
       corpus K, phase budgets               ``src/pipeline/sources.py``;
       K1/K2/K3 + B_max                      PhaseBudgets (budgets.py)
    B. Contract-driven recon (Alg. 1):       ``src/agents/recon.py`` +
       assert C.port, one handshake,         ``src/pipeline/scope.py``
       protocol ladder, closed discovery
       tuple U_disc = {/, /robots.txt,
       /.well-known/} bounded by K1
    C. Three-tier structured memory:         admit() / update() (state.py);
       Tier 1 M_W property graph (V, phi_E)  M_E = ``src/pipeline/ledger.py``
       Tier 2 M_E append-only SHA-256 ledger (is_repeated_action = H_cmds),
       + H_cmds replay guard                 M_D = ``src/memory/decision.py``
       Tier 3 M_D decision rationale tree
    D. Evidence-tiered retrieval (Alg. 2):   ``src/pipeline/evidence.py``,
       keyed by (vendor, product,            ``src/pipeline/sources.py``
       version_range), K(p) lookup
    E. Applicability tri-state               ``src/retrieval/applicability.py``
       confirmed | plausible | unknown       — verdict vocabulary
       + bounded penalty for missing         confirmed/plausible/unknown
       prerequisites, G(c) -> subgoals       (``rejected`` = discarded)
    F. Protocol-aware compilation            registry membership in
       (registry R_f, no free-form           ``src/pipeline/renderers.py``;
       fallback) and dual-tier oversight:    Critic V1--V6 + calibrated
       Critic V1--V6 + calibrated policy,    policy and approve|recover|reject
       Verifier (K1/K2/K3 + B_max,           in gate.py; Verifier budgets in
       oracle-only success)                  ``src/agents/verifier.py``
    G. Budgeted dispatch and outcome         ``src/graph.py`` execute node;
       handling: failure classes             design_route() in
       auth_missing | version_unknown |      ``src/execution/classify.py``;
       payload_rejected | transport_error    outcomes.py
    Q. candidate queue                       ``src/pipeline/queue.py``
    -  evidence subgoals (G)                 SubgoalQueue (subgoals.py)
    Evaluator-private plane                  external_evaluator_verdict(),
       (C_E, verdict only)                   terminal_outcome() (outcomes.py)

Nothing in this package changes the run-time behaviour of the pipeline by
itself; it names the design, exposes the gate semantics of the workflow, and is
used additively by the graph (see ``gate_verdicts`` / ``subgoals`` /
``phase_budgets`` state keys).
"""
from src.design.budgets import PhaseBudgets
from src.design.contract import RuntimeContract
from src.design.gate import (
    GATE_INVARIANTS,
    INVARIANTS,
    GateDecision,
    GateVerdict,
    compile_and_gate,
    gate_decision_from_critic,
    resolve_env,
    resolve_proc,
)
from src.design.outcomes import (
    EvaluatorVerdict,
    TerminalOutcome,
    external_evaluator_verdict,
    terminal_outcome,
)
from src.design.state import Delta, admit, update
from src.design.subgoals import Subgoal, SubgoalKind, SubgoalQueue, subgoals_from_preconditions

__all__ = [
    # A
    "RuntimeContract",
    "PhaseBudgets",
    # C
    "Delta",
    "admit",
    "update",
    # F
    "GateDecision",
    "GateVerdict",
    "GATE_INVARIANTS",
    "INVARIANTS",
    "compile_and_gate",
    "gate_decision_from_critic",
    "resolve_env",
    "resolve_proc",
    # G (subgoals)
    "Subgoal",
    "SubgoalKind",
    "SubgoalQueue",
    "subgoals_from_preconditions",
    # evaluator plane
    "EvaluatorVerdict",
    "TerminalOutcome",
    "external_evaluator_verdict",
    "terminal_outcome",
]
