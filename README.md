# AuNePT

**AuNePT: Autonomous Network Penetration Testing under Partial Observability via Evidence-Constrained
Decision Making.**

AuNePT is an evidence-constrained multi-agent framework for autonomous penetration testing of remotely
accessible services. It separates evidence-backed world state, episodic execution evidence and decision
rationales into a world-state property graph, a content-hashed execution ledger and a decision-rationale
tree, so that observations, hypotheses and actions are never intermingled in a growing conversational
context and unsupported assumptions cannot propagate into live testing.

Contract-scoped reconnaissance admits service evidence into persistent state; product-keyed retrieval and
tri-state applicability keep missing prerequisites as evidence-acquisition subgoals instead of assuming
them away. Exploitation candidates are compiled into protocol-specific **registered procedures** whose
parameters must resolve to evidence-backed state or authorized contract fields, and a pre-dispatch policy
gate rejects exact-action repetition, enforces evidence grounding and network scope, routes recoverable
precondition failures and applies hierarchical execution budgets. Successful access is admitted only by an
**evaluator-private, per-case live success oracle** — never by the agent's own claim — and every case runs
against a matched patched control, so apparent progress is separated from objectively supported exploitation.

This repository is the framework. It contains no comparison frameworks, no benchmark data and no run
artifacts; those are maintained separately (see [Dataset](#dataset)).

## Architecture

An AuNePT assessment progresses through four coupled functions:

1. **Contract-scoped reconnaissance.** A bounded, LLM-free discovery step probes the authorized port,
   classifies the protocol family, and admits service evidence — with version evidence classified as
   explicit, inferred or unknown — into the world-state graph $M_W$, under an agent-visible authorized
   runtime contract $C_A$.
2. **Uncertainty-aware vulnerability hypotheses.** Candidates are retrieved from the advisory corpus by
   product identity only and judged tri-state (`confirmed` / `plausible` / `unresolved`); an unresolved
   prerequisite becomes an evidence-acquisition subgoal with a bounded penalty.
3. **Evidence-bound procedure compilation and policy-constrained execution.** Algorithm 1 builds hypotheses
   from retrieved advisories; Algorithm 2 runs the episode loop — `PopBest` → re-assessed applicability and
   prerequisites → `CompileAndGate` (invariants V1–V6, with the model critique folded in) →
   `BudgetedDispatch` → evaluator verdict → classify → route.
4. **Evaluator-verified adjudication.** Only the external differential oracle admits success, from an
   independent artifact (command output, file content, session log) containing an accepted-evidence marker;
   patched controls cannot yield proof from agent claims.

The corresponding source layout under `src/`:

| Module | Responsibility |
|--------|----------------|
| `design/` | World-state graph and its admission rules, the runtime contract $C_A$, gates V1–V6, subgoals, budgets, outcomes |
| `retrieval/` | Product-keyed retrieval, tri-state applicability, and the closed procedure registry (8 template ids) |
| `agents/` | Reconnaissance, the Algorithm-2 episode loop (`aunept_loop.py`), the model critique |
| `pipeline/` | Run manifest, event ledger, oracle, budgets, candidate collection, source snapshots, the production driver |
| `memory/` | World state, episodic memory, decision memory |
| `execution/` | Dispatch of compiled procedures (host and isolated runtime paths) |
| `scoring/`, `planning/` | Metrics collection, difficulty and policy locks |
| `prompts/`, `rag/` | Prompt templates and the retrieval-augmented document helpers |
| `tools/`, `utils/` | Tool-compatibility shims, the `ws-frame` WebSocket transport adapter, provider clients |

```
main.py                 CLI: run | recon | run-dataset
src/                    the framework (table above)
utils/                  vendored helper modules used by retrieval and RAG (searchers, CVE metadata)
configs/                config.yaml.example (the live config.yaml is git-ignored)
tools/ws_frame.py       first-party WebSocket transport, dispatched as `ws-frame`
pyproject.toml          runtime dependencies (poetry.lock pins them)
requirements.txt        auxiliary pin set for the vendored helpers
```

The container image, the deployment tooling and the benchmark harness are not part of this repository: the
framework is driven in-process by the CLI below, or by the production driver when it runs under an
orchestrator.

## Requirements

- Python 3.11 or later (`requires-python = ">=3.11,<3.13"`).
- Docker, to run the target containers (the vulnerable services an assessment attacks).
- A model provider. The manuscript's locked backbones are Gemini/Gemma on Vertex AI; any OpenAI-compatible
  endpoint works through the dependency-free `openai_compat` provider (`AUNEPT_BASE_URL` / `AUNEPT_API_KEY`).

## Installation

```
git clone <this repository>
cd AuNePT-Framework
python3 -m venv .venv && source .venv/bin/activate
poetry install                     # the declared dependency set (pyproject.toml + poetry.lock)
# or, without poetry:
pip install -r requirements.txt    # auxiliary pin set; see the note below
```

`pyproject.toml` (with `poetry.lock`) is the authoritative dependency list for the runtime
(`langgraph`, `pyyaml`, `google-genai`, `httpx`, `packaging`, `cryptography`, `certifi`, `urllib3`,
`pydantic`, `gitpython`, `tqdm`, `msgpack`, …). The older `requirements.txt` pin set does not cover all of
them — installing from it alone leaves `langgraph` and `pyyaml` missing — so use it only to add the
vendored-helper packages (`requests`, `readability-lxml`, `python-dotenv`, `scikit-learn`) that `utils/`
needs.

## Configuration

All settings live in `configs/config.yaml`, which is git-ignored.

1. Copy the template and fill it in:

   ```
   cp configs/config.yaml.example configs/config.yaml
   ```

2. Either point the config at a provider, or export the environment variables the OpenAI-compatible
   profile reads: `AUNEPT_MODEL`, `AUNEPT_BASE_URL`, `AUNEPT_API_KEY`. Vertex runs additionally need
   `GOOGLE_CLOUD_PROJECT` / `GOOGLE_APPLICATION_CREDENTIALS`.

3. The LLM roles (the model critique and the planner/verifier helpers) run only when the run selects a model
   profile: `state["model_profile"]` must name a profile from `configs/config.yaml` — the template ships
   `aunept_proxy` for any OpenAI-compatible endpoint. Without a profile the framework still runs, but only
   the deterministic gates decide, and the ledger records `critic=unavailable`.

4. WebSocket cases dispatch through the first-party transport adapter, so it must be on `PATH`:

   ```
   ln -sf "$PWD/tools/ws_frame.py" "$HOME/.local/bin/ws-frame"
   ```

5. Product-keyed retrieval reads an advisory corpus. Point a run at one with
   `source_snapshot_dir=<path>`, or set `PENTEST_SOURCE_SNAPSHOT` (plus `PENTEST_SOURCE_SNAPSHOT_HASH` to pin
   it); inside the runtime image the corpus is mounted read-only at `/run/veriplanpt/source-snapshot`, which
   the framework picks up automatically. With no corpus the retrieval returns nothing and no candidate can
   be collected.

## Usage

Against one authorized target:

```
python main.py recon --target <ip> --port <port>
python main.py run   --target <ip> --port <port> [--attacker <ip>] [--thread-id <id>]
python main.py run-dataset --tasks <tasks.json> --case-id <case> --benchmark-cve-cache <json>
```

Inside the runtime image (or under the study runner) the entry point is the production driver, which reads a
signed public invocation instead of CLI flags:

```
VERIPLANPT_PUBLIC_INVOCATION_FILE=<invocation.json> python -m src.pipeline.production_driver
```

Programmatic use — the graph takes a run id and, optionally, a hash-verified source snapshot directory:

```python
from src.graph import build_graph
from src.state import initial_state

graph, config = build_graph("my-run", source_snapshot_dir="/path/to/snapshot")
final = graph.invoke(initial_state("127.0.0.1", target_port="8080"), config=config)
```

Lint and type checks run over the source tree:

```
python3 -m ruff check src
```

## Evaluation protocol

The manuscript evaluates AuNePT against four representative frameworks (PentestAgent, PentestGPT, VulnBot,
HackSynth) on **27 upstream CVE targets** under **three LLM backbones**
(`gemini-3.5-flash`, `gemini-3.6-flash`, `gemma-4-26b-a4b-it`) — 405 episodes — with the same harness,
runtime contract and evaluator for every framework. Success is the evaluator-verified live outcome success
rate (`OSR_live`), reported together with its matched-framework difference and the conditional
`OSR_live|exploit`, and each target has a matched patched control. Diagnostic scores (service and version
identification, exploit applicability, invalid-command and repeated-action rates, error recovery) separate
workflow quality from outcome.

## Dataset

The benchmark data is not distributed in this repository. Download it here:

**https://drive.google.com/drive/folders/1_9K8hwZkh4puRLKROjoN685cJnAFUSZc?usp=sharing**

The Drive folder is laid out like the dataset the experiments expect — one directory per CVE, plus a launch
manifest and the advisory corpus:

```
manifest.yaml            one entry per CVE: case id, split, product, versions, lab, verification
CVE-YYYY-NNNNN/          one directory per case (27 held-out targets), each holding
  target/                  the case's container environment (Dockerfile, app, patches)
  hidden/                  host-side truth and the case oracle (never mounted into a target container)
  public_task.yml          the case's public surface
  lab/                     the real-CVE lab, where one exists: vulnerable + patched Dockerfiles and the PoC
splits/validation/, splits/train/   the remaining frozen cases, same per-case layout
rag_corpus/              where the advisory corpus lives (the retrieval knowledge base)
dataset-metadata/        frozen-dataset metadata: lock, snapshot manifest, robustness variants
```

The held-out verification expects, from that folder:

- the **CVE corpus** — one directory per CVE holding the real-CVE lab (the upstream product's vulnerable
  build, the patched build as the differential, and the published PoC) and the frozen fixture case tree with
  its public task surface;
- the **advisory corpus** used by product-keyed retrieval, together with its manifest and hash (a mounted
  vendor-advisory set supplies advisories the pinned corpus cannot);
- the **run artifacts** (per-cell records, ledgers, checkpoints) of the recorded runs;

Point a run at the corpus with `source_snapshot_dir=<path>`, or by setting `PENTEST_SOURCE_SNAPSHOT`.

## Responsible use

AuNePT performs active exploitation and is intended for authorized security research only. Use it exclusively
against systems you own or have explicit written permission to test, and prefer the isolated, containerized
targets shipped with the benchmark. Because the agent acts autonomously, keep it on controlled networks and
under human oversight. The authors accept no liability for misuse.

## Citation

This work is under peer review. If you use AuNePT, please cite:

```bibtex
@unpublished{aunept2026,
  title  = {{AuNePT}: Autonomous Network Penetration Testing under Partial Observability
            via Evidence-Constrained Decision Making},
  author = {Le Xuan, Nguyen and Tran Dinh, Nguyen and Nguyen Huu, Quyen and Pham, Van-Hau},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

Licensed under the Apache License, Version 2.0 — see the [LICENSE](LICENSE) file for the full text.
