# Pipeline

The active evidence-driven research pipeline. The default LangGraph workflow is
`recon -> pipeline_prepare -> pipeline_retrieve -> pipeline_queue ->
pipeline_execute -> pipeline_oracle`. The legacy PoC-only workflow remains
available only as the frozen variant 1 baseline under the
`baseline-poc-only-v1` git tag.

## Modules

| Module                       | Responsibility                                                       |
| ---------------------------- | -------------------------------------------------------------------- |
| `manifest.py`                | `RunManifest`, atomic `RunContext`, scope/limits, secret redaction   |
| `budget.py`                  | Hard resource gates per `ResourceLimits`                             |
| `ledger.py`                  | Append-only event ledger; allowed outcomes and failure classes       |
| `scope.py`                   | Full endpoint validation (IPv4/IPv6/hostname/URL/redirect/callback)  |
| `oracle.py`                  | `BenchmarkOracle` (independent proof) and `TextualMarkerChecker`      |
| `evaluator.py`               | Drives a variant runner through independent oracle adjudication      |
| `evidence.py`                | Service fingerprinting with observation/inference separation         |
| `sources.py`                 | CVE List V5, NVD, Vulnx, KEV, EPSS adapters; snapshot mode           |
| `candidates.py`              | `ExploitCandidate` interface + legacy `PocCandidate` reader           |
| `collectors.py`              | Method collectors: PoC, ExploitDB, Metasploit, Nuclei, Nmap NSE, etc. |
| `queue.py`                   | Deterministic applicability, ranking, and per-CVE shortlist          |
| `renderers.py`               | Structured argv-array renderers (no free-form shell)                 |
| `runner.py`                  | Top-level pipeline runner                                            |
| `benchmark.py`               | Benchmark manifests, metrics-from-events, reporting                  |
| `vertex_runtime.py`          | Pinned model identity, pricing snapshot, and fakeable Vertex transports |
| `gateway_relay.py`           | Dependency-free HTTP-to-Unix-socket relay; no provider credentials      |
| `runtime_topology.py`        | Label-owned canary/smoke Docker lifecycle and topology evidence          |
| `runtime_runner.py`          | Relay-gated 3-canary then 15-smoke production coordinator               |
| `runtime_ledger.py`          | Thread-safe host-observed request/response and billing ledger           |

Gemini 3.5/3.6 currently resolve through Google's `default` publisher alias.
Profiles may use `resolution_mode=provider_alias` only with a resolution
evidence hash and timestamp; Gemma MaaS remains revision-pinned. Alias mode is
explicitly recorded in every profile/run identity and does not provide the
same reproducibility guarantee as an immutable model revision.

## Run modes

* **live** – real HTTP / filesystem reads; raw responses preserved.
* **snapshot** – only reads from a fixed source/candidate snapshot dir.
* **replay** – no retrieval or execution; reproduces metrics from stored events.

## Test layout

Tests live under `tests/pipeline/` and follow the milestone naming:

* `test_m1_foundation.py` – manifest / ledger / budget / scope / oracle / evaluator
* `test_m2_evidence_sources.py` – evidence normalization + source adapters
* `test_m3_candidates.py` – `ExploitCandidate` interface and collectors
* `test_m4_queue_runner.py` – queue, renderers, runner
* `test_m5_metrics_benchmark.py` – metrics-from-events + benchmark manifests
* `test_m5_acceptance.py` – handoff completion gates

## Hard limits (from the preregistration)

| Resource                  | Limit       |
| ------------------------- | ----------- |
| Runtime per target        | 20 minutes  |
| Tool calls                | 50          |
| Executed commands         | 40          |
| CVEs per service          | 5           |
| Methods per CVE           | 2           |
| Executed candidates       | 3           |
| Attempts per candidate    | 3           |

Provider, model, and temperature are frozen for primary comparisons.

## Variants

1. Frozen legacy PoC-only graph.
2. Active graph plus official fresh CVE sources.
3. Active graph plus heterogeneous methods on a frozen snapshot.
4. Full active graph with live, snapshot, and replay modes.

Literature results are contextual comparison only; the benchmark reports
source freshness latency, validated-vulnerability discovery, method diversity,
repeated-method rate, fallback rescue rate, oracle-confirmed proof, and false
positives on patched controls.

## Outcome taxonomy

Allowed semantic outcomes (defined in `ledger.ALLOWED_OUTCOMES`):

* `vulnerability_confirmed`
* `task_proof_obtained` — only this counts toward the primary metric.
* `execution_failed`
* `not_applicable`
* `not_executable`
* `blocked_by_policy`
