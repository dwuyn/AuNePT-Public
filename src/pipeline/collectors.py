"""
src/pipeline/collectors.py
──────────────────────────
Method-specific candidate collectors.

Each collector returns zero or more :class:`ExploitCandidate` records for one
CVE. Trust is set conservatively; only ``trusted`` and ``lab_approved``
candidates may execute. Unknown / unofficial GitHub results remain
``discovery_only``.

Pinning rules (enforced in v1):

  * Nuclei templates are pinned to a repository commit; template ID and
    classification are preserved. Nuclei updates are disabled by callers.
    Code / headless / self-contained / AI-generated / unsigned templates remain
    disabled.
  * Metasploit modules pin module name and revision; the simplest reproducible
    integration is a generated resource script driven by ``MSF_CFGROOT_CONFIG``.
  * ExploitDB and NSE scripts are read from locally indexed sources; their
    binary hashes and versions are captured.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.pipeline.candidates import (
    ExploitCandidate,
    ProcedureStep,
    Provenance,
    derive_candidate_id,
)
from src.pipeline.evidence import normalize_advisory_range

# ── Helpers ──────────────────────────────────────────────────────────────────


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _binary_path(name: str) -> str | None:
    return shutil.which(name)


# ── PoC collector (trusted public) ───────────────────────────────────────────


@dataclass
class PublicPocSpec:
    """A trusted public PoC reference that ships inside the benchmark manifest."""

    cve_id: str
    repo: str                          # "owner/name" (no scheme)
    commit: str                        # pinned SHA
    entry_point: str                   # path inside the repo
    capability: str = "code_execution"
    license: str = "unknown"


def collect_public_poc(spec: PublicPocSpec, *, working_dir: str = "") -> ExploitCandidate:
    """Pin a public PoC to a specific commit and capture its hash.

    The local artifact is fetched only if missing; otherwise we re-hash from
    the existing checkout.
    """
    repo_path = os.path.join(working_dir, "_poc_cache", spec.repo.replace("/", "__"))
    if not os.path.exists(repo_path) or not _pinned_at_commit(repo_path, spec.commit):
        if working_dir:
            _fetch_pinned(repo_path, spec.repo, spec.commit, working_dir)
    entry_path = os.path.join(repo_path, spec.entry_point) if working_dir else ""
    sha = _file_sha256(entry_path) if entry_path and os.path.exists(entry_path) else ""
    prov = Provenance(
        revision=spec.commit, sha256=sha,
        references=[f"https://github.com/{spec.repo}"],
        license=spec.license,
        trust="trusted" if spec.license.lower() not in {"unknown", "private", ""} else "discovery_only",
        source_kind="github", advisory_ref=f"https://github.com/{spec.repo}@{spec.commit}",
    )
    procedure = [ProcedureStep(stage="execute",
                                 argv=["python3", spec.entry_point],
                                 timeout_seconds=60)]
    cand = ExploitCandidate(
        candidate_id=derive_candidate_id(kind="poc", cve_id=spec.cve_id,
                                            locator=spec.entry_point, provenance=prov),
        cve_id=spec.cve_id, kind="poc", source=spec.repo,
        locator=spec.entry_point, provenance=prov, procedure=procedure,
        capability=spec.capability, side_effect_class="remote_exploit",
        working_dir=repo_path, artifact_hash=sha,
    )
    return cand


def _pinned_at_commit(repo_path: str, commit: str) -> bool:
    head = os.path.join(repo_path, ".git", "HEAD")
    if not os.path.exists(head):
        return False
    out = subprocess.run(["git", "-C", repo_path, "rev-parse", "HEAD"],
                            check=False, capture_output=True, text=True, timeout=5).stdout.strip()
    return out.lower() == commit.lower()


def _fetch_pinned(repo_path: str, repo: str, commit: str, working_dir: str) -> None:
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    subprocess.run(["git", "init", "-q", repo_path], check=False)
    subprocess.run(["git", "-C", repo_path, "remote", "add", "origin",
                    f"https://github.com/{repo}.git"], check=False)
    subprocess.run(["git", "-C", repo_path, "fetch", "-q", "--depth", "1", "origin", commit],
                    check=False, timeout=60)


# ── ExploitDB collector ─────────────────────────────────────────────────────


@dataclass
class ExploitDbSpec:
    cve_id: str
    edb_id: str
    local_path: str                   # path inside exploitdb/exploits
    capability: str = "code_execution"
    language: str = "unknown"

    def to_evidence(self) -> str:
        return f"exploitdb:{self.edb_id}"


def collect_exploitdb(spec: ExploitDbSpec) -> ExploitCandidate:
    """Index a local ExploitDB entry; trust is *lab_approved* (research use)."""
    sha = _file_sha256(spec.local_path) if os.path.exists(spec.local_path) else ""
    references = [f"https://www.exploit-db.com/exploits/{spec.edb_id}"]
    prov = Provenance(
        revision=f"edb-{spec.edb_id}", sha256=sha,
        references=references, license="unknown",
        trust="lab_approved",          # research-lab use only; not redistributed
        source_kind="exploitdb", advisory_ref=references[0],
    )
    if spec.language == "python":
        argv = ["python3", spec.local_path]
    elif spec.language in {"c", "cpp"}:
        argv = ["sh", "-c", f"cd {os.path.dirname(spec.local_path)} && make"]
    else:
        argv = ["sh", spec.local_path]
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="exploitdb", cve_id=spec.cve_id,
                                            locator=f"edb:{spec.edb_id}", provenance=prov),
        cve_id=spec.cve_id, kind="exploitdb", source="exploitdb",
        # The EDB id, not an installation-specific path, is the stable locator.
        locator=f"edb:{spec.edb_id}", provenance=prov,
        procedure=[ProcedureStep(stage="execute", argv=argv, timeout_seconds=60)],
        capability=spec.capability, side_effect_class="remote_exploit",
        artifact_hash=sha, working_dir=os.path.dirname(spec.local_path),
        extra={"artifact_path": spec.local_path, "language": spec.language},
    )


# ── Metasploit collector ─────────────────────────────────────────────────────


@dataclass
class MetasploitSpec:
    cve_id: str
    module_name: str                  # e.g. exploit/multi/http/path_traversal
    options: dict[str, str] = field(default_factory=dict)
    rank: str = "manual"
    check_supported: bool = True
    capability: str = "code_execution"


def render_metasploit_resource_script(module: str, options: Mapping[str, str]) -> str:
    """Produce a run-local Metasploit resource script that drives the module.

    Per the handoff, the simplest reproducible integration is a resource
    script invoked via ``MSF_CFGROOT_CONFIG`` rather than a new RPC service.
    """
    lines = [f"use {module}", "set verbose true"]
    for k, v in options.items():
        lines.append(f"set {k} {v}")
    if options.get("PAYLOAD"):
        lines.append(f"set PAYLOAD {options['PAYLOAD']}")
    lines.append("check")
    lines.append("exploit -j")
    lines.append("sleep 5")
    lines.append("sessions -l")
    lines.append("exit")
    return "\n".join(lines) + "\n"


def collect_metasploit(spec: MetasploitSpec) -> ExploitCandidate:
    """Capture a Metasploit module reference; the renderer writes a run-local
    resource script and configuration root at execution time."""
    references = [
        f"https://www.rapid7.com/db/modules/{spec.module_name}",
        "https://docs.rapid7.com/metasploit/modules/",
    ]
    prov = Provenance(
        revision=spec.module_name, sha256="",
        references=references, license="BSD-3-Clause",
        trust="trusted", source_kind="metasploit",
        advisory_ref=references[0],
    )
    script = render_metasploit_resource_script(spec.module_name, spec.options)
    procedure = [
        ProcedureStep(stage="setup", argv=["msfconsole", "-q", "-r", "<MSF_RC>"],
                       timeout_seconds=30),
        ProcedureStep(stage="execute", argv=["msfconsole", "-q", "-r", "<MSF_RC>"],
                       timeout_seconds=120),
        ProcedureStep(stage="verify", argv=["msfconsole", "-q", "-r", "<MSF_RC_VERIFY>"],
                       timeout_seconds=30),
    ]
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="metasploit", cve_id=spec.cve_id,
                                            locator=spec.module_name, provenance=prov),
        cve_id=spec.cve_id, kind="metasploit", source="metasploit",
        locator=spec.module_name, provenance=prov, procedure=procedure,
        capability=spec.capability, side_effect_class="remote_exploit",
        placeholders=["RHOSTS", "RHOST", "LHOST", "LPORT", "PAYLOAD"],
        extra={"resource_script": script, "options": dict(spec.options),
               "rank": spec.rank, "check_supported": spec.check_supported},
    )


# ── Nuclei collector ─────────────────────────────────────────────────────────


@dataclass
class NucleiSpec:
    cve_id: str
    template_id: str                  # e.g. CVE-2021-41773
    template_path: str                # local pinned template path
    classification: str               # e.g. cve, rce, sqli
    pinned_commit: str                # nuclei-templates repo commit
    capability: str = "detection"

    UNSAFE_CLASSIFICATIONS = {"code", "headless", "self-contained", "ai"}


def collect_nuclei(spec: NucleiSpec) -> ExploitCandidate:
    """Index a Nuclei template.

    Unsafe classifications (code, headless, self-contained, AI) are
    automatically rejected regardless of caller configuration. Frozen runs
    must pass ``update=false`` and ``disable-code/headless`` to ``nuclei`` at
    execution time.
    """
    if spec.classification.lower() in NucleiSpec.UNSAFE_CLASSIFICATIONS:
        raise ValueError(
            f"Unsafe Nuclei classification '{spec.classification}' is disabled by default."
        )
    sha = _file_sha256(spec.template_path) if os.path.exists(spec.template_path) else ""
    references = [
        "https://docs.projectdiscovery.io/templates/structure",
        "https://docs.projectdiscovery.io/opensource/nuclei/running",
    ]
    prov = Provenance(
        revision=spec.pinned_commit, sha256=sha,
        references=references + [f"https://github.com/projectdiscovery/nuclei-templates@{spec.pinned_commit}"],
        license="MIT",
        trust="trusted",
        source_kind="nuclei",
        advisory_ref=f"nuclei:{spec.template_id}",
    )
    argv = [
        "nuclei", "-update=false", "-duc", "-nc", "-t", spec.template_path,
        "-json-export", "<NUCLEI_OUTPUT>", "-disable-update-check",
    ]
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="nuclei", cve_id=spec.cve_id,
                                            locator=spec.template_id, provenance=prov),
        cve_id=spec.cve_id, kind="nuclei", source="nuclei",
        locator=spec.template_id, provenance=prov,
        procedure=[ProcedureStep(stage="execute", argv=argv, timeout_seconds=60)],
        capability=spec.capability, side_effect_class="read_only",
        placeholders=["<NUCLEI_OUTPUT>"],
        artifact_hash=sha,
    )


# ── Nmap NSE collector ──────────────────────────────────────────────────────


@dataclass
class NmapNseSpec:
    cve_id: str
    script_name: str                  # e.g. http-vuln-cve2021-41773
    script_path: str                  # local NSE path
    script_args: list[str] = field(default_factory=list)
    capability: str = "detection"


def collect_nmap_nse(spec: NmapNseSpec) -> ExploitCandidate:
    if not spec.script_path:
        raise ValueError("NmapNseSpec.script_path is required")
    if not (os.path.exists(spec.script_path) or _binary_path("nmap")):
        raise ValueError(f"NSE script path missing: {spec.script_path}")
    sha = _file_sha256(spec.script_path) if os.path.exists(spec.script_path) else ""
    prov = Provenance(
        revision=os.path.basename(spec.script_path), sha256=sha,
        references=["https://nmap.org/nsedoc/"],
        license="GPL-2.0", trust="trusted",
        source_kind="nmap", advisory_ref=f"nmap:{spec.script_name}",
    )
    argv = ["nmap", "--script", spec.script_name, *spec.script_args, "<RHOST>"]
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="nmap_nse", cve_id=spec.cve_id,
                                            locator=spec.script_name, provenance=prov),
        cve_id=spec.cve_id, kind="nmap_nse", source="nmap", locator=spec.script_name,
        provenance=prov,
        procedure=[ProcedureStep(stage="execute", argv=argv, timeout_seconds=120)],
        capability=spec.capability, side_effect_class="read_only",
        artifact_hash=sha,
    )


# ── Vendor / native collector ────────────────────────────────────────────────


@dataclass
class VendorRecipeSpec:
    cve_id: str
    vendor: str
    product: str
    steps: list[ProcedureStep]
    references: list[str] = field(default_factory=list)
    license: str = "vendor-documented"
    capability: str = "code_execution"
    #: ``rho(c)``'s procedure template + exploit-specific operands (paper §3.3).
    #: A candidate that names a registered template compiles through it; the
    #: template then renders the target bindings from ``C_A``/``M_W``.
    template_id: str = ""
    bindings: dict[str, Any] = field(default_factory=dict)
    #: ``rho(c)``'s preconditions and serialization switches (paper §3.3); see
    #: :class:`NativeToolSpec`.
    prerequisites: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def collect_vendor_recipe(spec: VendorRecipeSpec) -> ExploitCandidate:
    prov = Provenance(
        revision=f"{spec.vendor}-{spec.product}", sha256="",
        references=list(spec.references), license=spec.license,
        trust="trusted", source_kind="vendor",
        advisory_ref=spec.references[0] if spec.references else "",
    )
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="vendor_recipe", cve_id=spec.cve_id,
                                            locator=f"{spec.vendor}/{spec.product}", provenance=prov),
        cve_id=spec.cve_id, kind="vendor_recipe", source=spec.vendor,
        locator=f"{spec.vendor}/{spec.product}",
        provenance=prov, procedure=list(spec.steps),
        capability=spec.capability, side_effect_class="remote_exploit",
    )


@dataclass
class NativeToolSpec:
    cve_id: str
    tool_name: str
    argv: list[str]
    capability: str = "detection"
    references: list[str] = field(default_factory=list)
    #: ``rho(c)``'s procedure template + exploit-specific operands (paper §3.3);
    #: see :class:`VendorRecipeSpec`.
    template_id: str = ""
    bindings: dict[str, Any] = field(default_factory=dict)
    #: ``rho(c)``'s preconditions (paper §3.3), keyed like ``ExploitCandidate``:
    #: ``endpoint_prereqs``, ``network_prereqs``, ``auth_required``, ``platform``.
    prerequisites: dict[str, Any] = field(default_factory=dict)
    #: ``rho(c)``'s serialization switches (paper §3.3), e.g. ``raw_path`` for a
    #: procedure whose operand path must reach the target unnormalised.
    extra: dict[str, Any] = field(default_factory=dict)


def collect_native_tool(spec: NativeToolSpec) -> ExploitCandidate:
    if not _binary_path(spec.tool_name):
        # Still record the candidate but mark trust as discovery_only; the
        # runner will block execution.
        trust = "discovery_only"
    else:
        trust = "trusted"
    prov = Provenance(
        revision=_binary_path(spec.tool_name) or spec.tool_name, sha256="",
        references=list(spec.references), license="unknown",
        trust=trust, source_kind="native",
        advisory_ref=spec.references[0] if spec.references else "",
    )
    return ExploitCandidate(
        candidate_id=derive_candidate_id(kind="native_tool", cve_id=spec.cve_id,
                                            locator=spec.tool_name, provenance=prov),
        cve_id=spec.cve_id, kind="native_tool", source=spec.tool_name,
        locator=spec.tool_name, provenance=prov,
        procedure=[ProcedureStep(stage="execute", argv=spec.argv, timeout_seconds=60)],
        capability=spec.capability, side_effect_class="read_only",
    )


# ── Index helpers (locally indexed method sources) ──────────────────────────


def index_exploitdb(root: str) -> list[ExploitDbSpec]:
    """Walk a local exploitdb checkout and produce per-CVE specs."""
    specs: list[ExploitDbSpec] = []
    if not os.path.isdir(root):
        return specs
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        text = _file_text(path)
        m = re.search(r"CVE-(\d{4})-(\d{4,7})", text, re.IGNORECASE)
        if not m:
            continue
        edb_id_match = re.search(r"EDB-ID[:=\s]*(\d+)", text)
        specs.append(ExploitDbSpec(
            cve_id=f"CVE-{m.group(1)}-{m.group(2)}",
            edb_id=edb_id_match.group(1) if edb_id_match else os.path.basename(path),
            local_path=path,
            language="python",
        ))
    return specs


def index_nse_scripts(root: str) -> list[NmapNseSpec]:
    """Walk a local NSE script directory."""
    specs: list[NmapNseSpec] = []
    if not os.path.isdir(root):
        return specs
    for path in glob.glob(os.path.join(root, "**", "*.nse"), recursive=True):
        text = _file_text(path)
        m = re.search(r"CVE-(\d{4})-(\d{4,7})", text, re.IGNORECASE)
        if not m:
            continue
        specs.append(NmapNseSpec(
            cve_id=f"CVE-{m.group(1)}-{m.group(2)}",
            script_name=os.path.basename(path).replace(".nse", ""),
            script_path=path,
        ))
    return specs


def collect_for_cve(cve_id: str, *, specs: Iterable[Any]) -> list[ExploitCandidate]:
    """Filter candidate specs for one CVE and convert to ExploitCandidates."""
    out: list[ExploitCandidate] = []
    for spec in specs:
        if getattr(spec, "cve_id", "") != cve_id:
            continue
        kind = getattr(spec, "kind_marker", None) or _kind_of(spec)
        if kind == "poc":
            candidate = collect_public_poc(spec)
        elif kind == "exploitdb":
            candidate = collect_exploitdb(spec)
        elif kind == "metasploit":
            candidate = collect_metasploit(spec)
        elif kind == "nuclei":
            candidate = collect_nuclei(spec)
        elif kind == "nmap_nse":
            candidate = collect_nmap_nse(spec)
        elif kind == "vendor_recipe":
            candidate = collect_vendor_recipe(spec)
        elif kind == "native_tool":
            candidate = collect_native_tool(spec)
        else:
            # guided_procedure candidates are created by the PlannerAgent,
            # not collected from specs.  Skip here.
            continue
        _apply_procedure_spec(candidate, spec)
        out.append(candidate)
    return out


def _apply_procedure_spec(candidate: ExploitCandidate, spec: Any) -> None:
    """Bind ``rho(c)``'s template, operands and preconditions from its spec.

    A spec that names a registered template makes the candidate compilable: the
    template supplies the argv shape and the target bindings, while the spec's
    ``bindings`` supply the exploit-specific operands and its ``prerequisites``
    the preconditions ``App(c, M_W)`` evaluates (paper §3.3).
    """
    template_id = str(getattr(spec, "template_id", "") or "")
    if template_id:
        candidate.template_id = template_id
    declared = getattr(spec, "bindings", None)
    if isinstance(declared, Mapping) and declared:
        candidate.bindings = {**dict(candidate.bindings or {}), **dict(declared)}
    declared_extra = getattr(spec, "extra", None)
    if isinstance(declared_extra, Mapping) and declared_extra:
        candidate.extra = {**dict(candidate.extra or {}), **dict(declared_extra)}
    prerequisites = getattr(spec, "prerequisites", None)
    if isinstance(prerequisites, Mapping):
        for key in ("endpoint_prereqs", "network_prereqs"):
            values = prerequisites.get(key)
            if isinstance(values, (list, tuple)) and values:
                setattr(candidate, key, [str(item) for item in values])
        for key in ("auth_required", "platform"):
            value = str(prerequisites.get(key, "") or "")
            if value:
                setattr(candidate, key, value)


def collect_from_records(
    records: Iterable[Any],
    *,
    specs: Iterable[Any] = (),
    candidates: Iterable[ExploitCandidate] = (),
    max_cves: int = 5,
) -> list[ExploitCandidate]:
    """Collect the configured candidates whose CVE the retrieval returned.

    The candidate universe is the spec set (plus already-seeded candidates);
    ``K(p)`` only decides which of them the retrieved advisory admits, so the
    work is proportional to the candidate universe rather than to the size of a
    product's advisory list. ``max_cves`` caps how many of those CVEs may supply
    candidates for one service, newest-returned first.
    """
    advisory: dict[str, Any] = {}
    position: dict[str, int] = {}
    for index, rec in enumerate(records):
        cve_id = str(getattr(rec, "cve_id", "") or "").upper()
        if not cve_id:
            continue
        if cve_id not in advisory:
            advisory[cve_id] = rec
            position[cve_id] = index

    existing = list(candidates or ())
    wanted: list[str] = []
    for spec in specs:
        cve_id = str(getattr(spec, "cve_id", "") or "").upper()
        if cve_id and cve_id not in wanted:
            wanted.append(cve_id)
    for candidate in existing:
        cve_id = str(getattr(candidate, "cve_id", "") or "").upper()
        if cve_id and cve_id not in wanted:
            wanted.append(cve_id)

    admitted = sorted((cve_id for cve_id in wanted if cve_id in advisory),
                      key=lambda cve_id: (position[cve_id], cve_id))[:max_cves]
    out: list[ExploitCandidate] = []
    seen: set[str] = set()
    for cve_id in admitted:
        for cand in [c for c in existing if c.cve_id.upper() == cve_id]:
            if cand.candidate_id not in seen:
                out.append(cand)
                seen.add(cand.candidate_id)
        for cand in collect_for_cve(cve_id, specs=specs):
            if cand.candidate_id not in seen:
                out.append(cand)
                seen.add(cand.candidate_id)
    for cand in out:
        _bind_constraint_from_advisory(cand, advisory.get(str(cand.cve_id).upper()))
    return out


def _bind_constraint_from_advisory(candidate: ExploitCandidate, record: Any) -> None:
    """Bind ``rho(c)``'s product/version predicate to the advisory it came from.

    A candidate collected for a retrieved record inherits that record's
    vendor/product and affected-version bounds unless it declares its own, so
    ``App(c, M_W)`` evaluates the predicates the advisory states (paper §3.2)
    instead of being vacuous. A record that names no range leaves the candidate
    version-independent rather than inventing bounds.
    """
    if record is None:
        return
    constraint = getattr(candidate, "constraint", None)
    if constraint is None:
        return
    if not str(getattr(constraint, "product", "") or ""):
        constraint.product = str(getattr(record, "product", "") or "")
    if not str(getattr(constraint, "vendor", "") or ""):
        constraint.vendor = str(getattr(record, "vendor", "") or "")
    start = str(getattr(constraint, "version_start", "") or "")
    end = str(getattr(constraint, "version_end", "") or "")
    if not start and not end:
        low, high, low_inclusive, high_inclusive = normalize_advisory_range(
            str(getattr(record, "version_start", "") or ""),
            str(getattr(record, "version_end", "") or ""),
            start_inclusive=bool(getattr(record, "version_start_inclusive", True)),
            end_inclusive=bool(getattr(record, "version_end_inclusive", True)),
        )
        constraint.version_start = low
        constraint.version_end = high
        constraint.version_start_inclusive = low_inclusive
        constraint.version_end_inclusive = high_inclusive
        constraint.is_unknown_version = not (low or high)


def _kind_of(spec: Any) -> str:
    name = type(spec).__name__.lower()
    if "publicpoc" in name or "pocspec" in name:
        return "poc"
    if "exploitdbspec" in name:
        return "exploitdb"
    if "metasploitspec" in name:
        return "metasploit"
    if "nucleispec" in name:
        return "nuclei"
    if "nmapnsespec" in name:
        return "nmap_nse"
    if "vendorrecipespec" in name:
        return "vendor_recipe"
    if "nativetoolspec" in name:
        return "native_tool"
    if "guidedprocedurespec" in name:
        return "guided_procedure"
    return "unknown"
