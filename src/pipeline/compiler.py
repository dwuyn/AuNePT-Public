"""Deterministic evidence-to-execution candidate compilation.

The local-source ladder (searchsploit / Metasploit) lives here and carries no
agent behaviour.  The closed protocol-family registry and the template compiler
of the execution model (paper §3.3) live in :mod:`src.retrieval.registry` and are
re-exported at the bottom of this module so the pipeline keeps one import path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from src.pipeline.candidates import ExploitCandidate, derive_candidate_id
from src.pipeline.collectors import (
    ExploitDbSpec,
    MetasploitSpec,
    collect_exploitdb,
    collect_metasploit,
)


def _language(path: str) -> str:
    return {
        ".py": "python", ".sh": "shell", ".pl": "perl", ".rb": "ruby",
        ".c": "c", ".cc": "cpp", ".cpp": "cpp", ".java": "java",
    }.get(os.path.splitext(path)[1].lower(), "unknown")


def searchsploit_json(cve_id: str, *, binary: str = "searchsploit") -> dict[str, Any]:
    """Read SearchSploit's JSON output without assuming its database path.

    Kali's installed version exposes JSON as ``-j`` rather than ``--json``.
    """
    numeric = cve_id.upper().replace("CVE-", "")
    proc = subprocess.run([binary, "--cve", numeric, "-j"], check=False,
                          capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "searchsploit failed")
    payload = json.loads(proc.stdout or "{}")
    if not isinstance(payload, dict):
        raise ValueError("searchsploit JSON was not an object")
    return payload


def exploitdb_specs(cve_id: str, payload: dict[str, Any], *, run_dir: str) -> list[ExploitDbSpec]:
    """Copy every matching ExploitDB artifact into this run and describe it."""
    out: list[ExploitDbSpec] = []
    artifacts = os.path.join(run_dir, "artifacts", "exploitdb")
    os.makedirs(artifacts, exist_ok=True)
    requested = cve_id.upper()
    for key, rows in payload.items():
        if not key.startswith("RESULTS_") or not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            codes = {item.strip().upper() for item in str(row.get("Codes", "")).split(";")}
            path, edb_id = str(row.get("Path", "")), str(row.get("EDB-ID", ""))
            if requested not in codes or not path or not edb_id or not os.path.isfile(path):
                continue
            copied = os.path.join(artifacts, f"{edb_id}{os.path.splitext(path)[1].lower()}")
            shutil.copy2(path, copied)
            spec = ExploitDbSpec(cve_id=requested, edb_id=edb_id, local_path=copied,
                                 language=_language(path))
            # Keep SearchSploit fields in a harmless side channel consumed by
            # ``compile_exploitdb`` below.
            setattr(spec, "searchsploit", dict(row))
            out.append(spec)
    return out


class MetasploitDiscovery(Protocol):
    """Small RPC boundary; tests and the runtime can supply the implementation."""

    def search(self, cve_id: str) -> Iterable[dict[str, Any]]: ...
    def info(self, module_name: str) -> dict[str, Any]: ...
    def revision(self, module_name: str) -> tuple[str, str]: ...


@dataclass
class ExploitCompiler:
    """Compile known local sources in stable, source-ladder order."""

    run_dir: str
    msf: MetasploitDiscovery | None = None
    searchsploit_binary: str = "searchsploit"

    def compile_cve(self, cve_id: str) -> list[ExploitCandidate]:
        candidates: list[ExploitCandidate] = []
        if self.msf is not None:
            candidates.extend(self._metasploit(cve_id))
        try:
            payload = searchsploit_json(cve_id, binary=self.searchsploit_binary)
            raw = os.path.join(self.run_dir, "source_raw", f"searchsploit-{cve_id}.json")
            os.makedirs(os.path.dirname(raw), exist_ok=True)
            with open(raw, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
            for spec in exploitdb_specs(cve_id, payload, run_dir=self.run_dir):
                cand = collect_exploitdb(spec)
                row = getattr(spec, "searchsploit", {})
                cand.extra.update({"searchsploit": row, "source_raw": raw})
                cand.runtime_kind = "isolated_container"
                cand.requirements = {"binaries": [self._interpreter(spec.language)]}
                cand.expected_evidence = ["vulnerability_confirmation", "task_proof"]
                candidates.append(cand)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            # Discovery failure is recorded by the caller's ledger; one local
            # source must not suppress the other source ladder entries.
            pass
        return candidates

    def _metasploit(self, cve_id: str) -> list[ExploitCandidate]:
        out: list[ExploitCandidate] = []
        assert self.msf is not None
        for result in self.msf.search(cve_id):
            module = str(result.get("fullname") or result.get("name") or "")
            if not module.startswith("exploit/"):
                continue
            info = self.msf.info(module) or {}
            revision, sha256 = self.msf.revision(module)
            spec = MetasploitSpec(cve_id=cve_id, module_name=module,
                                  rank=str(info.get("rank") or result.get("rank") or "unknown"),
                                  check_supported=bool(info.get("check") or info.get("check_supported")),
                                  capability="session" if info.get("session_types") else "code_execution")
            cand = collect_metasploit(spec)
            cand.provenance.revision, cand.provenance.sha256 = revision or module, sha256
            cand.candidate_id = derive_candidate_id(
                kind=cand.kind, cve_id=cand.cve_id, locator=module, provenance=cand.provenance)
            cand.runtime_kind = "metasploit_rpc"
            cand.produces_session = bool(info.get("session_types"))
            cand.bindings = {str(k): {"required": bool(v.get("required")), "type": "string"}
                             for k, v in (info.get("options") or {}).items() if isinstance(v, dict)}
            cand.requirements = {"binaries": ["msfrpcd"]}
            cand.expected_evidence = ["vulnerability_confirmation", "session_verified", "task_proof"]
            cand.extra.update({"module_info": info, "search_result": result})
            out.append(cand)
        return out

    @staticmethod
    def _interpreter(language: str) -> str:
        return {"python": "python3", "shell": "bash", "perl": "perl", "ruby": "ruby"}.get(language, "sh")


# ── Closed-registry compilation (paper §3.3) ─────────────────────────────────

from src.retrieval.registry import (  # noqa: E402
    CompiledProcedure as CompiledProcedure,
    RegistrationError as RegistrationError,
    compile_procedure as compile_procedure,
)

