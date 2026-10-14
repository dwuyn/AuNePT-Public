"""
src/pipeline/renderers.py
─────────────────────────
Structured argument-array renderers for each candidate kind.

Renderers never produce a free-form shell command; they emit
``list[str]`` argv arrays with placeholders resolved against a
typed values dictionary. Unresolved placeholders raise.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Mapping

from src.pipeline.candidates import (
    ExploitCandidate,
    ProcedureStep,
    substitute_placeholders,
)


class RenderError(Exception):
    """Raised when a procedure cannot be rendered (unresolved placeholder, etc.)."""


@dataclass
class RenderedStep:
    stage: str
    argv: list[str]
    timeout_seconds: int
    env: dict[str, str] = None  # type: ignore[assignment]


def render_procedure(
    candidate: ExploitCandidate,
    *,
    values: Mapping[str, str],
    working_dir: str = "",
    msf_cfgroot: str = "",
    nuclei_output_dir: str = "",
    ledger=None,
) -> list[RenderedStep]:
    """Render every step in *candidate.procedure* into concrete argv arrays."""
    if not candidate.procedure:
        raise RenderError(f"candidate {candidate.candidate_id} has no procedure")
    rendered: list[RenderedStep] = []
    for step in candidate.procedure:
        argv = list(step.argv)
        # Substitute placeholders inside the structured argv tokens.
        argv, unresolved = substitute_placeholders(argv, values, strict=False)
        # Renderer-specific rewrites for framework placeholders.
        argv = _rewrite_framework_placeholders(
            argv, candidate=candidate, step=step,
            working_dir=working_dir,
            msf_cfgroot=msf_cfgroot,
            nuclei_output_dir=nuclei_output_dir,
        )
        # Re-run placeholder substitution in case the renderer injected new ones.
        argv, unresolved = substitute_placeholders(argv, values, strict=True)
        if unresolved:
            raise RenderError(
                f"Unresolved placeholders in {candidate.candidate_id} stage {step.stage}: {unresolved}")
        rendered.append(RenderedStep(
            stage=step.stage, argv=argv, timeout_seconds=step.timeout_seconds,
            env={"MSF_CFGROOT_CONFIG": msf_cfgroot} if (msf_cfgroot and candidate.kind == "metasploit") else {},
        ))
    return rendered


def _rewrite_framework_placeholders(argv: list[str], *,
                                      candidate: ExploitCandidate,
                                      step: ProcedureStep,
                                      working_dir: str,
                                      msf_cfgroot: str,
                                      nuclei_output_dir: str,
                                      ) -> list[str]:
    out: list[str] = []
    for tok in argv:
        if tok == "<MSF_RC>" and candidate.kind == "metasploit":
            script = candidate.extra.get("resource_script", "")
            if not script:
                raise RenderError("metasploit candidate missing resource_script")
            if not msf_cfgroot:
                raise RenderError("metasploit step requires MSF_CFGROOT_CONFIG")
            rc_path = os.path.join(msf_cfgroot, "msf_run.rc")
            os.makedirs(msf_cfgroot, exist_ok=True)
            with open(rc_path, "w") as fh:
                fh.write(script)
            out.append(rc_path)
            continue
        if tok == "<MSF_RC_VERIFY>" and candidate.kind == "metasploit":
            verify_script = _build_msf_verify_script(candidate)
            rc_path = os.path.join(msf_cfgroot or tempfile.mkdtemp(prefix="msf_"),
                                     "msf_verify.rc")
            os.makedirs(os.path.dirname(rc_path), exist_ok=True)
            with open(rc_path, "w") as fh:
                fh.write(verify_script)
            out.append(rc_path)
            continue
        if tok == "<NUCLEI_OUTPUT>" and candidate.kind == "nuclei":
            os.makedirs(nuclei_output_dir or working_dir, exist_ok=True)
            out.append(os.path.join(nuclei_output_dir or working_dir,
                                     f"{candidate.cve_id}.json"))
            continue
        # guided_procedure: <FRAMEWORK_COMMAND> is a sentinel that gets
        # stripped — the actual command is in the remaining argv tokens.
        if tok == "<FRAMEWORK_COMMAND>" and candidate.kind == "guided_procedure":
            continue
        out.append(tok)
    return out


def _build_msf_verify_script(candidate: ExploitCandidate) -> str:
    """A run-local Metasploit verification script that runs ``check`` only."""
    module = candidate.locator
    options = candidate.extra.get("options", {}) if isinstance(candidate.extra, dict) else {}
    lines = [f"use {module}", "set verbose true"]
    for k, v in options.items():
        lines.append(f"set {k} {v}")
    if candidate.extra.get("check_supported", True):
        lines.append("check")
    lines.append("exit")
    return "\n".join(lines) + "\n"
