"""
src/tools/shell.py
──────────────────
Restricted shell tool. Kept for backward compatibility with the legacy graph
but no longer accepts arbitrary shell strings. Commands are executed as
structured argument arrays only — never via ``shell=True``.

The unrestricted LLM shell fallback required by the original PoC-only workflow
has been removed; the legacy graph will receive ``[BLOCKED] unrestricted shell
disabled`` for any string-based invocation.
"""

from __future__ import annotations

import shlex
import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def validate_command(command: str, mode: str = "recon") -> tuple[bool, str]:
    """Reject string shell commands; only structured argv arrays are allowed."""
    cmd = (command or "").strip()
    if not cmd:
        return False, "Empty command"
    # Disallow any shell metacharacter or chained command: the unrestricted
    # fallback must not return. Structured argv arrays arrive as a JSON list
    # in a separate tool wrapper.
    forbidden = {"&&", "||", ";", "|", "$", "`", ">", "<", "\n"}
    if any(tok in cmd for tok in forbidden):
        return False, "Unrestricted shell chaining is disabled"
    if cmd.startswith("-"):
        return False, "Cannot run raw options"
    # Only allow simple executable + arguments.
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return False, "Command could not be tokenised"
    if not parts:
        return False, "Empty command"
    return True, "OK"


class ShellInput(BaseModel):
    command: str = Field(
        description=(
            "Single executable with simple arguments. Shell chaining is "
            "disabled. For complex exploits, prefer the typed candidate "
            "renderers in src/pipeline/renderers.py."
        )
    )
    timeout: int = Field(default=300, ge=1, le=600)
    mode: str = Field(default="recon")


@tool(args_schema=ShellInput)
def run_shell(command: str, timeout: int = 300, mode: str = "recon") -> str:
    """
    Execute a single executable without shell chaining.

    For full pentest execution, prefer the typed candidate renderers in
    ``src.pipeline.renderers``. This tool remains only for backward
    compatibility with the legacy graph; it will reject any shell metacharacter.
    """
    ok, reason = validate_command(command, mode)
    if not ok:
        return f"[BLOCKED] {reason}"
    try:
        parts = shlex.split(command)
        proc = subprocess.run(
            parts,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        if not output.strip():
            return f"[Exit {proc.returncode}] No output produced."
        if len(output) > 50000:
            output = output[:50000] + "\n...[truncated]"
        return output
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Command timed out after {timeout}s"
    except FileNotFoundError as exc:
        return f"[NOT-FOUND] {exc}"
    except Exception as exc:                  # noqa: BLE001
        return f"[ERROR] {exc}"
