"""
Preflight checks and workspace-local setup planning for execution.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

from src.execution.placeholders import extract_placeholder_names, render_commands

logger = logging.getLogger(__name__)

_SYSTEM_INSTALL_RE = re.compile(r"(?i)\b(?:sudo\s+)?(?:apt(?:-get)?|yum|dnf|apk|pacman)\s+install\b")
_PIP_INSTALL_RE = re.compile(r"(?i)\bpip(?:3)?\s+install\b")
_NPM_INSTALL_RE = re.compile(r"(?i)\bnpm\s+install\b")


def _has_unmatched_quotes(command: str) -> bool:
    """Check for unmatched single or double quotes."""
    in_single = False
    in_double = False
    for ch in command:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
    return in_single or in_double


def _is_malformed_fragment(command: str) -> bool:
    """Detect dangling loop terminators, unmatched quotes, or bare done."""
    stripped = command.strip()
    if re.match(r"^done\s*$", stripped):
        return True
    if re.search(r";\s*done\s*$", stripped):
        return True
    if _has_unmatched_quotes(stripped):
        return True
    return False


def _contains_foreign_ip(command: str, target_ip: str, attacker_ip: str) -> bool:
    """Check if a rendered command contains a literal IP that is not target, attacker, or localhost."""
    ips = set(re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", command))
    allowed = {ip for ip in (target_ip, attacker_ip, "127.0.0.1", "0.0.0.0") if ip}
    return bool(ips - allowed)


def _extract_foreign_ips(command: str, target_ip: str, attacker_ip: str) -> set[str]:
    """Return the set of offending literal IPs not belonging to target/attacker/localhost."""
    ips = set(re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", command))
    allowed = {ip for ip in (target_ip, attacker_ip, "127.0.0.1", "0.0.0.0") if ip}
    return ips - allowed


def _filter_commands(commands: list[str]) -> list[str]:
    """Drop malformed shell fragments from a command list."""
    return [cmd for cmd in commands if not _is_malformed_fragment(cmd)]


@dataclass
class PreflightResult:
    status: str
    reason: str = ""
    workspace_dir: str = ""
    working_directory: str = ""
    rendered_commands: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    required_placeholders: list[str] = field(default_factory=list)
    missing_placeholders: list[str] = field(default_factory=list)
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_working_directory(exploit: dict[str, Any]) -> str:
    explicit = str(exploit.get("working_directory", "") or "").strip()
    path = str(exploit.get("file_path", "") or "").strip()
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        if path:
            base = path if os.path.isdir(path) else os.path.dirname(path)
            return os.path.abspath(os.path.join(base or ".", explicit))
        return os.path.abspath(explicit)
    if not path:
        return ""
    return path if os.path.isdir(path) else os.path.dirname(path)


def _requirements_file(working_directory: str) -> str:
    if not working_directory:
        return ""
    for name in ("requirements.txt", "requirements-dev.txt"):
        path = os.path.join(working_directory, name)
        if os.path.exists(path):
            return path
    return ""


def _prefix_workdir(command: str, working_directory: str) -> str:
    if not working_directory:
        return command
    stripped = command.strip()
    if stripped.startswith("cd "):
        return stripped
    return f"cd {shlex.quote(working_directory)} && {stripped}"


def _derive_setup_commands(
    exploit: dict[str, Any],
    working_directory: str,
    workspace_dir: str,
    allow_workspace_installs: bool,
) -> tuple[list[str], str]:
    raw_setup = list(exploit.get("setup_commands", [])) or list(exploit.get("dependencies", []))
    if not raw_setup:
        requirements = _requirements_file(working_directory)
        candidate_commands = list(exploit.get("commands", []))
        if requirements and any(str(cmd).strip().startswith(("python ", "python3 ")) for cmd in candidate_commands):
            raw_setup = [f"pip install -r {requirements}"]
    if not raw_setup:
        return [], ""

    if not allow_workspace_installs:
        return [], "Workspace installs are disabled."

    venv_dir = os.path.join(workspace_dir, ".venv")
    os.path.join(venv_dir, "bin", "python")
    venv_pip = os.path.join(venv_dir, "bin", "pip")
    commands: list[str] = []
    venv_bootstrap_added = False

    for raw in raw_setup:
        line = str(raw or "").strip()
        if not line:
            continue
        if _SYSTEM_INSTALL_RE.search(line):
            return [], f"Dependency setup requires system package install: {line}"
        if _PIP_INSTALL_RE.search(line):
            if not venv_bootstrap_added:
                commands.extend([
                    f"python3 -m venv {shlex.quote(venv_dir)}",
                    f"{shlex.quote(venv_pip)} install --upgrade pip",
                ])
                venv_bootstrap_added = True
            install_args = line.split("install", 1)[1].strip()
            if install_args.startswith("-r "):
                req_path = install_args[3:].strip()
                if req_path and not os.path.isabs(req_path):
                    install_args = f"-r {shlex.quote(os.path.join(working_directory, req_path))}"
            if install_args:
                commands.append(f"{shlex.quote(venv_pip)} install {install_args}")
            continue
        if _NPM_INSTALL_RE.search(line) and " -g" not in f" {line} ":
            commands.append(f"npm install --prefix {shlex.quote(working_directory or workspace_dir)} {line.split('install', 1)[1].strip()}".strip())
            continue
        commands.append(_prefix_workdir(line, working_directory))
    return commands, ""


def prepare_candidate(
    exploit: dict[str, Any],
    workspace_dir: str,
    placeholder_values: dict[str, str],
    missing_placeholders: list[str],
    allow_workspace_installs: bool = True,
) -> PreflightResult:
    file_path = str(exploit.get("file_path", "") or "").strip()
    if not file_path or not os.path.exists(file_path):
        return PreflightResult(
            status="preflight_failed",
            reason=f"Exploit path missing: {file_path or '(empty)'}",
            workspace_dir=workspace_dir,
        )

    os.makedirs(workspace_dir, exist_ok=True)
    working_directory = _default_working_directory(exploit)
    if working_directory and not os.path.exists(working_directory):
        return PreflightResult(
            status="preflight_failed",
            reason=f"Working directory missing: {working_directory}",
            workspace_dir=workspace_dir,
            working_directory=working_directory,
        )

    required_placeholders = extract_placeholder_names(
        exploit.get("commands", []),
        exploit.get("verify_commands", []),
        exploit.get("required_placeholders", []),
        exploit.get("placeholders", []),
    )
    missing = [name for name in required_placeholders if name in missing_placeholders]
    if missing:
        return PreflightResult(
            status="preflight_failed",
            reason=f"Missing placeholders: {', '.join(missing)}",
            workspace_dir=workspace_dir,
            working_directory=working_directory,
            required_placeholders=required_placeholders,
            missing_placeholders=missing,
        )

    rendered_commands = [
        _prefix_workdir(command, working_directory)
        for command in render_commands(
            _filter_commands(list(exploit.get("commands", []))),
            placeholder_values,
        )
    ]
    if not rendered_commands:
        return PreflightResult(
            status="ready",
            reason="No structured commands available; LLM fallback may be needed.",
            workspace_dir=workspace_dir,
            working_directory=working_directory,
            required_placeholders=required_placeholders,
        )

    setup_commands, setup_reason = _derive_setup_commands(
        exploit,
        working_directory,
        workspace_dir,
        allow_workspace_installs=allow_workspace_installs,
    )
    if setup_reason:
        return PreflightResult(
            status="blocked",
            reason=setup_reason,
            workspace_dir=workspace_dir,
            working_directory=working_directory,
            rendered_commands=rendered_commands,
            required_placeholders=required_placeholders,
        )

    verify_commands = [
        _prefix_workdir(command, working_directory)
        for command in render_commands(list(exploit.get("verify_commands", [])), placeholder_values)
    ]

    # Validate rendered commands against foreign IP injection
    target_ip = str(exploit.get("target_ip") or "").strip()
    attacker_ip = str(exploit.get("attacker_ip") or "").strip()
    validated_commands = []
    for cmd in rendered_commands:
        if target_ip or attacker_ip:
            if _contains_foreign_ip(cmd, target_ip, attacker_ip):
                foreign = _extract_foreign_ips(cmd, target_ip, attacker_ip)
                logger.warning("Dropping command with foreign IP(s) %s", sorted(foreign))
                continue
        validated_commands.append(cmd)
    if rendered_commands and not validated_commands:
        return PreflightResult(
            status="invalid_command",
            reason="All rendered commands contain foreign literal IPs.",
            workspace_dir=workspace_dir,
            working_directory=working_directory,
            required_placeholders=required_placeholders,
        )

    return PreflightResult(
        status="ready",
        workspace_dir=workspace_dir,
        working_directory=working_directory,
        rendered_commands=validated_commands,
        setup_commands=setup_commands,
        verify_commands=verify_commands,
        required_placeholders=required_placeholders,
        success_indicators=list(exploit.get("success_indicators", [])),
        failure_indicators=list(exploit.get("failure_indicators", [])),
    )
