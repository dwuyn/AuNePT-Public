"""
src/utils/tool_compat.py
────────────────────────
Compatibility helpers for OpenAI-compatible backends that advertise tools but
sometimes return JSON/text instructions instead of structured tool_calls.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from uuid import uuid4

from src.utils.json_parser import extract_json


def content_to_text(content: Any) -> str:
    """Flatten LangChain message content into plain text for JSON parsing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


KNOWN_PENTEST_TOOLS = frozenset({
    "nmap", "curl", "nc", "ncat", "nikto", "sqlmap", "gobuster", "dirb",
    "hydra", "nuclei", "msfconsole", "searchsploit", "feroxbuster",
    "wfuzz", "medusa", "python3", "python", "bash", "sh", "wget", "netcat",
    "ping", "traceroute", "whois", "dig", "host", "nslookup",
})

_CMD_TAG_RE = re.compile(r"<(?:CMD|cmd|command|tool_call)>\s*(.*?)\s*</(?:CMD|cmd|command|tool_call)>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:bash|sh|shell|zsh)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def _extract_plain_text_commands(text: str) -> list[str]:
    """Extract shell commands from plain text / markdown representations."""
    commands: list[str] = []

    # 1. XML-style command tags (<CMD>...</CMD>, <command>...</command>, etc.)
    for match in _CMD_TAG_RE.finditer(text):
        cmd = match.group(1).strip()
        if cmd:
            commands.append(cmd)

    # 2. Markdown fenced code blocks (```bash ... ```)
    for match in _CODE_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("$ ") or line.startswith("# "):
                line = line[2:].strip()
            if line and not line.startswith("#") and not line.startswith("//"):
                first_token = line.split()[0].lower() if line.split() else ""
                if first_token in KNOWN_PENTEST_TOOLS or any(line.startswith(t + " ") for t in KNOWN_PENTEST_TOOLS):
                    commands.append(line)
                elif len(block.splitlines()) == 1 and line:
                    commands.append(line)

    # 3. Inline backtick commands (`nmap -sT ...`)
    for match in _INLINE_CODE_RE.finditer(text):
        cmd = match.group(1).strip()
        if cmd.startswith("$ ") or cmd.startswith("# "):
            cmd = cmd[2:].strip()
        first_token = cmd.split()[0].lower() if cmd.split() else ""
        if first_token in KNOWN_PENTEST_TOOLS:
            commands.append(cmd)

    # 4. Line-by-line fallback for prompts outputting bare command lines
    if not commands:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("$ ") or line.startswith("# "):
                line = line[2:].strip()
            if not line or line.startswith("//") or line.startswith("*") or line.startswith("-"):
                continue
            first_token = line.split()[0].lower() if line.split() else ""
            if first_token in KNOWN_PENTEST_TOOLS:
                commands.append(line)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for cmd in commands:
        cleaned = cmd.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def extract_tool_calls_from_content(
    content: Any,
    allowed_tools: Iterable[str],
    *,
    default_tool_name: str | None = None,
    default_args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse tool intents from message content when the backend failed to populate
    response.tool_calls.

    Supported fallback formats:
    - {"tool": "run_shell", "args": {...}}
    - {"name": "search_cve", "arguments": {...}}
    - {"tool_calls": [{...}, {...}]}
    - {"executable": "<command>"} for legacy run_shell-style flows
    - Markdown code blocks (```bash ... ```)
    - XML command tags (<CMD>...</CMD>)
    - Plain text commands with known tool prefixes
    """
    allowed = set(allowed_tools)
    raw_text = content_to_text(content)
    parsed = extract_json(raw_text)

    tool_calls: list[dict[str, Any]] = []

    if parsed is not None:
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            raw_calls = parsed["tool_calls"]
        elif isinstance(parsed, list):
            raw_calls = parsed
        else:
            raw_calls = [parsed]

        for raw_call in raw_calls:
            normalized = _normalize_tool_call(
                raw_call,
                allowed,
                default_tool_name=default_tool_name,
                default_args=default_args,
            )
            if normalized:
                tool_calls.append(normalized)

    # If JSON parsing yielded no valid tool calls and default_tool_name is allowed,
    # attempt multi-pattern plain-text extraction
    if not tool_calls and default_tool_name and default_tool_name in allowed:
        commands = _extract_plain_text_commands(raw_text)
        for cmd in commands:
            args = {"command": cmd}
            if default_args:
                for k, v in default_args.items():
                    args.setdefault(k, v)
            tool_calls.append({
                "id": f"compat-{uuid4()}",
                "name": default_tool_name,
                "args": args,
            })

    return tool_calls


def _normalize_tool_call(
    raw_call: Any,
    allowed_tools: set[str],
    *,
    default_tool_name: str | None,
    default_args: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(raw_call, dict):
        return None

    name: str | None = None
    args: Any = None

    if isinstance(raw_call.get("function"), dict):
        fn = raw_call["function"]
        name = fn.get("name")
        args = fn.get("arguments")

    if not name:
        for key in ("tool", "tool_name", "name"):
            value = raw_call.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break

    if args is None:
        for key in ("args", "arguments", "parameters", "kwargs", "input"):
            if key in raw_call:
                args = raw_call[key]
                break

    if name is None and default_tool_name:
        command = raw_call.get("command") or raw_call.get("executable")
        if isinstance(command, str) and command.strip() and command.strip().lower() != "none":
            name = default_tool_name
            args = {"command": command.strip()}

    if name not in allowed_tools:
        return None

    normalized_args = _normalize_args(args)
    if default_args:
        for key, value in default_args.items():
            normalized_args.setdefault(key, value)

    if name == default_tool_name:
        command = normalized_args.get("command") or raw_call.get("command") or raw_call.get("executable")
        if not isinstance(command, str) or not command.strip() or command.strip().lower() == "none":
            return None
        normalized_args["command"] = command.strip()
        if "timeout" not in normalized_args and "timeout" in raw_call:
            normalized_args["timeout"] = raw_call["timeout"]
        if "mode" not in normalized_args and "mode" in raw_call:
            normalized_args["mode"] = raw_call["mode"]

    return {
        "id": f"compat-{uuid4()}",
        "name": name,
        "args": normalized_args,
    }


def _normalize_args(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        parsed = extract_json(raw_args)
        if isinstance(parsed, dict):
            return parsed
    return {}
