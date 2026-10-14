"""
Snippet-level procedure extraction for shortlisted candidates.
"""

from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config
from src.retrieval.models import PocCandidate, ProcedureSnippet

_COMMAND_RE = re.compile(
    r"(?im)^(?:\$ |# |> )?((?:python|python3|bash|sh|ruby|perl|java|php|msfconsole|curl|wget)\b[^\n]*)"
)
_DEP_RE = re.compile(r"(?im)\b(?:pip(?:3)? install|apt(?:-get)? install|gem install|go install)\b[^\n]*")
_PLACEHOLDER_RE = re.compile(r"(\{\{[^}]+\}\}|<[A-Z0-9_:-]+>|RHOSTS?|LHOST|RPORT|LPORT|TARGET(?:_IP)?|USERNAME|PASSWORD)")
_ASSUMPTION_RE = re.compile(r"(?im)\b(windows|linux|apache|nginx|mysql|postgres|authenticated|credential|reverse shell)\b")
_VERIFY_RE = re.compile(r"(?im)^(?:\$ |# |> )?((?:id|whoami|curl|wget)\b[^\n]*)")
_SUCCESS_RE = re.compile(r"(?im)\b(uid=|root@|www-data@|meterpreter >|flag\{|htb\{|access granted|pwned)\b")
_FAILURE_RE = re.compile(r"(?im)\b(not vulnerable|patched|permission denied|unauthorized|connection refused|modulenotfounderror|syntaxerror)\b")


def _normalize_placeholder(token: str) -> str:
    cleaned = token.strip().strip("{}<>").strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1]
    return cleaned.upper()


def _derive_working_directory(candidate: PocCandidate) -> str:
    if not candidate.path:
        return ""
    return candidate.path if os.path.isdir(candidate.path) else os.path.dirname(candidate.path)


def _read_candidate_text(candidate: PocCandidate) -> str:
    if not candidate.path:
        return ""
    path = candidate.path
    chunks: list[str] = []
    if os.path.isfile(path):
        paths = [path]
    else:
        names = sorted(os.listdir(path))[:20]
        paths = [os.path.join(path, name) for name in names]
    for file_path in paths:
        if not os.path.isfile(file_path):
            continue
        if os.path.getsize(file_path) > 250_000:
            continue
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read(50_000)
        except Exception:
            continue
        chunks.append(f"\n# FILE: {os.path.basename(file_path)}\n{text}")
    return "\n".join(chunks)


def _regex_extract(candidate: PocCandidate, text: str) -> ProcedureSnippet:
    commands = list(dict.fromkeys(match.strip() for match in _COMMAND_RE.findall(text)))
    dependencies = list(dict.fromkeys(match.strip() for match in _DEP_RE.findall(text)))
    placeholders = list(dict.fromkeys(match.strip() for match in _PLACEHOLDER_RE.findall(text)))
    normalized_placeholders = list(dict.fromkeys(_normalize_placeholder(item) for item in placeholders if item.strip()))
    assumptions = list(dict.fromkeys(match.strip().lower() for match in _ASSUMPTION_RE.findall(text)))
    verify_commands = list(dict.fromkeys(match.strip() for match in _VERIFY_RE.findall(text)))
    success_indicators = list(dict.fromkeys(match.strip() for match in _SUCCESS_RE.findall(text)))
    failure_indicators = list(dict.fromkeys(match.strip() for match in _FAILURE_RE.findall(text)))
    notes: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if "usage" in lower or "example" in lower or "requires" in lower:
            stripped = line.strip()
            if stripped:
                notes.append(stripped[:200])
        if len(notes) >= 6:
            break
    confidence = 0.25
    if commands:
        confidence += 0.35
    if dependencies:
        confidence += 0.15
    if notes:
        confidence += 0.15
    if placeholders:
        confidence += 0.10
    return ProcedureSnippet(
        candidate_id=candidate.candidate_id,
        commands=commands[:10],
        dependencies=dependencies[:10],
        placeholders=placeholders[:12],
        required_placeholders=normalized_placeholders[:12],
        target_assumptions=assumptions[:8],
        usage_notes=notes[:8],
        working_directory=_derive_working_directory(candidate),
        setup_commands=dependencies[:10],
        verify_commands=verify_commands[:5],
        success_indicators=success_indicators[:8],
        failure_indicators=failure_indicators[:8],
        confidence=round(min(confidence, 0.95), 3),
    )


def _llm_fallback(candidate: PocCandidate, text: str) -> ProcedureSnippet | None:
    if not text.strip():
        return None
    try:
        cfg = get_config()
        llm = cfg.get_llm(cfg.planning["model"])
        response = llm.invoke([
            SystemMessage(content=(
                "Extract exploit execution procedure. Return JSON only with keys "
                "commands, dependencies, placeholders, required_placeholders, "
                "target_assumptions, usage_notes, working_directory, setup_commands, "
                "verify_commands, success_indicators, failure_indicators."
            )),
            HumanMessage(content=text[:12000]),
        ], stream=False)
        from src.utils.json_parser import extract_json

        parsed = extract_json(getattr(response, "content", "") or "")
        if not isinstance(parsed, dict):
            return None
        return ProcedureSnippet(
            candidate_id=candidate.candidate_id,
            commands=list(parsed.get("commands", []))[:10],
            dependencies=list(parsed.get("dependencies", []))[:10],
            placeholders=list(parsed.get("placeholders", []))[:12],
            required_placeholders=list(parsed.get("required_placeholders", parsed.get("placeholders", [])))[:12],
            target_assumptions=list(parsed.get("target_assumptions", []))[:8],
            usage_notes=list(parsed.get("usage_notes", []))[:8],
            working_directory=str(parsed.get("working_directory", _derive_working_directory(candidate))),
            setup_commands=list(parsed.get("setup_commands", parsed.get("dependencies", [])))[:10],
            verify_commands=list(parsed.get("verify_commands", []))[:5],
            success_indicators=list(parsed.get("success_indicators", []))[:8],
            failure_indicators=list(parsed.get("failure_indicators", []))[:8],
            confidence=0.45,
        )
    except Exception:
        return None


def extract_procedure_snippets(
    candidates: list[PocCandidate],
    economic_mode: bool = True,
    allow_llm_fallback: bool = True,
) -> list[ProcedureSnippet]:
    snippets: list[ProcedureSnippet] = []
    for candidate in candidates:
        text = _read_candidate_text(candidate)
        snippet = _regex_extract(candidate, text)
        if not snippet.commands and not snippet.dependencies and economic_mode and allow_llm_fallback:
            fallback = _llm_fallback(candidate, text)
            if fallback is not None:
                snippet = fallback
        snippets.append(snippet)
    return snippets
