"""
src/agents/critic.py
────────────────────
Model-based critique of a single compiled proposal (paper §3.3).

The critique is advisory only: model-generated text can block a proposal but
can never authorize one. With no model configured the critique reports nothing
and the deterministic audit alone decides; a model that fails is reported as a
finding (fail closed).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


def _parse_challenges(text: str) -> list[str]:
    """Parse a JSON array of challenge strings from LLM output."""
    import json
    import re

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in items if isinstance(item, str)]


#: A critique that states there is no objection is not a finding: only a raised
#: objection may block a proposal (paper §3.3 keeps the model able to block, not
#: to authorize). Free-weights models sometimes narrate the absence of a problem
#: inside the array instead of returning an empty one.
_NO_OBJECTION_RE = re.compile(
    r"^\s*(?:no|none)\b[^.]{0,40}objection"
    r"|no concrete objection"
    r"|nothing to flag"
    r"|no issues? (?:found|identified|to report)",
    re.IGNORECASE,
)


def findings_from_reply(text: str) -> list[str]:
    """Objections parsed from a critique reply, with no-objection notes dropped."""
    return [item for item in _parse_challenges(text) if not _NO_OBJECTION_RE.search(item)]


def critique(proposal: Mapping[str, Any], *, llm: Any | None = None) -> list[str]:
    """Model-based critique of one compiled proposal (paper §3.3).

    The critique is collected before the deterministic audit and its findings
    fold into the same ``reject``/``recover`` outcomes: model-generated text can
    block a proposal but can never authorize one. With no model configured the
    critique reports nothing and the deterministic audit alone decides; a model
    that fails is reported as a finding (fail closed).
    """
    if llm is None:
        return []
    scope = proposal.get("scope") or {}
    description = (
        "You are a security critic reviewing ONE proposed penetration-testing step that has "
        "already passed the deterministic pre-dispatch audit.\n"
        "The step targets an AUTHORIZED assessment. Raise an objection ONLY when it is concrete "
        "and grounded in the command, its payload, or a precondition that the world state does "
        "not satisfy. Do NOT object to the host, port, scheme or template being present, and do "
        "NOT object merely because the command is an HTTP request.\n"
        "The step is dispatched as an argv array and executed directly, never through a shell: "
        "there is no word splitting, globbing, or quote removal. Do NOT object that an argument "
        "is unquoted or contains spaces, newlines or double quotes — such a payload reaches the "
        "tool verbatim.\n"
        "Do NOT object that a step reads a payload artifact the procedure itself declares (an "
        "argument such as `@<path>`, or a request body read from a file): the run materializes "
        "declared payloads under its own artifact root, which is inside the authorized scope, so "
        "reading one is not an out-of-scope side effect.\n"
        "Reply with a JSON array of short strings. Reply with an empty array [] when there is no "
        "such objection.\n\n"
        f"Candidate: {proposal.get('candidate_id', '')}\n"
        f"Template: {proposal.get('template_id', '')}\n"
        f"Family: {proposal.get('family', '')}\n"
        f"Command (argv array, executed without a shell): "
        f"{json.dumps([str(token) for token in (proposal.get('argv') or ())])}\n"
        f"Rationale: {proposal.get('rationale', '')}\n"
        f"Authorized host: {proposal.get('authorized_host', '')}\n"
        f"Authorized port: {proposal.get('authorized_port', '')}\n"
        f"Authorized families: {', '.join(str(item) for item in (proposal.get('authorized_families') or ()))}\n"
        f"Authorized scope: hosts={scope.get('allowed_hosts')} networks={scope.get('allowed_networks')} "
        f"ports={scope.get('allowed_ports')}\n"
        f"Local artifact root (the run's own directory): {proposal.get('artifact_root', '')}\n"
        f"World-state grounding: {proposal.get('grounding', '')}\n"
        f"Missing prerequisites: {', '.join(str(item) for item in (proposal.get('missing_prerequisites') or ())) or 'none'}\n"
    )
    try:
        response = llm.invoke(description)
    except Exception as exc:  # pragma: no cover - defensive: fail closed
        return [f"critic-unavailable: {type(exc).__name__}"]
    text = getattr(response, "content", response)
    return findings_from_reply(str(text))
