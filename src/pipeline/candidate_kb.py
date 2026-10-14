"""The candidate knowledge base ``K`` (paper §3.2) as an experiment input.

``K`` is public corpus data — ExploitDB entries, Nuclei templates, public PoCs,
vendor recipes — never case truth. The runtime loads it before the graph starts
and hands it to the pipeline as ``exploit_candidate_specs``; a candidate is only
collected when the product-keyed retrieval returns the CVE its spec names, so the
knowledge base never bypasses retrieval.

Format: JSON Lines (one spec object per line) or a JSON array of spec objects.
Each entry needs ``kind`` and ``cve_id``; the remaining keys are the spec fields
the collectors read (``template_id``, ``bindings``, ``prerequisites``, …).

Loading fails closed: a missing path yields an empty ``K`` (a run without a
corpus is legitimate and simply has no candidates), while a file that exists but
is malformed raises, because a truncated or corrupted corpus must not silently
degrade into an empty one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Where the deployment mounts the corpus, in order of precedence.
DEFAULT_KB_PATHS: tuple[str, ...] = (
    "/run/veriplanpt/candidate-kb.jsonl",
    "/run/veriplanpt/candidate-kb.json",
)
#: Explicit override.
KB_ENV = "PENTEST_CANDIDATE_KB"


def resolve_kb_path(path: str | Path | None = None) -> Path | None:
    """The corpus path to load, or ``None`` when the run has no corpus."""
    if path is not None:
        return Path(path)
    override = str(os.environ.get(KB_ENV) or "").strip()
    if override:
        return Path(override)
    for candidate in DEFAULT_KB_PATHS:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def _spec_objects(payload: str, *, source: Path) -> list[dict[str, Any]]:
    text = payload.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate knowledge base {source} is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"candidate knowledge base {source} must be a JSON array of specs")
        return [dict(item) for item in data]
    specs: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"candidate knowledge base {source} line {number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(item, dict):
            raise ValueError(f"candidate knowledge base {source} line {number} is not a spec object")
        specs.append(dict(item))
    return specs


def load_candidate_kb(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ``K`` from the experiment's corpus file.

    Returns an empty list when no corpus is mounted. Raises ``ValueError`` when a
    corpus exists but cannot be read or does not contain spec objects.
    """
    resolved = resolve_kb_path(path)
    if resolved is None or not resolved.is_file():
        return []
    try:
        payload = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"candidate knowledge base {resolved} is unreadable: {exc}") from exc
    specs = _spec_objects(payload, source=resolved)
    for spec in specs:
        if not str(spec.get("kind") or ""):
            raise ValueError(f"candidate knowledge base {resolved} has a spec without a kind")
        if not str(spec.get("cve_id") or ""):
            raise ValueError(f"candidate knowledge base {resolved} has a spec without a cve_id")
    return specs
