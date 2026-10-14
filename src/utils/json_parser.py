"""
src/utils/json_parser.py
────────────────────────
Single shared JSON extraction utility.
Fixes P3 (duplication): extract_json_data() was copy-pasted in
recon_agent.py and execution_agent.py. One place now.
"""

from __future__ import annotations

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict | list]:
    """
    Extract the first JSON object or array from *text*.

    Strategy:
    1. Strip ```json ... ``` fences.
    2. Try json.loads on the stripped text.
    3. Fall back to a greedy regex search for {...}.
    Returns None if nothing parsable is found.
    """
    if not text:
        return None

    # 1. Remove markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 2. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Greedy search for object or array
    for pattern in (r"(\{.*\})", r"(\[.*\])"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def require_json(text: str, keys: list[str] | None = None) -> tuple[Optional[dict], str]:
    """
    Parse JSON and optionally validate required keys.

    Returns:
        (parsed_dict, error_message)
        error_message is "" on success.
    """
    parsed = extract_json(text)
    if parsed is None:
        return None, "Response did not contain valid JSON."
    if not isinstance(parsed, dict):
        return None, f"Expected a JSON object, got {type(parsed).__name__}."
    if keys:
        missing = [k for k in keys if k not in parsed]
        if missing:
            return None, f"JSON missing required keys: {missing}"
    return parsed, ""
