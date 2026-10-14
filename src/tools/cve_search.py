"""
src/tools/cve_search.py
───────────────────────
CVE / exploit search tools as proper LangChain @tool functions.
The planning agent binds these tools and calls them natively via
function calling — no JSON regex parsing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


# ── CVE lookup ────────────────────────────────────────────────────────────────

class CVESearchInput(BaseModel):
    query: str = Field(
        description="CVE ID (e.g. CVE-2021-41773) or product name to search"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results")


@tool(args_schema=CVESearchInput)
def search_cve(query: str, limit: int = 10) -> str:
    """
    Search the vulnerability database for CVEs matching a CVE ID or product name.
    Returns a JSON list with cve_id, description, cvss_score, and affected products.
    """
    try:
        result = subprocess.run(
            ["vulnx", "search", query, "--limit", str(limit), "-j"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return f"[ERROR] {result.stderr.strip()}"
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = data.get("results", [])
        if not isinstance(data, list):
            return "[ERROR] CVE search returned an unexpected JSON shape"
        return json.dumps(data[:limit], indent=2)
    except subprocess.TimeoutExpired:
        return "[ERROR] CVE search timed out after 50s"
    except json.JSONDecodeError:
        return "[ERROR] Could not parse CVE search response"
    except FileNotFoundError:
        return "[ERROR] vulnx not installed. Run: go install github.com/projectdiscovery/cvemap/cmd/cvemap@latest"
    except Exception as exc:
        return f"[ERROR] {exc}"


@tool
def get_cve_detail(cve_id: str) -> str:
    """
    Fetch detailed information for a specific CVE ID (description, CVSS,
    EPSS, affected products, PoC references).
    """
    try:
        result = subprocess.run(
            ["vulnx", "id", cve_id, "-j"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return f"[ERROR] {result.stderr.strip()}"
        data = json.loads(result.stdout)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return "[ERROR] CVE detail lookup returned an unexpected JSON shape"
        compact = {
            "cve_id": data.get("cve_id") or data.get("id") or cve_id,
            "title": data.get("title") or data.get("name") or cve_id,
            "description": data.get("description") or data.get("cve_description") or "",
            "cvss_score": data.get("cvss_score") or data.get("cvss", {}).get("score", 0.0),
            "epss_percentile": data.get("epss_percentile") or data.get("epss", {}).get("epss_percentile", 0.0),
            "weaknesses": data.get("weaknesses") or data.get("cwe") or [],
            "references": data.get("references") or data.get("citations") or [],
            "affected_versions": data.get("affected_versions") or data.get("affected_products") or data.get("affected") or [],
            "is_kev": data.get("is_kev", False),
            "is_poc": data.get("is_poc", False),
            "severity": data.get("severity") or "",
            "requirements": data.get("requirements") or "",
        }
        return json.dumps(compact, indent=2)
    except json.JSONDecodeError:
        return "[ERROR] Could not parse CVE detail response"
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── ExploitDB search ──────────────────────────────────────────────────────────

class ExploitDBInput(BaseModel):
    keyword: str = Field(description="CVE ID or keyword to search in ExploitDB")
    output_dir: str = Field(description="Directory to copy matching exploit files into")


@tool(args_schema=ExploitDBInput)
def search_exploitdb(keyword: str, output_dir: str) -> str:
    """
    Search local ExploitDB (searchsploit) for exploits matching a CVE ID or keyword.
    Copies matching exploit files to output_dir. Returns a summary of what was found.
    """
    try:
        from utils.searchers.exploitdb_searcher import ExploitDBSearcher  # noqa
        os.makedirs(output_dir, exist_ok=True)
        ExploitDBSearcher().search_keyword_local(keyword, output_dir)
        entries = [e for e in os.listdir(output_dir) if not e.startswith(".")]
        if not entries:
            return f"No exploits found for '{keyword}' in ExploitDB."
        return f"Found {len(entries)} exploit(s) in {output_dir}: {entries}"
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── GitHub search ─────────────────────────────────────────────────────────────

class GitHubInput(BaseModel):
    keyword: str = Field(description="CVE ID or keyword to search on GitHub")
    output_dir: str = Field(description="Directory to clone repositories into")


@tool(args_schema=GitHubInput)
def search_github(keyword: str, output_dir: str) -> str:
    """
    Search GitHub for repositories containing exploits for the given CVE or keyword.
    Clones the highest-scoring repositories into output_dir.
    Returns a list of cloned repositories.
    """
    try:
        from utils.searchers.github_searcher import GithubSearcher  # noqa
        os.makedirs(output_dir, exist_ok=True)
        GithubSearcher().search_keyword(keyword, output_dir)
        entries = [e for e in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, e))]
        if not entries:
            return f"No GitHub repositories found for '{keyword}'."
        return f"Cloned {len(entries)} repo(s) to {output_dir}: {entries}"
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── Version-range filter ──────────────────────────────────────────────────────

class VersionFilterInput(BaseModel):
    cvemap_results: str = Field(description="JSON string of cvemap results")
    target_version: str = Field(description="Target app version to filter by (e.g. 5.2.17)")


@tool(args_schema=VersionFilterInput)
def filter_cves_by_version(cvemap_results: str, target_version: str) -> str:
    """
    Given a JSON list of CVE entries and a target version string, return only
    the CVEs whose affected version range includes the target version.
    """
    try:
        from utils.version_limit import get_affected_cve  # noqa
        data = json.loads(cvemap_results)
        affected = get_affected_cve(data, target_version)
        return json.dumps(affected, indent=2)
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── Tool registry ─────────────────────────────────────────────────────────────

PLANNING_TOOLS = [search_cve, get_cve_detail, search_exploitdb, search_github, filter_cves_by_version]


def _safe_json_parse(payload: str, fallback: Any) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return fallback


def search_cve_records(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Structured helper used by the retrieval stack."""
    result = search_cve.invoke({"query": query, "limit": limit})
    if not isinstance(result, str):
        raise RuntimeError("CVE search returned non-string result")
    if result.startswith("[ERROR]"):
        raise RuntimeError(f"CVE search failed: {result}")
    try:
        parsed = json.loads(result)
    except Exception as exc:
        raise RuntimeError(f"CVE search returned invalid JSON payload: {exc}") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("results", [])
    if not isinstance(parsed, list):
        raise RuntimeError("CVE search returned invalid JSON payload (not a list)")
    return parsed


def get_cve_detail_record(cve_id: str) -> dict[str, Any]:
    """Structured helper used by the retrieval stack."""
    result = get_cve_detail.invoke({"cve_id": cve_id})
    if not isinstance(result, str):
        raise RuntimeError(f"CVE detail lookup for {cve_id} returned non-string result")
    if result.startswith("[ERROR]"):
        raise RuntimeError(f"CVE detail lookup for {cve_id} failed: {result}")
    try:
        parsed = json.loads(result)
    except Exception as exc:
        raise RuntimeError(f"CVE detail lookup for {cve_id} returned invalid JSON payload: {exc}") from exc
    if isinstance(parsed, list):
        return parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"CVE detail lookup for {cve_id} returned invalid JSON payload")
    return parsed
