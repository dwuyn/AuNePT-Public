"""
Structured PoC candidate retrieval and selective materialization.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.retrieval.models import AuthoritativeRecord, PocCandidate
from utils.searchers.exploitdb_searcher import ExploitDBSearcher
from utils.searchers.github_searcher import GithubSearcher
from utils.searchers.google_searcher import GoogleSearcher


def _candidate_id(prefix: str, cve_id: str, name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in name.lower())
    return f"{prefix}:{cve_id}:{safe[:80]}"


def _github_candidates(
    cve_id: str,
    output_dir: str,
    retrieval_cfg: dict[str, Any],
) -> list[PocCandidate]:
    searcher = GithubSearcher()
    repo_limit = int(retrieval_cfg.get("github_search_limit", 12))
    clone_top_k = int(retrieval_cfg.get("github_clone_top_k", 2))
    results = searcher.search_keyword_metadata(cve_id, limit=repo_limit)
    candidates: list[PocCandidate] = []
    for item in results[:clone_top_k]:
        local_path = searcher.clone_candidate(item, output_dir)
        repo_name = str(item.get("name", "repo"))
        candidate = PocCandidate(
            candidate_id=_candidate_id("github", cve_id, repo_name),
            cve_id=cve_id,
            source="github",
            path=local_path,
            locator=str(item.get("clone_url") or item.get("html_url") or ""),
            repo_name=repo_name,
            entry_files=[],
            language=str(item.get("language") or ""),
            stars=int(item.get("stars_count") or item.get("stargazers_count") or 0),
            forks=int(item.get("forks_count") or 0),
            created_at=str(item.get("created_at") or ""),
            has_readme=os.path.exists(os.path.join(local_path, "README.md")) if local_path else False,
            has_usage=False,
            raw_confidence=float(item.get("efct_score") or 0.0),
            evidence=[
                f"GitHub repo candidate {repo_name}",
                f"stars={int(item.get('stars_count') or item.get('stargazers_count') or 0)}",
            ],
        )
        candidates.append(candidate)
    return candidates


def _exploitdb_candidates(
    cve_id: str,
    output_dir: str,
    retrieval_cfg: dict[str, Any],
) -> list[PocCandidate]:
    searcher = ExploitDBSearcher()
    copy_top_k = int(retrieval_cfg.get("exploitdb_copy_top_k", 2))
    results = searcher.search_keyword_metadata(cve_id)[:copy_top_k]
    candidates: list[PocCandidate] = []
    for item in results:
        local_path = searcher.copy_candidate(item, output_dir)
        file_name = os.path.basename(str(item.get("relative_path") or item.get("name") or "exploit"))
        candidates.append(PocCandidate(
            candidate_id=_candidate_id("exploitdb", cve_id, file_name),
            cve_id=cve_id,
            source="exploitdb",
            path=local_path,
            locator=str(item.get("relative_path") or ""),
            repo_name=file_name,
            entry_files=[file_name] if local_path else [],
            language=os.path.splitext(file_name)[1].lstrip("."),
            stars=0,
            forks=0,
            created_at="",
            has_readme=os.path.exists(os.path.join(os.path.dirname(local_path), "README.md")) if local_path else False,
            has_usage=False,
            raw_confidence=float(item.get("rank") or 0.0),
            evidence=[
                f"ExploitDB candidate {file_name}",
                f"path={item.get('relative_path')}",
            ],
        ))
    return candidates


def _google_candidates(
    cve_id: str,
    output_dir: str,
    retrieval_cfg: dict[str, Any],
) -> list[PocCandidate]:
    if not retrieval_cfg.get("enable_google_fallback", True):
        return []
    top_k = int(retrieval_cfg.get("google_fallback_top_k", 1))
    searcher = GoogleSearcher()
    links = searcher.search_keyword_metadata(cve_id, limit=top_k)
    crawled = searcher.materialize_links(links[:top_k], output_dir)
    candidates: list[PocCandidate] = []
    for item in crawled:
        path = str(item.get("path") or "")
        link = str(item.get("link") or "")
        name = os.path.basename(os.path.dirname(path)) if path else link
        candidates.append(PocCandidate(
            candidate_id=_candidate_id("google", cve_id, name or "doc"),
            cve_id=cve_id,
            source="google",
            path=path,
            locator=link,
            repo_name=name,
            entry_files=[os.path.basename(path)] if path else [],
            language="md",
            stars=0,
            forks=0,
            created_at="",
            has_readme=False,
            has_usage=True,
            raw_confidence=0.1,
            evidence=[f"Google fallback doc from {link}"],
        ))
    return candidates


def collect_poc_candidates(
    records: list[AuthoritativeRecord],
    output_dir: str,
    retrieval_cfg: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> list[PocCandidate]:
    cfg = retrieval_cfg or {}
    os.makedirs(output_dir, exist_ok=True)
    all_candidates: list[PocCandidate] = []
    for record in records[: int(cfg.get("top_cves", 5))]:
        cve_dir = os.path.join(output_dir, record.cve_id)
        github_dir = os.path.join(cve_dir, "GitHub")
        exploitdb_dir = os.path.join(cve_dir, "ExploitDB")
        google_dir = os.path.join(cve_dir, "Google")
        os.makedirs(cve_dir, exist_ok=True)

        candidates = []
        try:
            candidates = _exploitdb_candidates(record.cve_id, exploitdb_dir, cfg)
        except Exception as exc:
            if errors is not None:
                errors.append(f"ExploitDB search for '{record.cve_id}' failed: {exc}")

        try:
            github_res = _github_candidates(record.cve_id, github_dir, cfg)
            candidates.extend(github_res)
        except Exception as exc:
            if errors is not None:
                errors.append(f"GitHub search for '{record.cve_id}' failed: {exc}")

        if not candidates:
            try:
                google_res = _google_candidates(record.cve_id, google_dir, cfg)
                candidates.extend(google_res)
            except Exception as exc:
                if errors is not None:
                    errors.append(f"Google search fallback for '{record.cve_id}' failed: {exc}")

        all_candidates.extend(candidates)
    return all_candidates


def serialize_candidates(candidates: list[PocCandidate]) -> str:
    return json.dumps([candidate.to_dict() for candidate in candidates], indent=2)
