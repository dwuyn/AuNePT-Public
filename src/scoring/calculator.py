"""
src/scoring/calculator.py
─────────────────────────
Unified CVE scoring logic.

Fixes P3 (duplication): merges cve_info.py (1 212 lines) and
cve_info_ec.py (1 150 lines) into one module with an `economic_mode` flag.

Key differences between full and economic mode:
  - Full mode:   DocHandler runs multi-query LLM analysis per feature
                 (vul_type, exp_maturity, isRemote, attack_complexity)
                 → more accurate, more tokens
  - Economic mode: DocHandler runs a single direct-score query per repo
                   → faster, cheaper
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def build_exploit_plan_from_bundle(
    retrieval_bundle: dict,
    economic_mode: bool = True,
) -> list[dict[str, Any]]:
    """
    Project a retrieval bundle into the execution-facing exploit_plan schema.

    This is the new, retrieval-driven path. It does not rerun broad searches.
    """
    shortlist = list(retrieval_bundle.get("shortlist", []))
    poc_map = {item.get("candidate_id"): item for item in retrieval_bundle.get("poc_candidates", [])}
    assessment_map = {item.get("candidate_id"): item for item in retrieval_bundle.get("assessments", [])}
    plan: list[dict[str, Any]] = []
    for item in shortlist:
        candidate = poc_map.get(item.get("candidate_id"), {})
        assessment = assessment_map.get(item.get("candidate_id"), {})
        base_score = float(item.get("score", assessment.get("score", 0.0)) or 0.0)
        plan.append({
            "name": f"{item.get('cve_id', 'unknown')}::{candidate.get('repo_name', candidate.get('candidate_id', 'candidate'))}",
            "file_path": candidate.get("path", ""),
            "locator": candidate.get("locator", ""),
            "working_directory": item.get("working_directory", "") or (candidate.get("path", "") if os.path.isdir(candidate.get("path", "")) else os.path.dirname(candidate.get("path", ""))),
            "candidate_id": item.get("candidate_id", ""),
            "cve_id": item.get("cve_id", ""),
            "source": candidate.get("source", item.get("source", "")),
            "score": round(base_score * 100, 2),
            "priority_score": base_score,
            "trust_score": float(item.get("trust_score", assessment.get("trust_score", 0.0)) or 0.0),
            "estimated_cost": float(item.get("estimated_cost", assessment.get("estimated_cost", 0.0)) or 0.0),
            "execution_readiness": 0.9 if item.get("commands") else 0.6 if economic_mode else 0.75,
            "exploitability": "easy" if item.get("verdict") == "confirmed" else "medium" if item.get("verdict") == "plausible" else "hard",
            "commands": list(item.get("commands", [])),
            "dependencies": list(item.get("dependencies", [])),
            "placeholders": list(item.get("placeholders", [])),
            "required_placeholders": list(item.get("required_placeholders", item.get("placeholders", []))),
            "setup_commands": list(item.get("setup_commands", [])),
            "verify_commands": list(item.get("verify_commands", [])),
            "success_indicators": list(item.get("success_indicators", [])),
            "failure_indicators": list(item.get("failure_indicators", [])),
            "reasons": list(item.get("reasons", [])),
            "service": item.get("service", ""),
            "version": item.get("version", ""),
            "target_ip": item.get("target_ip", ""),
            "target_port": item.get("port", 0),
            "title": item.get("title", ""),
        })
    plan.sort(key=lambda entry: (entry.get("priority_score", 0.0), entry.get("score", 0.0)), reverse=True)
    return plan


# ── CVE metadata fetch ────────────────────────────────────────────────────────

def fetch_cve_metadata(cve_id: str, info_dir: str) -> Optional[Dict]:
    """Call vulnx to get CVE metadata; cache result in info_dir/cvemap.json."""
    os.makedirs(info_dir, exist_ok=True)
    cache = os.path.join(info_dir, "cvemap.json")

    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0] if data else None
        return data

    try:
        result = subprocess.run(
            f"vulnx id {cve_id} -j > {cache}",
            shell=True, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error("vulnx failed for %s: %s", cve_id, result.stderr)
            return None
        with open(cache) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as exc:
        logger.error("fetch_cve_metadata error: %s", exc)
        return None


# ── Score arithmetic ──────────────────────────────────────────────────────────

def _get_final_score(func_score: float, comp_score: float) -> float:
    numerator = 2 * func_score / 30 * (35 - comp_score)
    if func_score < 12:
        denominator = 10
    elif func_score < 22:
        denominator = 5
    else:
        denominator = 1
    multiplier = 0.2 if (func_score < 30 and comp_score < 15) else 1
    return numerator / denominator * multiplier


def _categorize_cvss(score: float) -> str:
    if score < 4.0:
        return "hard"
    if score < 7.0:
        return "medium"
    return "easy"


def _categorize_epss(percentile: float) -> str:
    if percentile < 0.4:
        return "hard"
    if percentile < 0.94:
        return "medium"
    return "easy"


def _classify_exploitability(final_score: float) -> str:
    if final_score > 50:
        return "easy"
    if final_score > 35:
        return "medium"
    return "hard"


# ── Full-mode scoring ─────────────────────────────────────────────────────────

def _full_score(features: Dict, weights: Dict, trending_score: float) -> Tuple:
    """Calculate score using the full multi-feature analysis (cve_info.py logic)."""
    scores: Dict = {}
    best: dict[str, Any] = {"func": 0, "comp": 0, "final": 0, "repo": None}
    best_github = dict(best)
    best_expdb = dict(best)
    has_code = False

    for source in ("GitHub", "ExploitDB"):
        src_data = features.get("code", {}).get(source)
        if not src_data:
            continue
        has_code = True
        source_weight = weights["source_weights"]["expdb" if source == "ExploitDB" else "gthb"]

        for repo, vul_type in src_data.get("vul_type", {}).items():
            func = weights["vul_type"].get(vul_type, 0)
            func += weights["isRemote"].get(src_data.get("isRemote", {}).get(repo, ""), 0)
            if repo == "Code_File":
                func /= 2
            func *= weights["exp_maturity"].get(src_data.get("exp_maturity", {}).get(repo, ""), 1)

            comp = sum(
                weights["attack_complexity"].get(f, {}).get(v, 0)
                for f, v in src_data.get("attack_complexity", {}).get(repo, {}).items()
            )
            comp *= weights["lang_class"].get(src_data.get("lang_class", {}).get(repo, ""), 1)
            comp *= weights.get("exp_flexibility", {}).get(
                src_data.get("exp_flexibility", {}).get(repo, ""), 1
            )

            merged = _get_final_score(func, comp)
            final = merged * source_weight
            scores[repo] = {"functionality": func, "complexity": comp, "final": final}

            bucket = best_expdb if repo.isdigit() else best_github
            if final >= bucket["final"]:
                bucket.update({"func": func, "comp": comp, "final": final, "repo": repo})

    # Trending
    t_expdb = weights["expdb_default_score"] if trending_score == 999 else 0
    t_github = 0 if trending_score == 999 else trending_score
    w_t = weights["trending_score"]
    f_expdb = (best_expdb["final"] + t_expdb * w_t) if best_expdb["final"] > 0 else 0
    f_github = (best_github["final"] + t_github * w_t) if best_github["final"] > 0 else 0
    if has_code:
        if f_expdb > f_github:
            best = {**best_expdb, "final": f_expdb}
        else:
            best = {**best_github, "final": f_github}

    # Document score
    doc = features.get("doc", {})
    if doc:
        d_func = weights["vul_type"].get(doc.get("vul_type", ""), 0)
        d_func += weights["isRemote"].get(doc.get("isRemote", ""), 0)
        d_comp = sum(
            weights["attack_complexity"].get(f, {}).get(v, 0)
            for f, v in doc.get("attack_complexity", {}).items()
        )
        d_final = (d_func + 30 - d_comp) * weights["source_weights"]["gg"]
        scores["doc"] = {"functionality": d_func, "complexity": d_comp, "final": d_final}
        if d_final > best["final"]:
            best = {"func": d_func, "comp": d_comp, "final": d_final, "repo": "doc"}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["final"], reverse=True)
    return sorted_scores, best, has_code


# ── Economic-mode scoring ─────────────────────────────────────────────────────

def _economic_score(features: Dict, weights: Dict, trending_score: float) -> Tuple:
    """Calculate score using a single direct-score query (cve_info_ec.py logic)."""
    scores: Dict = {}
    max_score = 0.0
    max_repo = None
    final_score = 0.0
    has_code = False

    for source in ("GitHub", "ExploitDB"):
        src_data = features.get("code", {}).get(source)
        if not src_data:
            continue
        has_code = True
        source_weight = weights["source_weights"]["expdb" if source == "ExploitDB" else "gthb"]

        for repo, raw_score in src_data.get("score", {}).items():
            score = raw_score / 2 if repo == "Code_File" else float(raw_score)
            score *= weights["lang_class"].get(src_data.get("lang_class", {}).get(repo, ""), 1)
            score *= source_weight
            scores[repo] = score
            if score >= max_score:
                max_score = score
                max_repo = repo

    # Trending
    t_expdb = weights["expdb_default_score"] if trending_score == 999 else 0
    t_github = 0 if trending_score == 999 else trending_score
    w_t = weights["trending_score"]
    if has_code:
        final_score = max_score + (t_expdb if max_repo and max_repo.isdigit() else t_github) * w_t

    # Document
    doc = features.get("doc", {})
    if doc and "score" in doc:
        d_score = float(doc["score"]) * weights["source_weights"]["gg"]
        scores["doc"] = d_score
        if d_score > max_score:
            max_score = d_score
            max_repo = "doc"
            final_score = d_score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores, max_repo, max_score, final_score, has_code


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_and_score(
    cve: str,
    output_dir: str,
    economic_mode: bool,
    cve_description: str = "",
) -> Tuple[float, float]:
    """
    Run the full analysis pipeline for a single CVE.

    1. Fetch exploit sources (already expected in output_dir/{cve}/).
    2. Run DocHandler vulnerability analysis.
    3. Compute final exploitability score.

    Returns (searching_time, analysis_time).
    """
    from src.rag.doc_handler import UnifiedDocHandler  # avoid circular at module level

    cve_dir = os.path.join(output_dir, cve)
    info_dir = os.path.join(cve_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    t0 = time.time()
    doc_handler = UnifiedDocHandler(economic_mode=economic_mode)
    t1 = time.time()

    if economic_mode:
        result = doc_handler.vul_analysis(cve, output_dir)
        out_file = os.path.join(cve_dir, "direct_scores.json")
    else:
        result = doc_handler.vul_analysis(cve, output_dir, cve_description)
        out_file = os.path.join(cve_dir, "features.json")

    t2 = time.time()

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return t1 - t0, t2 - t1


def score_cve(
    cve: str,
    output_dir: str,
    economic_mode: bool,
    weights: Dict,
) -> Dict[str, Any]:
    """
    Load pre-computed features and return a classification dict with
    cvss_category, epss_category, exploitability, and final_score.
    """
    if economic_mode:
        feature_file = os.path.join(output_dir, cve, "direct_scores.json")
    else:
        feature_file = os.path.join(output_dir, cve, "features.json")

    if not os.path.exists(feature_file):
        logger.warning("Feature file not found for %s — returning zeros", cve)
        return {"exploitability": "hard", "final_score": 0, "has_code": False}

    with open(feature_file) as f:
        features = json.load(f)

    # Load trending score
    trending_path = os.path.join(output_dir, cve, "Trend_Score.json")
    trending = 999
    if features.get("code", {}).get("GitHub") and os.path.exists(trending_path):
        try:
            with open(trending_path) as f:
                trending = min(json.load(f).get("trend_score", 0), 50)
        except Exception:
            pass

    # Load CVSS / EPSS from metadata
    cvemap_path = os.path.join(output_dir, cve, "info", "cvemap.json")
    cvss_score, epss_percentile = 0.0, 0.0
    if os.path.exists(cvemap_path):
        with open(cvemap_path) as f:
            meta = json.load(f)
        if isinstance(meta, list):
            meta = meta[0] if meta else {}
        cvss_score = meta.get("cvss_score", 0.0)
        epss_percentile = meta.get("epss_percentile", meta.get("epss", {}).get("epss_percentile", 0.0))

    if economic_mode:
        sorted_scores, max_repo, max_score, final_score, has_code = _economic_score(features, weights, trending)
        classification = {
            "exploitability": _classify_exploitability(final_score),
            "cvss_category": _categorize_cvss(cvss_score),
            "epss_category": _categorize_epss(epss_percentile),
            "final_score": final_score,
            "max_score_repo": max_repo,
            "scores": sorted_scores,
            "has_code": has_code,
        }
        out = os.path.join(output_dir, cve, "classification_ec.json")
    else:
        sorted_scores, best, has_code = _full_score(features, weights, trending)
        final_score = best["final"]
        classification = {
            "exploitability": _classify_exploitability(final_score),
            "cvss_category": _categorize_cvss(cvss_score),
            "epss_category": _categorize_epss(epss_percentile),
            "functionality_score": best["func"],
            "complexity_score": best["comp"],
            "final_score": final_score,
            "max_score_repo": best["repo"],
            "scores": sorted_scores,
            "has_code": has_code,
        }
        out = os.path.join(output_dir, cve, "classification.json")

    with open(out, "w") as f:
        json.dump(classification, f, indent=2)

    return classification


def get_exp_info(
    cve_list: list[str],
    output_dir: str,
    app: str,
    economic_mode: bool = True,
    weights: Dict | None = None,
) -> Tuple[float, float]:
    """
    Full pipeline: for each CVE → analyze → score → return timing.
    Equivalent to the old cve_info.get_exp_info() / cve_info_ec.get_exp_info().
    """
    from src.config import get_config  # noqa
    cfg = get_config()
    _weights = weights or cfg.cve_scoring["weights"]

    total_search = 0.0
    total_analysis = 0.0

    for cve in cve_list:
        meta = fetch_cve_metadata(cve, os.path.join(output_dir, cve, "info"))
        description = (meta or {}).get("description", "")
        try:
            s, a = analyze_and_score(cve, output_dir, economic_mode, description)
            total_search += s
            total_analysis += a
            score_cve(cve, output_dir, economic_mode, _weights)
        except Exception as exc:
            logger.error("Error processing %s: %s", cve, exc)

    return total_search, total_analysis
