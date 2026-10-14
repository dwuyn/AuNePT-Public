"""
src/rag/doc_handler.py
──────────────────────
Unified DocHandler — merges doc_handler.py (full mode) and
doc_handler_ec.py (economic mode) into one class.

economic_mode=False → full multi-feature LLM analysis (doc_handler.py)
economic_mode=True  → single direct-score query per repo (doc_handler_ec.py)
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from functools import wraps
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ── Timeout decorator (shared) ────────────────────────────────────────────────

def _timeout(seconds: int, default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            q: multiprocessing.Queue = multiprocessing.Queue()

            def _worker(q, *a, **kw):
                try:
                    q.put(func(*a, **kw))
                except Exception as exc:
                    q.put({"error": str(exc)})

            p = multiprocessing.Process(target=_worker, args=(q, *args), kwargs=kwargs)
            p.start()
            p.join(timeout=seconds)
            if p.is_alive():
                logger.warning("%s timed out after %ds — using default", func.__name__, seconds)
                p.terminate()
                p.join()
                return default
            return q.get() if not q.empty() else default
        return wrapper
    return decorator


# ── Default fallbacks ─────────────────────────────────────────────────────────

_DEFAULT_VUL_TYPE_CONF = {k: "4" for k in (
    "code_code_execution", "code_privilege_escalation",
    "code_info_leak", "code_bypass", "code_dos",
)}
_DEFAULT_EXP_MATURITY_CONF = {k: "4" for k in ("code_poc", "code_flexibility", "code_functionality")}
_DEFAULT_ISREMOTE_CONF = "4"
_DEFAULT_ATTACK_COMPLEXITY = {k: "4" for k in (
    "code_attack_evasion", "code_info_dependency", "code_attack_condition",
    "code_attack_probability", "code_privilege_required", "code_user_interaction",
)}
_DEFAULT_FEATURES = {k: "False" for k in (
    "code_attack_evasion", "code_info_dependency", "code_attack_condition",
    "code_attack_probability", "code_privilege_required", "code_user_interaction",
)}
_DEFAULT_OTPT = {"note": "timeout"}


# ── UnifiedDocHandler ─────────────────────────────────────────────────────────

class UnifiedDocHandler:
    """
    Handles loading, indexing, and LLM-based analysis of exploit repositories.

    Parameters
    ----------
    economic_mode : bool
        If True, use a single direct-score query per repo (cheaper).
        If False, run the full multi-feature analysis (more accurate).
    """

    def __init__(self, economic_mode: bool = True) -> None:
        self.economic_mode = economic_mode
        self._query_engine = None
        self._summary_dict: Dict[str, str] = {}

    # ── Main entry point ──────────────────────────────────────────────────────

    def vul_analysis(
        self,
        cve: str,
        output_dir: str,
        vul_description: str = "",
    ) -> Dict[str, Any]:
        """
        Analyse all exploit sources for *cve* and return a feature/score dict.
        The caller writes the result to disk.
        """
        from llama_index.core import SimpleDirectoryReader, SummaryIndex  # noqa

        is_general = "exploit" in cve
        doc_dir = (
            f"{output_dir}/Google" if is_general else f"{output_dir}/{cve}/Google"
        )
        code_dirs = {
            "ExploitDB": f"{output_dir}/ExploitDB" if is_general else f"{output_dir}/{cve}/ExploitDB",
            "GitHub": f"{output_dir}/GitHub" if is_general else f"{output_dir}/{cve}/GitHub",
        }

        result: Dict = {"code": {}, "doc": {}}

        # ── Code repos ────────────────────────────────────────────────────────
        for source, code_dir in code_dirs.items():
            if not (os.path.exists(code_dir) and os.listdir(code_dir)):
                continue

            result["code"].setdefault(source, {"lang_class": {}})
            if self.economic_mode:
                result["code"][source]["score"] = {}
            else:
                result["code"][source].update({
                    "vul_type": {}, "exp_maturity": {}, "exp_flexibility": {},
                    "isRemote": {}, "attack_complexity": {},
                })

            subdirs = [f.name for f in os.scandir(code_dir) if f.is_dir()]
            for repo in subdirs:
                repo_dir = os.path.join(code_dir, repo)
                entries = [e for e in os.listdir(repo_dir) if not e.startswith(".")]
                if not entries:
                    continue

                try:
                    reader = SimpleDirectoryReader(repo_dir, recursive=True, num_files_limit=10)
                    docs = reader.load_data()
                    qe = SummaryIndex.from_documents(docs).as_query_engine()
                except Exception as exc:
                    logger.error("Failed to index %s: %s", repo_dir, exc)
                    continue

                from utils.dir_class import judge_class  # noqa
                result["code"][source]["lang_class"][repo] = judge_class(repo_dir)

                if self.economic_mode:
                    score = self._direct_score_from_code(qe)
                    result["code"][source]["score"][repo] = score
                else:
                    result["code"][source].update(
                        self._full_analysis_code(cve, qe, repo, output_dir, vul_description)
                    )

        # ── Google docs ───────────────────────────────────────────────────────
        if os.path.exists(doc_dir) and os.listdir(doc_dir):
            try:
                reader = SimpleDirectoryReader(doc_dir, recursive=True, num_files_limit=10)
                docs = reader.load_data()
                qe = SummaryIndex.from_documents(docs).as_query_engine()
                if self.economic_mode:
                    result["doc"]["score"] = self._direct_score_from_doc(qe)
                else:
                    result["doc"].update(self._full_analysis_doc(cve, qe, output_dir))
            except Exception as exc:
                logger.error("Failed to index doc_dir %s: %s", doc_dir, exc)

        return result

    # ── Economic helpers ──────────────────────────────────────────────────────

    @_timeout(300, default=(5, _DEFAULT_OTPT))
    def _direct_score_from_code(self, query_engine) -> int:
        from utils.vote import get_final_scr  # noqa
        from utils.prompt import PentestAgentPrompt  # noqa
        try:
            score, _ = get_final_scr(str(query_engine.query(PentestAgentPrompt.direct_judge_code)))
            return int(score)
        except Exception as exc:
            logger.error("direct_score_from_code error: %s", exc)
            return 5

    def _direct_score_from_doc(self, query_engine) -> int:
        from utils.vote import get_final_scr  # noqa
        from utils.prompt import PentestAgentPrompt  # noqa
        try:
            score, _ = get_final_scr(str(query_engine.query(PentestAgentPrompt.direct_judge_doc)))
            return int(score)
        except Exception as exc:
            logger.error("direct_score_from_doc error: %s", exc)
            return 5

    # ── Full-mode helpers ─────────────────────────────────────────────────────

    def _full_analysis_code(
        self, cve: str, qe, repo: str, output_dir: str, vul_description: str
    ) -> Dict:
        """Run full multi-feature analysis for a single code repo."""
        from utils.doc_handler import DocHandler  # noqa  (delegates to existing full handler)
        dh = DocHandler()
        # Delegate to existing full analysis methods to avoid reimplementing them
        vul_type, _, _ = dh.get_vul_category_from_code(cve, qe, repo, output_dir)
        exp_maturity, exp_flex, _, _ = dh.get_exp_maturity_analysis(cve, qe, vul_description, repo, output_dir)
        is_remote, _, _ = dh.get_isRemote_from_code(cve, qe, repo, output_dir)
        attack, _, _ = dh.get_attack_complexity_from_code(cve, qe, repo, output_dir)
        return {
            "vul_type": {repo: vul_type},
            "exp_maturity": {repo: exp_maturity},
            "exp_flexibility": {repo: exp_flex},
            "isRemote": {repo: is_remote},
            "attack_complexity": {repo: attack},
        }

    def _full_analysis_doc(self, cve: str, qe, output_dir: str) -> Dict:
        from utils.doc_handler import DocHandler  # noqa
        dh = DocHandler()
        return {
            "vul_type": dh.get_vul_category_from_doc(cve, qe, output_dir),
            "isRemote": dh.get_isRemote_from_doc(cve, qe, output_dir),
            "attack_complexity": dh.get_attack_complexity_from_doc(cve, qe, output_dir),
        }

    # ── Index / keyword query (planning phase) ────────────────────────────────

    def create_index(self, topic_dir: str, summary_prompt: str, keyword: str) -> None:
        """Build or load a keyword index across all repos under topic_dir."""
        from llama_index.core import (  # noqa
            SimpleDirectoryReader, SimpleKeywordTableIndex, SummaryIndex,
            StorageContext, load_index_from_storage,
        )
        from llama_index.core.schema import IndexNode
        from llama_index.core.query_engine import RetrieverQueryEngine

        idx_dir = os.path.join(os.environ.get("INDEX_STORAGE_DIR", "."), "keyword_repos", keyword)

        if os.path.exists(idx_dir):
            ctx = StorageContext.from_defaults(persist_dir=idx_dir)
            ki = load_index_from_storage(ctx)
        else:
            nodes = []
            for i, repo_dir in enumerate(
                f.path for f in os.scandir(topic_dir) if f.is_dir()
            ):
                visible = [x for x in os.listdir(repo_dir) if not x.startswith(".")]
                if not visible:
                    continue
                reader = SimpleDirectoryReader(repo_dir, recursive=True)
                docs = reader.load_data()
                qe = SummaryIndex.from_documents(docs).as_query_engine()
                summary = str(qe.query(summary_prompt))
                self._summary_dict[repo_dir] = summary
                nodes.append(IndexNode(
                    text=summary,
                    metadata={"repo path": repo_dir},
                    index_id=str(i),
                ))
            ki = SimpleKeywordTableIndex(objects=nodes)
            ki.storage_context.persist(persist_dir=idx_dir)

        self._query_engine = RetrieverQueryEngine.from_args(
            ki.as_retriever(verbose=True), verbose=True
        )

    def query(self, content: str):
        if self._query_engine is None:
            raise RuntimeError("Call create_index() before query()")
        return self._query_engine.query(content)
