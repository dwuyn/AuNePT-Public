"""
src/config.py
─────────────
Single source of truth for all configuration.
Replaces the scattered per-file `yaml.safe_load(config_path)` pattern.

Usage:
    from src.config import get_config
    cfg = get_config()
    llm = cfg.get_llm("ollama")
    target = cfg.recon["target_ip"]
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Dict

import yaml

# ── Path helpers ──────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_CANDIDATES = (
    os.path.join(_ROOT, "configs", "config.yaml"),
    os.path.join(_ROOT, "configs", "config.yaml.example"),
)
_CONFIG_YAML = next((path for path in _CONFIG_CANDIDATES if os.path.isfile(path)), _CONFIG_CANDIDATES[0])


# ── Env-var resolver ──────────────────────────────────────────────────────────

def _resolve(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var, "")
        if not resolved:
            import logging
            logging.getLogger(__name__).warning("Env var %s is not set", env_var)
        return resolved
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


# ── AppConfig ─────────────────────────────────────────────────────────────────

class AppConfig:
    """
    Loads configs/config.yaml once, resolves all env-var placeholders,
    and exposes typed accessors for each section.
    """

    def __init__(self, config_path: str = _CONFIG_YAML) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw: Dict = yaml.safe_load(f)
        except FileNotFoundError:
            sys.exit(f"[FATAL] config.yaml not found at {config_path}")
        self._data: Dict = _resolve(raw)

    # ── Model helpers ─────────────────────────────────────────────────────────

    def model_cfg(self, name: str) -> Dict[str, Any]:
        cfg = self._data.get("models", {}).get(name)
        if cfg is None:
            raise KeyError(f"Model '{name}' not found in configs/config.yaml → models")
        return cfg

    def model_profile_cfg(self, profile_name: str) -> Dict[str, Any]:
        """Resolve a preregistered Vertex profile without config-name aliases."""
        profiles = self._data.get("vertex_profiles", {})
        cfg = profiles.get(profile_name)
        if not isinstance(cfg, dict):
            raise KeyError(f"Vertex model profile {profile_name!r} is not configured")
        required = ("provider", "resource_id", "resource_revision", "region", "pricing")
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise ValueError(f"Vertex model profile {profile_name!r} missing: {', '.join(missing)}")
        if cfg["provider"] != "vertexai":
            raise ValueError(f"Vertex model profile {profile_name!r} must use provider='vertexai'")
        return dict(cfg)

    def get_llm(self, name: str):
        """Instantiate a LangChain-compatible LLM from the named model config."""
        if os.environ.get("VERIPLANPT_GATEWAY_LIVE", "").lower() == "true":
            sys.path.insert(0, _ROOT)
            from utils.llm_factory import GatewayLLM  # noqa: PLC0415
            return GatewayLLM({"model": os.environ.get("VERIPLANPT_MODEL_LABEL", name)})
        # Reuse existing factory — keeps backward compat with llm_factory.py
        sys.path.insert(0, _ROOT)
        from utils.llm_factory import create_llm_from_config  # noqa: PLC0415
        cfg = dict(self.model_cfg(name))
        cfg["name"] = name
        return create_llm_from_config(cfg)

    # ── Runtime section accessors ─────────────────────────────────────────────

    @property
    def recon(self) -> Dict[str, Any]:
        return self._data["runtime"]["recon"]

    @property
    def planning(self) -> Dict[str, Any]:
        return self._data["runtime"]["planning"]

    @property
    def hypothesis(self) -> Dict[str, Any]:
        runtime = self._data["runtime"]
        return runtime.get("hypothesis", {})

    @property
    def execution(self) -> Dict[str, Any]:
        return self._data["runtime"]["execution"]

    @property
    def verifier(self) -> Dict[str, Any]:
        # Falls back to recon model if verifier section not in config
        return self._data["runtime"].get("verifier", {"model": self.recon["model"]})

    def role_model(self, role: str, override: str = "") -> str:
        """Return the selected model for one active-pipeline role."""
        if override:
            return override
        roles = self._data.get("runtime", {}).get("agent_roles", {})
        configured = roles.get(role, {}) if isinstance(roles, dict) else {}
        if isinstance(configured, str):
            return configured
        if isinstance(configured, dict) and configured.get("model"):
            return str(configured["model"])
        fallback = {
            "planner": self.planning.get("model", self.recon["model"]),
            "restore_planner": self.planning.get("model", self.recon["model"]),
            "critic": self.verifier.get("model", self.recon["model"]),
            "verifier": self.verifier.get("model", self.recon["model"]),
            "executor": self.execution.get("model", self.recon["model"]),
        }
        return str(fallback.get(role, self.recon["model"]))

    def get_role_llm(self, role: str, override: str = ""):
        """Instantiate the configured LLM for an active-pipeline role."""
        if os.environ.get("VERIPLANPT_GATEWAY_LIVE", "").lower() == "true":
            # Production containers have no provider credentials. Bind the
            # real graph role to the adapter's relay-backed client surface.
            sys.path.insert(0, _ROOT)
            from utils.llm_factory import GatewayLLM  # noqa: PLC0415

            return GatewayLLM({"model": override or self.role_model(role)})
        if override:
            profiles = self._data.get("vertex_profiles", {})
            if override in profiles:
                sys.path.insert(0, _ROOT)
                from utils.llm_factory import create_llm_from_config  # noqa: PLC0415
                profile = self.model_profile_cfg(override)
                return create_llm_from_config({
                    "name": override, "provider": profile["provider"],
                    "model": profile["resource_id"], "location": profile["region"],
                    "project": profile.get("project", ""), "temperature": 0,
                })
        return self.get_llm(self.role_model(role, override))

    @property
    def cve_scoring(self) -> Dict[str, Any]:
        return self._data["cve_scoring"]

    # ── Convenience ───────────────────────────────────────────────────────────

    def raw(self) -> Dict[str, Any]:
        return self._data


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the singleton AppConfig. Call dotenv.load_dotenv() before first use."""
    return AppConfig()
