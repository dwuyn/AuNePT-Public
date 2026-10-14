"""
Placeholder resolution and command rendering for exploit plans.
"""

from __future__ import annotations

import re
import socket
from typing import Any, Mapping

from src.memory.world_state import Credential, WorldState

_TOKEN_RE = re.compile(r"(\{\{[^}]+\}\}|<[A-Za-z0-9_:-]+>|RHOSTS?|LHOST|RPORT|LPORT|TARGET(?:_IP|_PORT)?|URL|USERNAME|PASSWORD|CVE_ID)")
_HTTPS_PORTS = {"443", "8443"}
_HTTP_PORTS = {"80", "8080", "8000", "8081", "8888"} | _HTTPS_PORTS


def _normalize_placeholder(token: str) -> str:
    cleaned = token.strip().strip("{}<>").strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1]
    return cleaned.upper().replace("-", "_")


def extract_placeholder_names(*values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, str):
            names.extend(_normalize_placeholder(match) for match in _TOKEN_RE.findall(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    names.extend(_normalize_placeholder(match) for match in _TOKEN_RE.findall(item))
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _service_match(credential: Credential, exploit: dict[str, Any]) -> bool:
    service = str(exploit.get("service", "")).lower()
    target_service = str(credential.target_service or "").lower()
    return bool(service and target_service and service in target_service)


def _pick_credential(ws: WorldState, exploit: dict[str, Any]) -> Credential | None:
    verified = [item for item in ws.credentials if item.verified and _service_match(item, exploit)]
    if verified:
        return verified[0]
    verified_any = [item for item in ws.credentials if item.verified]
    if verified_any:
        return verified_any[0]
    unverified = [item for item in ws.credentials if _service_match(item, exploit)]
    if unverified:
        return unverified[0]
    return ws.credentials[0] if ws.credentials else None


def _guess_url(exploit: dict[str, Any], target_ip: str, target_port: str) -> str:
    if not target_ip:
        return ""
    port = target_port or str(exploit.get("target_port", "") or "")
    if not port:
        return f"http://{target_ip}"
    if port in _HTTP_PORTS:
        scheme = "https" if port in _HTTPS_PORTS else "http"
        suffix = "" if port in {"80", "443"} else f":{port}"
        return f"{scheme}://{target_ip}{suffix}"
    return f"http://{target_ip}:{port}"


def _pick_lport(preferred: int = 4444) -> str:
    for candidate in range(preferred, preferred + 56):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", candidate))
                return str(candidate)
        except OSError:
            continue
        except PermissionError:
            return str(preferred)
    return str(preferred)


def resolve_placeholder_values(
    exploit: dict[str, Any],
    state: Mapping[str, Any],
    ws: WorldState,
) -> tuple[dict[str, str], list[str]]:
    target_ip = str(exploit.get("target_ip") or state.get("target_ip") or "").strip()
    target_port = str(exploit.get("target_port") or state.get("target_port") or "").strip()
    attacker_ip = str(state.get("attacker_ip") or "").strip()
    lport = str(exploit.get("lport") or state.get("lport") or "").strip()
    credential = _pick_credential(ws, exploit)

    values: dict[str, str] = {}
    if target_ip:
        for key in ("TARGET_IP", "TARGET", "RHOST", "RHOSTS"):
            values[key] = target_ip
    if target_port:
        for key in ("TARGET_PORT", "RPORT"):
            values[key] = target_port
    if attacker_ip:
        values["LHOST"] = attacker_ip
    if attacker_ip:
        values["LPORT"] = lport or _pick_lport()
    if credential:
        if credential.username:
            values["USERNAME"] = credential.username
        if credential.password:
            values["PASSWORD"] = credential.password
    if exploit.get("cve_id"):
        values["CVE_ID"] = str(exploit.get("cve_id"))
    url = _guess_url(exploit, target_ip, target_port)
    if url:
        values["URL"] = url

    placeholder_names = extract_placeholder_names(
        exploit.get("commands", []),
        exploit.get("verify_commands", []),
        exploit.get("placeholders", []),
        exploit.get("required_placeholders", []),
    )
    missing = [name for name in placeholder_names if name not in values]
    return values, missing


def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
        rendered = rendered.replace(f"<{name}>", value)
        rendered = rendered.replace(f"<{name.lower()}>", value)
        rendered = rendered.replace(f"<{name.lower().replace('_', '-')}>", value)
    for name, value in values.items():
        rendered = re.sub(rf"\b{name}\b", lambda _: value, rendered)
    return rendered


def render_commands(commands: list[str], values: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            continue
        rendered.append(render_template(command, values).strip())
    return rendered
