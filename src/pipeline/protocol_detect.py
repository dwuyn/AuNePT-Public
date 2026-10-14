"""Deterministic protocol-family detection from a single service banner.

Reconnaissance classifies the contract port's protocol family before any
endpoint probing (TLS / SSH / HTTP / WebSocket / raw TCP). The classifier is
pure and offline: it inspects only the bytes returned by one connect/handshake
and the port number, and never performs network I/O or model inference.
"""
from __future__ import annotations

from typing import Any

#: The five protocol families the classifier can report.
PROTOCOL_FAMILIES: tuple[str, ...] = ("tls", "ssh", "http", "websocket", "raw-tcp")

_TLS_RECORD_PREFIX = b"\x16\x03"
_SSH_PREFIX = b"SSH-"
_HTTP_METHODS = (
    b"get ", b"post ", b"head ", b"put ", b"delete ", b"options ", b"patch ", b"trace ",
)


def detect_protocol(banner: bytes, port: int) -> tuple[str, dict[str, Any]]:
    """Classify ``banner`` into one of :data:`PROTOCOL_FAMILIES`.

    Returns the family name and a detail mapping carrying the matched
    ``signature`` and a human-readable ``reason``. Pure, deterministic and
    network-free: the same ``(banner, port)`` always yields the same result.
    """
    data = bytes(banner or b"")
    if len(data) >= 2 and data[0:2] == _TLS_RECORD_PREFIX:
        return "tls", {
            "signature": "tls-record-header",
            "reason": "banner opens with a TLS record header (0x16 0x03)",
            "port": int(port),
        }
    if data.startswith(_SSH_PREFIX):
        return "ssh", {
            "signature": "ssh-identification",
            "reason": "banner starts with the SSH identification prefix 'SSH-'",
            "port": int(port),
        }
    head = data[:8192].lower()
    first_line = head.split(b"\r\n", 1)[0].split(b"\n", 1)[0]
    if b"upgrade:" in head and b"websocket" in head:
        return "websocket", {
            "signature": "http-upgrade-websocket",
            "reason": "HTTP request carries an 'Upgrade: websocket' header",
            "port": int(port),
        }
    if first_line.startswith(b"http/") or _is_http_request_line(first_line):
        return "http", {
            "signature": "http-status-or-request-line",
            "reason": "banner begins with an HTTP status line or request line",
            "port": int(port),
        }
    return "raw-tcp", {
        "signature": "unmatched",
        "reason": "no known protocol signature matched; treating the port as raw TCP",
        "port": int(port),
    }


def _is_http_request_line(first_line: bytes) -> bool:
    if b"http/" not in first_line:
        return False
    return first_line.startswith(_HTTP_METHODS)
