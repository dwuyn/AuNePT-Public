"""Dependency-free HTTP relay from an internal network to a Unix gateway socket."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

SOCKET_PATH = "/run/veriplanpt-gateway/gateway.sock"
RELAY_ENDPOINT = "http://gateway-relay:8080/v1/generate"
# The relay must outlive the pinned 120-second host SDK deadline so a late
# structured terminal response can reach the framework container.
RELAY_FORWARD_TIMEOUT_SECONDS = 130.0


class RelayError(ValueError):
    """A relay request or relay configuration is invalid."""


class UnixHTTPConnection(http.client.HTTPConnection):
    """Small ``http.client`` transport whose peer is a Unix socket."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost", timeout=RELAY_FORWARD_TIMEOUT_SECONDS)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def forward_json(
    *, socket_path: str, body: bytes, headers: Mapping[str, str],
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Forward exactly one HTTP request to the host gateway Unix socket."""
    if not Path(socket_path).is_socket():
        raise RelayError("gateway Unix socket is unavailable")
    connection = UnixHTTPConnection(socket_path)
    try:
        forward_headers = {
            "Content-Type": headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        authorization = headers.get("Authorization", "")
        if authorization:
            forward_headers["Authorization"] = authorization
        connection.request("POST", "/v1/generate", body=body, headers=forward_headers)
        response = connection.getresponse()
        payload = response.read()
        return response.status, [(key, value) for key, value in response.getheaders()], payload
    finally:
        connection.close()


def relay_handler(socket_path: str) -> type[BaseHTTPRequestHandler]:
    """Create a handler that exposes only the pinned POST endpoint."""
    if not socket_path or not socket_path.startswith("/"):
        raise RelayError("relay socket path must be absolute")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/generate":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    raise RelayError("relay request size is invalid")
                body = self.rfile.read(length)
                json.loads(body.decode("utf-8"))
                status, headers, response_body = forward_json(
                    socket_path=socket_path, body=body, headers=dict(self.headers.items()),
                )
                self.send_response(status)
                for key, value in headers:
                    if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except (RelayError, OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(502)

        def do_GET(self) -> None:  # noqa: N802
            self.send_error(404)

        def do_PUT(self) -> None:  # noqa: N802
            self.send_error(404)

        def do_DELETE(self) -> None:  # noqa: N802
            self.send_error(404)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def serve_relay(*, host: str = "0.0.0.0", port: int = 8080, socket_path: str = SOCKET_PATH) -> ThreadingHTTPServer:
    """Construct a relay server; callers own its lifecycle."""
    if host not in {"0.0.0.0", "127.0.0.1"} or not 0 <= port <= 65535:
        raise RelayError("relay bind address is invalid")
    if os.geteuid() == 0:
        raise RelayError("gateway relay refuses root")
    return ThreadingHTTPServer((host, port), relay_handler(socket_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--socket", default=SOCKET_PATH)
    args = parser.parse_args(argv)
    server = serve_relay(host=args.host, port=args.port, socket_path=args.socket)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
