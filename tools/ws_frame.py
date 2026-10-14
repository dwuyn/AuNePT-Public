#!/usr/bin/env python3
"""``ws-frame`` — the WebSocket transport adapter (paper §3.3).

A procedure adapter implements transport and serialization; it carries no
success logic. This one opens a WebSocket handshake (optionally offering a
subprotocol), sends exactly one text frame, prints every frame the peer sends
back until the wait window closes, and exits non-zero when the handshake is
refused.

It exists because the registered ``websocket-frame`` template needs a transport
that can actually send a frame: ``curl`` can open a ``ws://`` URL but its CLI has
no option that sends an application frame (``curl_ws_send()`` is library-only), so
a WebSocket CVE — e.g. Vite's ``vite:invoke`` file read — cannot be dispatched
with it. The adapter is stdlib-only and must be on ``PATH`` where procedures are
dispatched (``ws-frame``).

Usage:
  ws-frame <ws-url> [--subprotocol NAME] [--frame TEXT] [--wait SECONDS]
           [--origin URL] [--header 'Name: value'] [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import base64
import os
import socket
import sys
import time
from urllib.parse import urlsplit


def handshake(sock: socket.socket, url: str, *, subprotocol: str, origin: str,
              headers: list[str], timeout: float) -> None:
    parts = urlsplit(url)
    if parts.scheme != "ws":
        raise SystemExit(f"ws-frame: only ws:// is supported, got {parts.scheme!r}")
    host = parts.hostname or ""
    port = parts.port or 80
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    key = base64.b64encode(os.urandom(16)).decode()
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if subprotocol:
        lines.append(f"Sec-WebSocket-Protocol: {subprotocol}")
    if origin:
        lines.append(f"Origin: {origin}")
    lines.extend(headers)
    sock.settimeout(timeout)
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
    head = response.decode("latin-1", "replace")
    status_line = head.split("\r\n", 1)[0]
    if " 101 " not in status_line:
        sys.stderr.write(f"ws-frame: handshake refused: {status_line}\n")
        raise SystemExit(2)


def send_text_frame(sock: socket.socket, payload: bytes) -> None:
    header = bytearray([0x81])
    mask = os.urandom(4)
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 1 << 16:
        header.append(0x80 | 126)
        header += length.to_bytes(2, "big")
    else:
        header.append(0x80 | 127)
        header += length.to_bytes(8, "big")
    header += mask
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + masked)


def read_frames(sock: socket.socket, *, wait: float) -> None:
    deadline = time.monotonic() + wait
    buffer = b""
    while time.monotonic() < deadline:
        sock.settimeout(max(deadline - time.monotonic(), 0.1))
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk
        while len(buffer) >= 2:
            first, second = buffer[0], buffer[1]
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            offset = 2
            if length == 126:
                if len(buffer) < 4:
                    break
                length = int.from_bytes(buffer[2:4], "big")
                offset = 4
            elif length == 127:
                if len(buffer) < 10:
                    break
                length = int.from_bytes(buffer[2:10], "big")
                offset = 10
            mask_key = b""
            if masked:
                if len(buffer) < offset + 4:
                    break
                mask_key = buffer[offset:offset + 4]
                offset += 4
            if len(buffer) < offset + length:
                break
            payload = buffer[offset:offset + length]
            buffer = buffer[offset + length:]
            if masked and mask_key:
                payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                return
            if opcode in (0x1, 0x2, 0x0):
                sys.stdout.write(payload.decode("utf-8", "replace"))
                sys.stdout.flush()
                if not payload.endswith(b"\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ws-frame", description="send one WebSocket frame")
    parser.add_argument("url")
    parser.add_argument("--subprotocol", default="")
    parser.add_argument("--frame", default="")
    parser.add_argument("--wait", type=float, default=3.0)
    parser.add_argument("--origin", default="")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    parts = urlsplit(args.url)
    sock = socket.create_connection((parts.hostname or "", parts.port or 80), timeout=args.timeout)
    try:
        handshake(sock, args.url, subprotocol=args.subprotocol, origin=args.origin,
                  headers=[str(item) for item in args.header], timeout=args.timeout)
        if args.frame:
            send_text_frame(sock, args.frame.encode())
        read_frames(sock, wait=args.wait)
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
