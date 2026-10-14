"""Minimal, per-run Metasploit MessagePack RPC client."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import msgpack

from src.pipeline.runtime import rpc_secret


@dataclass
class MetasploitRpcService:
    run_dir: str
    msfrpcd: str = "msfrpcd"
    host: str = "127.0.0.1"
    port: int = 0
    process: subprocess.Popen | None = None
    username: str = "msf"
    password: str = ""

    def start(self) -> "MetasploitRpcClient":
        self.password = rpc_secret()
        self.port = self.port or _free_port()
        secret_dir = os.path.join(self.run_dir, "secrets")
        os.makedirs(secret_dir, mode=0o700, exist_ok=True)
        secret_path = os.path.join(secret_dir, "metasploit-rpc.secret")
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(self.password)
        self.process = subprocess.Popen(
            [self.msfrpcd, "-f", "-a", self.host, "-p", str(self.port), "-U", self.username, "-P", self.password],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.run_dir,
        )
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                client = MetasploitRpcClient(self.host, self.port, self.username, self.password)
                client.login()
                return client
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("metasploit RPC did not start")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        secret = os.path.join(self.run_dir, "secrets", "metasploit-rpc.secret")
        try:
            os.remove(secret)
        except OSError:
            pass


class MetasploitRpcClient:
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host, self.port, self.username, self.password, self.token = host, port, username, password, ""

    def call(self, method: str, *args: Any) -> dict[str, Any]:
        values: list[Any] = [method]
        if method != "auth.login":
            values.append(self.token)
        values.extend(args)
        with socket.create_connection((self.host, self.port), timeout=10) as conn:
            conn.sendall(msgpack.packb(values, use_bin_type=True))
            data = conn.recv(10 * 1024 * 1024)
        result = msgpack.unpackb(data, raw=False)
        if not isinstance(result, dict):
            raise RuntimeError("invalid Metasploit RPC response")
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def login(self) -> None:
        result = self.call("auth.login", self.username, self.password)
        self.token = str(result.get("token") or "")
        if not self.token:
            raise RuntimeError("Metasploit RPC authentication failed")

    # Discovery protocol used by ExploitCompiler.
    def search(self, cve_id: str):
        result = self.call("module.search", cve_id)
        return result.get("modules", result.get("results", []))

    def info(self, module_name: str) -> dict[str, Any]:
        module_type, _, name = module_name.partition("/")
        return self.call("module.info", module_type, name)

    def revision(self, module_name: str) -> tuple[str, str]:
        # RPC metadata is the reproducible source available to this client.
        info = self.info(module_name)
        return str(info.get("fullname") or module_name), str(info.get("sha256") or "")

    def check(self, module: str, options: dict[str, str]) -> dict[str, Any]:
        kind, _, name = module.partition("/")
        return self.call("module.check", kind, name, options)

    def execute(self, module: str, options: dict[str, str]) -> dict[str, Any]:
        kind, _, name = module.partition("/")
        return self.call("module.execute", kind, name, options)

    def wait_for_session(self, job_id: str, timeout: int = 30) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            sessions = self.call("session.list")
            for sid, value in sessions.items():
                if sid != "error" and isinstance(value, dict):
                    return {"id": sid, **value}
            time.sleep(1)
        return None

    def stop_session(self, session_id: str) -> None:
        self.call("session.stop", session_id)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
