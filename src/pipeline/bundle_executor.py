"""Fail-closed execution bridge for independent evaluator/oracle bundles."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from typing import Any, Iterator, Mapping


class BundleExecutionError(RuntimeError):
    """An independent evaluation bundle violated its execution contract."""


_COMMIT_LENGTH = 40
_HASH_LENGTH = 64
_MAX_OUTPUT_BYTES = 64 * 1024
_SUPPORTED_SCHEMA = "2.0.0"


def _archive_target(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise BundleExecutionError("bundle archive contains an unsafe path")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise BundleExecutionError("bundle archive escapes its extraction root")
    return target


def _extract_zip(bundle: Path, root: Path) -> None:
    seen: set[Path] = set()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            target = _archive_target(root, member.filename)
            if target in seen:
                raise BundleExecutionError("bundle archive contains duplicate paths")
            seen.add(target)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise BundleExecutionError("bundle archive contains a symlink")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                copyfileobj(source, destination)
            target.chmod((member.external_attr >> 16) & 0o777 or 0o600)


def _extract_tar(bundle: Path, root: Path) -> None:
    seen: set[Path] = set()
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            target = _archive_target(root, member.name)
            if target in seen:
                raise BundleExecutionError("bundle archive contains duplicate paths")
            seen.add(target)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise BundleExecutionError("bundle archive contains a non-regular file")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise BundleExecutionError("bundle archive member cannot be read")
            with source, target.open("wb") as destination:
                copyfileobj(source, destination)
            target.chmod(member.mode & 0o777 or 0o600)


@contextmanager
def _materialize_bundle(bundle: Path) -> Iterator[Path]:
    if bundle.is_dir():
        yield bundle
        return
    if not bundle.is_file():
        raise BundleExecutionError(f"bundle is missing: {bundle}")
    with tempfile.TemporaryDirectory(prefix="runtime-bundle-") as temporary:
        root = Path(temporary).resolve()
        try:
            if zipfile.is_zipfile(bundle):
                _extract_zip(bundle, root)
            elif tarfile.is_tarfile(bundle):
                _extract_tar(bundle, root)
            else:
                raise BundleExecutionError("file bundle must be a zip or tar archive")
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            raise BundleExecutionError("bundle archive could not be materialized") from exc
        yield root


def _read_manifest(root: Path, expected_kind: str) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    try:
        manifest_resolved = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise BundleExecutionError("bundle manifest is missing") from exc
    if not manifest_resolved.is_relative_to(root.resolve()) or not manifest_resolved.is_file():
        raise BundleExecutionError("bundle manifest must be a regular file inside the bundle")
    try:
        value = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleExecutionError("bundle manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise BundleExecutionError("bundle manifest must be a JSON object")
    if value.get("schema_version") != _SUPPORTED_SCHEMA:
        raise BundleExecutionError("bundle manifest schema is unsupported")
    if value.get("kind") != expected_kind:
        raise BundleExecutionError("bundle manifest kind does not match the registered bundle")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != _COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise BundleExecutionError("bundle manifest source_commit is invalid")
    entrypoint = value.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise BundleExecutionError("bundle manifest entrypoint is missing")
    if expected_kind == "evaluator":
        feature_schema_hash = value.get("feature_schema_hash")
        image_digest = value.get("image_digest")
        if not isinstance(feature_schema_hash, str) or len(feature_schema_hash) != _HASH_LENGTH or any(
            character not in "0123456789abcdef" for character in feature_schema_hash
        ):
            raise BundleExecutionError("evaluator feature_schema_hash is invalid")
        if not isinstance(image_digest, str) or not image_digest.startswith("sha256:") or len(image_digest) != 71:
            raise BundleExecutionError("evaluator image_digest is invalid")
        if any(character not in "0123456789abcdef" for character in image_digest[7:]):
            raise BundleExecutionError("evaluator image_digest is invalid")
    else:
        image_digest = value.get("image_digest")
        if not isinstance(image_digest, str) or not image_digest.startswith("sha256:") or len(image_digest) != 71:
            raise BundleExecutionError("oracle image_digest is invalid")
        if any(character not in "0123456789abcdef" for character in image_digest[7:]):
            raise BundleExecutionError("oracle image_digest is invalid")
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or any(part in {"", ".", ".."} for part in entrypoint_path.parts):
        raise BundleExecutionError("bundle entrypoint must be a safe relative path")
    try:
        resolved_entrypoint = (root / entrypoint_path).resolve(strict=True)
    except OSError as exc:
        raise BundleExecutionError("bundle entrypoint is missing") from exc
    if not resolved_entrypoint.is_relative_to(root.resolve()) or not resolved_entrypoint.is_file():
        raise BundleExecutionError("bundle entrypoint must be inside the bundle")
    if not os.access(resolved_entrypoint, os.X_OK):
        raise BundleExecutionError("bundle entrypoint is not executable")
    return value, resolved_entrypoint


def _isolated_environment(home: Path, temporary: Path) -> dict[str, str]:
    """Return a minimal environment with cloud credentials/configuration absent."""

    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "XDG_CONFIG_HOME": str(temporary / "config"),
        "XDG_CACHE_HOME": str(temporary / "cache"),
        "CLOUDSDK_CONFIG": str(temporary / "gcloud"),
        "PYTHONNOUSERSITE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


class IndependentBundleExecutor:
    """Execute registered evaluator/oracle bundles without credential inheritance."""

    def __init__(
        self, *, evaluator_bundle: str | Path, oracle_bundle: str | Path,
        hidden_case_root: str | Path | None = None,
        timeout_seconds: float = 30.0, max_output_bytes: int = _MAX_OUTPUT_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("bundle timeout must be positive")
        if max_output_bytes <= 0:
            raise ValueError("bundle output limit must be positive")
        evaluator = Path(evaluator_bundle).resolve()
        oracle = Path(oracle_bundle).resolve()
        if evaluator == oracle:
            raise ValueError("evaluator and oracle bundles must be independent")
        hidden_root = Path(hidden_case_root).resolve() if hidden_case_root is not None else None
        if hidden_root is not None and (hidden_root.is_symlink() or not hidden_root.is_dir()):
            raise BundleExecutionError("oracle hidden case root must be a real directory")
        self._registered = {evaluator: "evaluator", oracle: "oracle"}
        self.hidden_case_root = hidden_root
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def __call__(self, bundle_path: Path, run_dir: Path, run_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
        bundle = Path(bundle_path).resolve()
        expected_kind = self._registered.get(bundle)
        if expected_kind is None:
            raise BundleExecutionError("bundle path is not registered as evaluator or oracle")
        run_root = Path(run_dir).resolve()
        if not run_root.is_dir():
            raise BundleExecutionError("bundle run directory is missing")
        try:
            serialized_artifact = json.dumps(dict(run_artifact), sort_keys=True, indent=2) + "\n"
        except (TypeError, ValueError) as exc:
            raise BundleExecutionError("run artifact is not JSON serializable") from exc

        with _materialize_bundle(bundle) as root, tempfile.TemporaryDirectory(
            prefix=".bundle-exec-", dir=str(run_root)
        ) as temporary:
            temporary_root = Path(temporary)
            _, entrypoint = _read_manifest(root, expected_kind)
            input_path = temporary_root / "run-artifact.json"
            output_path = temporary_root / "verdict.json"
            input_path.write_text(serialized_artifact, encoding="utf-8")
            input_path.chmod(0o444)
            output_path.touch(mode=0o600)
            environment = _isolated_environment(temporary_root / "home", temporary_root / "tmp")
            for directory in (environment["HOME"], environment["TMPDIR"], environment["XDG_CONFIG_HOME"], environment["XDG_CACHE_HOME"], environment["CLOUDSDK_CONFIG"]):
                Path(directory).mkdir(parents=True, exist_ok=True)
            argv = [
                str(entrypoint), "--run-dir", str(run_root),
                "--run-artifact", str(input_path), "--output", str(output_path),
            ]
            if expected_kind == "oracle":
                if self.hidden_case_root is None:
                    raise BundleExecutionError("oracle execution requires a hidden case root")
                argv.extend(["--hidden-case-root", str(self.hidden_case_root)])
            try:
                completed = subprocess.run(
                    argv, cwd=str(run_root), env=environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=self.timeout_seconds, close_fds=True,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BundleExecutionError("bundle entrypoint failed or timed out") from exc
            if completed.returncode != 0:
                raise BundleExecutionError("bundle entrypoint returned failure")
            try:
                if output_path.stat().st_size > self.max_output_bytes:
                    raise BundleExecutionError("bundle verdict exceeds output limit")
                verdict = json.loads(output_path.read_text(encoding="utf-8"))
            except BundleExecutionError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BundleExecutionError("bundle verdict is not valid JSON") from exc
            if not isinstance(verdict, dict) or verdict.get("schema_version") != "2.0.0" or verdict.get("kind") != expected_kind or verdict.get("status") not in {"passed", "failed"}:
                raise BundleExecutionError("bundle verdict did not pass")
            if verdict.get("status") == "passed" and not isinstance(verdict.get("outcome"), dict):
                raise BundleExecutionError("passed bundle verdict is missing outcome")
            if verdict.get("status") == "failed" and not isinstance(verdict.get("errors"), list):
                raise BundleExecutionError("failed bundle verdict is missing errors")
            return dict(verdict)
