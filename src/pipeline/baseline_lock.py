"""Generate baseline locks from detached Git trees and Docker inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_tree_hash(root: Path, path: Path | None = None) -> str:
    """Return the Git tree object, excluding ignored runtime/cache content."""
    repository = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    target = (path or root).resolve()
    try:
        relative = target.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValueError(f"Git tree path {target} is outside repository {repository}") from exc
    expression = "HEAD^{tree}" if relative in {"", "."} else f"HEAD:{relative}"
    tree_hash = _git(repository, "rev-parse", expression)
    if len(tree_hash) != 40:
        raise ValueError(f"Git tree hash is invalid for {target}")
    return tree_hash


def _directory_hash(path: Path) -> str:
    """Hash a non-Git build context without trusting archive metadata."""
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _build_context_hash(source_root: Path, context: Path) -> tuple[str, str]:
    """Return an observed Git tree or deterministic staged-context hash."""
    try:
        return _git_tree_hash(source_root, context), "git_tree_object"
    except ValueError:
        if not context.is_dir():
            raise
        return _directory_hash(context), "directory_sha256"


def _tree_hash(root: Path) -> str:
    """Compatibility name for callers that previously used filesystem hashing."""
    return _git_tree_hash(root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_path(spec: Mapping[str, Any], *keys: str) -> Path:
    for key in keys:
        value = str(spec.get(key, "")).strip()
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                raise ValueError(f"baseline spec {key} is not a file: {path}")
            return path
    raise ValueError(f"baseline spec requires one of: {', '.join(keys)}")


def _adapter_bundle(spec: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    bundle = spec.get("adapter_bundle")
    if not isinstance(bundle, Mapping):
        bundle = {}
    paths = {
        "common": str(
            bundle.get("common")
            or spec.get("common_adapter_path")
            or spec.get("common_adapter")
            or ""
        ),
        "framework": str(
            bundle.get("framework")
            or spec.get("framework_adapter_path")
            or spec.get("framework_adapter")
            or ""
        ),
        "wrapper": str(
            bundle.get("wrapper")
            or spec.get("wrapper_path")
            or spec.get("wrapper")
            or ""
        ),
        "runtime": str(
            bundle.get("runtime")
            or spec.get("runtime_adapter_path")
            or spec.get("runtime_adapter")
            or ""
        ),
        "client_driver": str(
            bundle.get("client_driver")
            or spec.get("client_driver_path")
            or spec.get("client_driver")
            or ""
        ),
        "readiness_transport": str(
            bundle.get("readiness_transport")
            or spec.get("readiness_transport_path")
            or spec.get("readiness_transport")
            or ""
        ),
    }
    for key, value in bundle.items():
        if key not in {"contract_version"} and value:
            paths[key] = str(value)
    missing = [role for role, value in paths.items() if not value]
    if missing:
        raise ValueError(f"adapter bundle missing path(s): {', '.join(missing)}")
    resolved = {role: str(Path(value).resolve()) for role, value in paths.items()}
    for role, value in resolved.items():
        if not Path(value).is_file():
            raise ValueError(f"adapter bundle {role} is not a file: {value}")
    version = str(
        bundle.get("contract_version")
        or spec.get("adapter_contract_version")
        or spec.get("adapter_version")
        or ""
    ).strip()
    if not version:
        raise ValueError("adapter bundle requires a contract_version")
    return resolved, version


def _adapter_bundle_hash(paths: Mapping[str, str], contract_version: str) -> str:
    payload = {
        "contract_version": contract_version,
        "files": {
            role: _sha256_file(Path(path)) for role, path in sorted(paths.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate_baseline_lock(
    specs: Sequence[Mapping[str, Any]], *, output: str | Path | None = None,
) -> dict[str, Any]:
    """Read Git HEAD/tree/remote, recipe/context, adapter bundle, and image digest."""
    if len(specs) != 4:
        raise ValueError("baseline lock requires exactly four baseline specs")
    entries: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec.get("name", ""))
        root = Path(str(spec.get("path", ""))).resolve()
        if not name or not root.is_dir():
            raise ValueError("baseline spec requires an existing path and name")
        if _git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"baseline {name} worktree is dirty")
        commit = _git(root, "rev-parse", "HEAD")
        remotes = _git(root, "remote", "-v").splitlines()
        remote = next((line.split()[1] for line in remotes if "(fetch)" in line), "")
        expected_remote = str(spec.get("repo_url", ""))
        if expected_remote and remote and remote != expected_remote:
            raise ValueError(f"baseline {name} remote mismatch")

        recipe = _required_path(spec, "docker_recipe_path", "recipe_path", "dockerfile")
        context_value = str(
            spec.get("build_context_path") or spec.get("build_context") or root
        )
        context = Path(context_value).resolve()
        if not context.is_dir():
            raise ValueError(f"baseline {name} build context is not a directory: {context}")
        context_hash, context_hash_kind = _build_context_hash(root, context)
        adapter_paths, contract_version = _adapter_bundle(spec)

        image = str(spec.get("image", ""))
        if not image:
            raise ValueError(f"baseline {name} image is required")
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode or not inspect.stdout.strip().startswith("sha256:"):
            raise ValueError(f"baseline {name} image is unavailable or unpinned")

        labels_result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
        image_labels: dict[str, Any] = {}
        if labels_result.returncode == 0:
            try:
                parsed_labels = json.loads(labels_result.stdout.strip())
                if isinstance(parsed_labels, Mapping):
                    image_labels = dict(parsed_labels)
            except json.JSONDecodeError:
                # Older test doubles and Docker versions may only expose the ID.
                image_labels = {}
        input_hashes = spec.get("input_hashes")
        if not isinstance(input_hashes, Mapping):
            input_hashes = {}
        normalized_input_hashes = {
            str(key): str(value) for key, value in sorted(input_hashes.items())
        }
        dependency_lock_path = str(spec.get("dependency_lock_path", "")).strip()
        dependency_lock_hash = ""
        staged_dependency_lock_hash = ""
        if dependency_lock_path:
            dependency_lock = Path(dependency_lock_path).resolve()
            if not dependency_lock.is_file():
                raise ValueError(f"baseline {name} dependency lock is not a file: {dependency_lock}")
            staged_dependency_lock_hash = _sha256_file(dependency_lock)
            declared_dependency_lock_hash = str(spec.get("dependency_lock_hash", "")).strip()
            if declared_dependency_lock_hash:
                if len(declared_dependency_lock_hash) != 64 or any(c not in "0123456789abcdef" for c in declared_dependency_lock_hash):
                    raise ValueError(f"baseline {name} dependency_lock_hash is not a SHA-256")
                dependency_lock_hash = declared_dependency_lock_hash
            else:
                dependency_lock_hash = staged_dependency_lock_hash
        label_dependency_lock_hash = str(image_labels.get("com.veriplanpt.dependency-lock-hash", ""))
        if dependency_lock_hash and label_dependency_lock_hash and label_dependency_lock_hash != dependency_lock_hash:
            raise ValueError(f"baseline {name} dependency lock hash does not match image label")
        adapter_bundle_hash = _adapter_bundle_hash(adapter_paths, contract_version)
        label_adapter_bundle_hash = str(image_labels.get("com.veriplanpt.adapter-bundle-hash", ""))
        if label_adapter_bundle_hash and label_adapter_bundle_hash != adapter_bundle_hash:
            raise ValueError(f"baseline {name} adapter bundle hash does not match image label")
        os_packages = spec.get("os_package_requirements", [])
        if not isinstance(os_packages, list) or any(not str(item).strip() for item in os_packages):
            raise ValueError(f"baseline {name} os_package_requirements must be a list of non-empty strings")

        entry: dict[str, Any] = {
                "name": name,
                "repo_url": expected_remote or remote,
                "commit": commit,
                "tree_hash": _git_tree_hash(root),
                "tree_hash_kind": "git_tree_object",
                "build_context_tree_hash": context_hash,
                "build_context_hash_kind": context_hash_kind,
                "docker_recipe_hash": _sha256_file(recipe),
                "recipe_hash": _sha256_file(recipe),
                "image": image,
                "image_id": inspect.stdout.strip(),
                "image_digest": inspect.stdout.strip(),
                "adapter_version": contract_version,
                "adapter_contract_version": contract_version,
                "adapter_bundle_hash": adapter_bundle_hash,
            }
        if normalized_input_hashes:
            entry["input_hashes"] = normalized_input_hashes
        if dependency_lock_hash:
            entry["dependency_lock_hash"] = dependency_lock_hash
        if staged_dependency_lock_hash:
            entry["staged_dependency_lock_hash"] = staged_dependency_lock_hash
        entry["os_package_requirements"] = [str(item) for item in os_packages]
        if image_labels:
            entry["image_labels"] = image_labels
        entries.append(entry)
    lock = {"schema_version": "2.0.0", "baselines": entries}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    """Create a lock exclusively from inspected Git and Docker state.

    The inventory intentionally contains local paths and image references only;
    all commit, remote, tree, recipe, image digest, and adapter hash values are
    observed here rather than accepted as hand-authored lock data.
    """
    parser = argparse.ArgumentParser(
        description="Generate a VeriPlanPT baseline lock from a build inventory."
    )
    parser.add_argument(
        "--inventory", required=True, help="JSON inventory with exactly four baseline specs"
    )
    parser.add_argument("--output", required=True, help="Destination baseline.lock.json")
    args = parser.parse_args(argv)
    inventory_path = Path(args.inventory)
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read build inventory: {exc}")
    specs = inventory.get("baselines") if isinstance(inventory, dict) else inventory
    if not isinstance(specs, list):
        parser.error("inventory must be an array or an object containing a 'baselines' array")
    try:
        lock = generate_baseline_lock(specs, output=args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"baseline_count": len(lock["baselines"]), "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
