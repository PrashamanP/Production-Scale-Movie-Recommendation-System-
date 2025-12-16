import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git_command(args: list[str]) -> Optional[str]:
    """Executes a git command relative to the repository root."""
    try:
        result = subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return result.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_git_metadata() -> Dict[str, Optional[str]]:
    """Returns commit metadata for the current repository, if available."""
    commit = os.environ.get("PIPELINE_GIT_COMMIT") or _run_git_command(
        ["rev-parse", "HEAD"]
    )
    branch = os.environ.get("PIPELINE_GIT_BRANCH") or _run_git_command(
        ["rev-parse", "--abbrev-ref", "HEAD"]
    )
    dirty_flag = os.environ.get("PIPELINE_GIT_DIRTY")
    if dirty_flag is None:
        status = _run_git_command(["status", "--porcelain"])
        dirty_flag = "1" if status else "0" if status is not None else None

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty_flag,
    }


def compute_file_sha256(path: str) -> Optional[str]:
    """Returns the sha256 checksum for the specified file."""
    if not path or not os.path.exists(path):
        return None

    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def summarize_artifact(path: str) -> Dict[str, Optional[str]]:
    """Captures immutable metadata about a produced artifact."""
    if not path:
        return {"path": None, "sha256": None, "size_bytes": None}

    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        size_bytes = None

    return {
        "path": os.path.relpath(path, REPO_ROOT) if os.path.isabs(path) else path,
        "sha256": compute_file_sha256(path),
        "size_bytes": size_bytes,
    }


def build_model_manifest(
    *,
    model_family: str,
    artifact_paths: Dict[str, str],
    hyperparameters: Dict[str, Any],
    data_profile: Optional[Dict[str, Any]],
    data_path: str,
    data_version: Optional[str] = None,
    model_version: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds a structured manifest describing how the model was trained.
    """
    now = datetime.now(timezone.utc)
    git_info = collect_git_metadata()
    data_profile = data_profile or {}
    dataset_sha = compute_file_sha256(data_path) if data_path else None
    dataset_version = data_version or dataset_sha

    profile = {
        "dataset_path": os.path.relpath(data_path, REPO_ROOT)
        if data_path and os.path.isabs(data_path)
        else data_path,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha,
        "row_count": data_profile.get("row_count"),
        "unique_users": data_profile.get("unique_users"),
        "unique_movies": data_profile.get("unique_movies"),
    }

    try:
        profile["dataset_size_bytes"] = (
            os.path.getsize(data_path) if data_path and os.path.exists(data_path) else None
        )
    except OSError:
        profile["dataset_size_bytes"] = None

    manifest = {
        "model_family": model_family,
        "model_version": model_version
        or f"{model_family}-{now.strftime('%Y%m%d%H%M%S')}",
        "trained_at": now.isoformat(),
        "pipeline": git_info,
        "hyperparameters": hyperparameters,
        "data": profile,
        "artifacts": {
            name: summarize_artifact(path) for name, path in artifact_paths.items()
        },
        # Deployment tracking (updated by CI/CD pipeline post-deployment)
        "deployed_at": None,
        "deployment_build": None,
        "deployment_commit": None,
    }

    if extra_metadata:
        manifest["metadata"] = extra_metadata

    return manifest


def persist_manifest(
    manifest: Dict[str, Any],
    artifact_dir: str,
    model_version: Optional[str] = None,
) -> str:
    """
    Writes the manifest both to a versioned file and a canonical pointer.
    """
    artifact_dir_path = Path(artifact_dir)
    manifests_dir = artifact_dir_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_copy = dict(manifest)
    manifest_copy.setdefault(
        "model_version",
        model_version or manifest_copy.get("model_version"),
    )

    serialized = json.dumps(manifest_copy, sort_keys=True).encode("utf-8")
    manifest_hash = hashlib.sha256(serialized).hexdigest()
    manifest_copy["manifest_sha256"] = manifest_hash

    manifest_version = manifest_copy["model_version"]
    versioned_path = manifests_dir / f"{manifest_version}.json"
    with open(versioned_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_copy, handle, indent=2, sort_keys=True)

    latest_path = artifact_dir_path / "model_manifest.json"
    shutil.copy2(versioned_path, latest_path)
    
    # Backward compatibility: also write provenance.json
    legacy_path = artifact_dir_path / "provenance.json"
    shutil.copy2(versioned_path, legacy_path)
    return str(latest_path)
