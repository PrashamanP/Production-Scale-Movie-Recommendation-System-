import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ProvenanceTracker:
    """
    Lightweight registry for loading manifest metadata at serving time.
    """

    def __init__(self, manifest: Optional[Dict[str, Any]] = None, path: Optional[str] = None):
        self._manifest = manifest or {}
        self._path = path

    @classmethod
    def load(
        cls,
        artifacts_dir: str,
        manifest_filename: str = "model_manifest.json",
    ) -> "ProvenanceTracker":
        """Loads the manifest from disk if available."""
        artifacts_path = Path(artifacts_dir)
        manifest_path = artifacts_path / manifest_filename
        manifest = {}

        if manifest_path.exists():
            manifest = cls._safe_load(manifest_path)
        else:
            manifests_dir = artifacts_path / "manifests"
            if manifests_dir.exists():
                candidates = sorted(manifests_dir.glob("*.json"))
                if candidates:
                    manifest_path = candidates[-1]
                    manifest = cls._safe_load(manifest_path)

        return cls(manifest, str(manifest_path) if manifest else None)

    @staticmethod
    def _safe_load(path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def available(self) -> bool:
        return bool(self._manifest)

    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    def manifest_path(self) -> Optional[str]:
        return self._path

    def context_tags(self) -> Dict[str, Optional[str]]:
        if not self.available():
            return {}

        pipeline = self._manifest.get("pipeline", {})
        data = self._manifest.get("data", {})
        return {
            "model_version": self._manifest.get("model_version"),
            "model_family": self._manifest.get("model_family"),
            "pipeline_commit": pipeline.get("git_commit"),
            "pipeline_branch": pipeline.get("git_branch"),
            "pipeline_dirty": pipeline.get("git_dirty"),
            "data_version": data.get("dataset_version"),
            "data_sha256": data.get("dataset_sha256"),
            "manifest_sha256": self._manifest.get("manifest_sha256"),
        }

    def response_headers(self) -> Dict[str, str]:
        tags = self.context_tags()
        return {
            "X-Model-Version": tags["model_version"] or "",
            "X-Pipeline-Commit": tags["pipeline_commit"] or "",
            "X-Data-Version": tags["data_version"] or "",
            "X-Model-Manifest": tags["manifest_sha256"] or "",
        } if tags else {}

    def extra_log_suffix(self) -> str:
        """
        Generates a comma-separated suffix for log enrichment.
        """
        tags = self.context_tags()
        if not tags:
            return ""

        serialised = ", ".join(
            f"{key}={value}"
            for key, value in tags.items()
            if value not in (None, "")
        )
        return serialised
