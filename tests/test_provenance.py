import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.provenance.manifest import (
    build_model_manifest,
    persist_manifest,
    collect_git_metadata,
    compute_file_sha256,
    summarize_artifact,
    _run_git_command,
)
from src.provenance.tracker import ProvenanceTracker


# =============================================================================
# Tests for manifest.py
# =============================================================================

class TestRunGitCommand:
    """Tests for _run_git_command helper function."""

    def test_successful_git_command(self):
        """Git command should return stripped output on success."""
        with patch("subprocess.check_output") as mock_output:
            mock_output.return_value = b"abc123\n"
            result = _run_git_command(["rev-parse", "HEAD"])
            assert result == "abc123"

    def test_git_command_failure(self):
        """Git command should return None on CalledProcessError."""
        with patch("subprocess.check_output") as mock_output:
            from subprocess import CalledProcessError
            mock_output.side_effect = CalledProcessError(1, "git")
            result = _run_git_command(["rev-parse", "HEAD"])
            assert result is None

    def test_git_not_found(self):
        """Git command should return None when git is not installed."""
        with patch("subprocess.check_output") as mock_output:
            mock_output.side_effect = FileNotFoundError("git not found")
            result = _run_git_command(["rev-parse", "HEAD"])
            assert result is None


class TestCollectGitMetadata:
    """Tests for collect_git_metadata function."""

    def test_uses_env_vars_when_available(self):
        """Should prefer environment variables over git commands."""
        env_vars = {
            "PIPELINE_GIT_COMMIT": "env-commit-sha",
            "PIPELINE_GIT_BRANCH": "env-branch",
            "PIPELINE_GIT_DIRTY": "1",
        }
        with patch.dict("os.environ", env_vars, clear=False):
            result = collect_git_metadata()
            assert result["git_commit"] == "env-commit-sha"
            assert result["git_branch"] == "env-branch"
            assert result["git_dirty"] == "1"

    def test_falls_back_to_git_commands(self):
        """Should fall back to git commands when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.provenance.manifest._run_git_command") as mock_git:
                mock_git.side_effect = [
                    "git-commit-sha",  # rev-parse HEAD
                    "main",  # rev-parse --abbrev-ref HEAD
                    "",  # status --porcelain (clean)
                ]
                result = collect_git_metadata()
                assert result["git_commit"] == "git-commit-sha"
                assert result["git_branch"] == "main"
                assert result["git_dirty"] == "0"

    def test_dirty_flag_when_changes_present(self):
        """Should set dirty flag to 1 when git status shows changes."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.provenance.manifest._run_git_command") as mock_git:
                mock_git.side_effect = [
                    "commit-sha",
                    "feature-branch",
                    " M src/file.py",  # Modified file
                ]
                result = collect_git_metadata()
                assert result["git_dirty"] == "1"

    def test_dirty_flag_none_when_git_fails(self):
        """Should set dirty flag to None when git status fails."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.provenance.manifest._run_git_command") as mock_git:
                mock_git.side_effect = [
                    "commit-sha",
                    "main",
                    None,  # git status failed
                ]
                result = collect_git_metadata()
                assert result["git_dirty"] is None


class TestComputeFileSha256:
    """Tests for compute_file_sha256 function."""

    def test_computes_sha256_correctly(self, tmp_path):
        """Should compute correct SHA256 hash for file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        result = compute_file_sha256(str(test_file))
        # SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert result == expected

    def test_returns_none_for_empty_path(self):
        """Should return None for empty path."""
        result = compute_file_sha256("")
        assert result is None

    def test_returns_none_for_nonexistent_file(self):
        """Should return None for non-existent file."""
        result = compute_file_sha256("/nonexistent/path/file.txt")
        assert result is None

    def test_handles_large_files(self, tmp_path):
        """Should handle files larger than chunk size."""
        test_file = tmp_path / "large.bin"
        # Write more than 1MB to test chunking
        data = b"x" * (2 * 1024 * 1024)
        test_file.write_bytes(data)
        result = compute_file_sha256(str(test_file))
        assert result is not None
        assert len(result) == 64  # SHA256 hex digest length


class TestSummarizeArtifact:
    """Tests for summarize_artifact function."""

    def test_summarizes_existing_file(self, tmp_path):
        """Should capture path, sha256, and size for existing file."""
        test_file = tmp_path / "artifact.bin"
        test_file.write_bytes(b"artifact data")
        result = summarize_artifact(str(test_file))
        assert result["sha256"] is not None
        assert result["size_bytes"] == 13
        assert "path" in result

    def test_empty_path_returns_none_values(self):
        """Should return None values for empty path."""
        result = summarize_artifact("")
        assert result["path"] is None
        assert result["sha256"] is None
        assert result["size_bytes"] is None

    def test_handles_oserror_on_getsize(self, tmp_path):
        """Should handle OSError when getting file size."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data", encoding="utf-8")
        with patch("os.path.getsize") as mock_size:
            mock_size.side_effect = OSError("Permission denied")
            result = summarize_artifact(str(test_file))
            assert result["size_bytes"] is None
            # SHA256 should still work
            assert result["sha256"] is not None

    def test_converts_absolute_to_relative_path(self, tmp_path):
        """Should convert absolute paths to relative paths."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("data", encoding="utf-8")
        result = summarize_artifact(str(test_file))
        # Path should not start with /
        assert not result["path"].startswith("/") or "tmp" in result["path"]


class TestBuildModelManifest:
    """Tests for build_model_manifest function."""

    def test_builds_complete_manifest(self, tmp_path):
        """Should build manifest with all required fields."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        model_file = tmp_path / "model.npz"
        model_file.write_bytes(b"model-data")

        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={"model": str(model_file)},
            hyperparameters={"factors": 10, "iterations": 20},
            data_profile={"row_count": 1, "unique_users": 1, "unique_movies": 1},
            data_path=str(data_file),
            data_version="v1.0",
            model_version="als-v1",
        )

        assert manifest["model_family"] == "als"
        assert manifest["model_version"] == "als-v1"
        assert "trained_at" in manifest
        assert manifest["hyperparameters"]["factors"] == 10
        assert manifest["data"]["dataset_version"] == "v1.0"
        assert manifest["data"]["row_count"] == 1
        assert "model" in manifest["artifacts"]
        assert manifest["deployed_at"] is None
        assert manifest["deployment_build"] is None
        assert manifest["deployment_commit"] is None

    def test_auto_generates_model_version(self, tmp_path):
        """Should auto-generate model version from timestamp if not provided."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("data", encoding="utf-8")

        manifest = build_model_manifest(
            model_family="svd",
            artifact_paths={},
            hyperparameters={},
            data_profile=None,
            data_path=str(data_file),
        )

        assert manifest["model_version"].startswith("svd-")
        assert len(manifest["model_version"]) > 10

    def test_uses_sha_as_version_when_no_data_version(self, tmp_path):
        """Should use dataset SHA as version when data_version not provided."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("dataset content", encoding="utf-8")

        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={},
            hyperparameters={},
            data_profile=None,
            data_path=str(data_file),
        )

        # dataset_version should be the SHA256
        assert manifest["data"]["dataset_version"] == manifest["data"]["dataset_sha256"]

    def test_handles_none_data_profile(self, tmp_path):
        """Should handle None data_profile gracefully."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("x", encoding="utf-8")

        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={},
            hyperparameters={},
            data_profile=None,
            data_path=str(data_file),
        )

        assert manifest["data"]["row_count"] is None
        assert manifest["data"]["unique_users"] is None
        assert manifest["data"]["unique_movies"] is None

    def test_includes_extra_metadata(self, tmp_path):
        """Should include extra_metadata in manifest when provided."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("x", encoding="utf-8")

        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={},
            hyperparameters={},
            data_profile=None,
            data_path=str(data_file),
            extra_metadata={"custom_field": "custom_value", "experiment_id": 42},
        )

        assert "metadata" in manifest
        assert manifest["metadata"]["custom_field"] == "custom_value"
        assert manifest["metadata"]["experiment_id"] == 42

    def test_handles_nonexistent_data_path(self, tmp_path):
        """Should handle non-existent data path."""
        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={},
            hyperparameters={},
            data_profile=None,
            data_path="/nonexistent/data.csv",
        )

        assert manifest["data"]["dataset_sha256"] is None
        assert manifest["data"]["dataset_size_bytes"] is None


class TestPersistManifest:
    """Tests for persist_manifest function."""

    def test_creates_manifests_directory(self, tmp_path):
        """Should create manifests subdirectory."""
        manifest = {"model_version": "test-v1", "data": {}}
        persist_manifest(manifest, str(tmp_path))

        manifests_dir = tmp_path / "manifests"
        assert manifests_dir.exists()
        assert manifests_dir.is_dir()

    def test_writes_versioned_manifest_file(self, tmp_path):
        """Should write versioned JSON file in manifests directory."""
        manifest = {"model_version": "als-v2", "data": {}}
        persist_manifest(manifest, str(tmp_path))

        versioned_file = tmp_path / "manifests" / "als-v2.json"
        assert versioned_file.exists()

    def test_writes_latest_manifest_pointer(self, tmp_path):
        """Should write model_manifest.json as latest pointer."""
        manifest = {"model_version": "test-v1", "data": {}}
        result_path = persist_manifest(manifest, str(tmp_path))

        latest_file = tmp_path / "model_manifest.json"
        assert latest_file.exists()
        assert result_path == str(latest_file)

    def test_writes_backward_compatibility_file(self, tmp_path):
        """Should write provenance.json for backward compatibility."""
        manifest = {"model_version": "test-v1", "data": {}}
        persist_manifest(manifest, str(tmp_path))

        legacy_file = tmp_path / "provenance.json"
        assert legacy_file.exists()

    def test_adds_manifest_sha256(self, tmp_path):
        """Should add manifest_sha256 to the written manifest."""
        manifest = {"model_version": "test-v1", "data": {}}
        persist_manifest(manifest, str(tmp_path))

        latest_file = tmp_path / "model_manifest.json"
        with open(latest_file, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert "manifest_sha256" in saved
        assert len(saved["manifest_sha256"]) == 64

    def test_all_files_have_same_content(self, tmp_path):
        """Should ensure all three files have identical content."""
        manifest = {"model_version": "test-v1", "data": {}}
        persist_manifest(manifest, str(tmp_path))

        versioned = tmp_path / "manifests" / "test-v1.json"
        latest = tmp_path / "model_manifest.json"
        legacy = tmp_path / "provenance.json"

        with open(versioned, "r") as f1, open(latest, "r") as f2, open(legacy, "r") as f3:
            assert f1.read() == f2.read() == f3.read()

    def test_respects_model_version_parameter(self, tmp_path):
        """Should use model_version parameter when provided."""
        manifest = {"data": {}}
        persist_manifest(manifest, str(tmp_path), model_version="override-v1")

        versioned_file = tmp_path / "manifests" / "override-v1.json"
        assert versioned_file.exists()

        with open(versioned_file, "r") as f:
            saved = json.load(f)
        assert saved["model_version"] == "override-v1"


# =============================================================================
# Tests for tracker.py
# =============================================================================

class TestProvenanceTrackerInit:
    """Tests for ProvenanceTracker initialization."""

    def test_init_with_manifest(self):
        """Should initialize with provided manifest."""
        manifest = {"model_version": "v1", "model_family": "als"}
        tracker = ProvenanceTracker(manifest)
        assert tracker.available()
        assert tracker.manifest() == manifest

    def test_init_without_manifest(self):
        """Should initialize with empty manifest when none provided."""
        tracker = ProvenanceTracker()
        assert not tracker.available()
        assert tracker.manifest() == {}

    def test_init_with_path(self):
        """Should store manifest path when provided."""
        tracker = ProvenanceTracker({}, path="/path/to/manifest.json")
        assert tracker.manifest_path() == "/path/to/manifest.json"


class TestProvenanceTrackerLoad:
    """Tests for ProvenanceTracker.load class method."""

    def test_loads_from_model_manifest_json(self, tmp_path):
        """Should load from model_manifest.json when it exists."""
        manifest_data = {"model_version": "loaded-v1", "model_family": "als"}
        manifest_file = tmp_path / "model_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

        tracker = ProvenanceTracker.load(str(tmp_path))
        assert tracker.available()
        assert tracker.manifest()["model_version"] == "loaded-v1"

    def test_loads_from_manifests_directory_fallback(self, tmp_path):
        """Should fall back to manifests directory when main file missing."""
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()

        manifest_data = {"model_version": "fallback-v1"}
        versioned_file = manifests_dir / "fallback-v1.json"
        with open(versioned_file, "w") as f:
            json.dump(manifest_data, f)

        tracker = ProvenanceTracker.load(str(tmp_path))
        assert tracker.available()
        assert tracker.manifest()["model_version"] == "fallback-v1"

    def test_loads_latest_from_manifests_directory(self, tmp_path):
        """Should load the latest (last sorted) manifest from directory."""
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()

        # Create multiple versioned manifests
        for version in ["v1", "v2", "v3"]:
            with open(manifests_dir / f"{version}.json", "w") as f:
                json.dump({"model_version": version}, f)

        tracker = ProvenanceTracker.load(str(tmp_path))
        # Should load v3 as it's last alphabetically
        assert tracker.manifest()["model_version"] == "v3"

    def test_returns_empty_tracker_when_no_manifests(self, tmp_path):
        """Should return empty tracker when no manifest files found."""
        tracker = ProvenanceTracker.load(str(tmp_path))
        assert not tracker.available()
        assert tracker.manifest() == {}
        assert tracker.manifest_path() is None

    def test_handles_custom_manifest_filename(self, tmp_path):
        """Should load from custom manifest filename."""
        manifest_data = {"model_version": "custom-v1"}
        custom_file = tmp_path / "custom_manifest.json"
        with open(custom_file, "w") as f:
            json.dump(manifest_data, f)

        tracker = ProvenanceTracker.load(str(tmp_path), manifest_filename="custom_manifest.json")
        assert tracker.manifest()["model_version"] == "custom-v1"

    def test_handles_corrupted_json(self, tmp_path):
        """Should handle corrupted JSON gracefully."""
        manifest_file = tmp_path / "model_manifest.json"
        manifest_file.write_text("{ invalid json }", encoding="utf-8")

        tracker = ProvenanceTracker.load(str(tmp_path))
        assert not tracker.available()

    def test_handles_file_read_error(self, tmp_path):
        """Should handle file read errors gracefully."""
        manifest_file = tmp_path / "model_manifest.json"
        manifest_file.write_text("{}", encoding="utf-8")

        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError("Permission denied")
            tracker = ProvenanceTracker.load(str(tmp_path))
            assert not tracker.available()


class TestProvenanceTrackerContextTags:
    """Tests for context_tags method."""

    def test_returns_all_context_tags(self):
        """Should return all expected context tags."""
        manifest = {
            "model_version": "als-v1",
            "model_family": "als",
            "manifest_sha256": "abc123",
            "pipeline": {
                "git_commit": "commit-sha",
                "git_branch": "main",
                "git_dirty": "0",
            },
            "data": {
                "dataset_version": "data-v1",
                "dataset_sha256": "data-sha",
            },
        }
        tracker = ProvenanceTracker(manifest)
        tags = tracker.context_tags()

        assert tags["model_version"] == "als-v1"
        assert tags["model_family"] == "als"
        assert tags["pipeline_commit"] == "commit-sha"
        assert tags["pipeline_branch"] == "main"
        assert tags["pipeline_dirty"] == "0"
        assert tags["data_version"] == "data-v1"
        assert tags["data_sha256"] == "data-sha"
        assert tags["manifest_sha256"] == "abc123"

    def test_returns_empty_dict_when_not_available(self):
        """Should return empty dict when manifest not available."""
        tracker = ProvenanceTracker()
        tags = tracker.context_tags()
        assert tags == {}

    def test_handles_missing_nested_fields(self):
        """Should handle missing nested fields gracefully."""
        manifest = {"model_version": "v1"}
        tracker = ProvenanceTracker(manifest)
        tags = tracker.context_tags()

        assert tags["model_version"] == "v1"
        assert tags["pipeline_commit"] is None
        assert tags["data_version"] is None


class TestProvenanceTrackerResponseHeaders:
    """Tests for response_headers method."""

    def test_returns_formatted_headers(self):
        """Should return properly formatted HTTP headers."""
        manifest = {
            "model_version": "als-v1",
            "manifest_sha256": "manifest-sha",
            "pipeline": {"git_commit": "commit-sha"},
            "data": {"dataset_version": "data-v1"},
        }
        tracker = ProvenanceTracker(manifest)
        headers = tracker.response_headers()

        assert headers["X-Model-Version"] == "als-v1"
        assert headers["X-Pipeline-Commit"] == "commit-sha"
        assert headers["X-Data-Version"] == "data-v1"
        assert headers["X-Model-Manifest"] == "manifest-sha"

    def test_returns_empty_strings_for_none_values(self):
        """Should use empty strings for None values."""
        manifest = {"model_version": "v1"}
        tracker = ProvenanceTracker(manifest)
        headers = tracker.response_headers()

        assert headers["X-Model-Version"] == "v1"
        assert headers["X-Pipeline-Commit"] == ""
        assert headers["X-Data-Version"] == ""

    def test_returns_empty_dict_when_not_available(self):
        """Should return empty dict when manifest not available."""
        tracker = ProvenanceTracker()
        headers = tracker.response_headers()
        assert headers == {}


class TestProvenanceTrackerExtraLogSuffix:
    """Tests for extra_log_suffix method."""

    def test_generates_comma_separated_suffix(self):
        """Should generate comma-separated key=value pairs."""
        manifest = {
            "model_version": "als-v1",
            "model_family": "als",
            "manifest_sha256": "sha256hash",
            "pipeline": {
                "git_commit": "abc123",
                "git_branch": "main",
                "git_dirty": "0",
            },
            "data": {
                "dataset_version": "data-v1",
                "dataset_sha256": "datahash",
            },
        }
        tracker = ProvenanceTracker(manifest)
        suffix = tracker.extra_log_suffix()

        assert "model_version=als-v1" in suffix
        assert "model_family=als" in suffix
        assert "pipeline_commit=abc123" in suffix
        assert ", " in suffix  # Comma-separated

    def test_returns_empty_string_when_not_available(self):
        """Should return empty string when manifest not available."""
        tracker = ProvenanceTracker()
        suffix = tracker.extra_log_suffix()
        assert suffix == ""

    def test_excludes_none_and_empty_values(self):
        """Should exclude None and empty string values from suffix."""
        manifest = {
            "model_version": "v1",
            "pipeline": {},
            "data": {},
        }
        tracker = ProvenanceTracker(manifest)
        suffix = tracker.extra_log_suffix()

        assert "model_version=v1" in suffix
        assert "pipeline_commit=" not in suffix
        assert "=None" not in suffix


# =============================================================================
# Integration Tests
# =============================================================================

class TestManifestRoundTrip:
    """Integration tests for the full manifest lifecycle."""

    def test_manifest_round_trip(self, tmp_path):
        """Building, persisting, and loading a manifest should retain key metadata."""
        data_file = tmp_path / "interactions.csv"
        data_file.write_text("user_id,movie_id,rating\n1,m1,5\n", encoding="utf-8")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        model_file = artifacts_dir / "als_model.npz"
        model_file.write_bytes(b"fake-model-bytes")
        user_map = artifacts_dir / "user_map.json"
        user_map.write_text("{}", encoding="utf-8")

        manifest = build_model_manifest(
            model_family="als",
            artifact_paths={
                "als_model": str(model_file),
                "user_map": str(user_map),
            },
            hyperparameters={"factors": 10},
            data_profile={"row_count": 1, "unique_users": 1, "unique_movies": 1},
            data_path=str(data_file),
            data_version="sample-v1",
            model_version="als-test",
        )

        persist_manifest(manifest, str(artifacts_dir))

        tracker = ProvenanceTracker.load(str(artifacts_dir))
        assert tracker.available()

        tags = tracker.context_tags()
        assert tags["model_version"] == "als-test"
        assert tags["data_version"] == "sample-v1"

        headers = tracker.response_headers()
        assert headers["X-Model-Version"] == "als-test"

        suffix = tracker.extra_log_suffix()
        assert "model_version=als-test" in suffix
        assert "data_version=sample-v1" in suffix

    def test_full_workflow_with_all_features(self, tmp_path):
        """Complete workflow testing all manifest and tracker features."""
        # Setup
        data_file = tmp_path / "train.csv"
        data_file.write_text("user,item,rating\n", encoding="utf-8")

        artifacts_dir = tmp_path / "models"
        artifacts_dir.mkdir()

        # Build manifest with extra metadata
        with patch("src.provenance.manifest.collect_git_metadata") as mock_git:
            mock_git.return_value = {
                "git_commit": "test-commit",
                "git_branch": "feature/test",
                "git_dirty": "0",
            }

            manifest = build_model_manifest(
                model_family="collaborative",
                artifact_paths={},
                hyperparameters={"learning_rate": 0.01},
                data_profile={"row_count": 100, "unique_users": 10, "unique_movies": 20},
                data_path=str(data_file),
                data_version="dataset-2024",
                model_version="collab-v1",
                extra_metadata={"experiment": "baseline"},
            )

        assert manifest["metadata"]["experiment"] == "baseline"

        # Persist
        persist_manifest(manifest, str(artifacts_dir))

        # Load and verify
        tracker = ProvenanceTracker.load(str(artifacts_dir))
        assert tracker.manifest_path() is not None

        full_manifest = tracker.manifest()
        assert full_manifest["pipeline"]["git_commit"] == "test-commit"
        assert full_manifest["hyperparameters"]["learning_rate"] == 0.01

        # Verify backward compatibility
        legacy_file = artifacts_dir / "provenance.json"
        assert legacy_file.exists()
