# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for automatic model artifact resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_syrup.runtime.auto_model_cache import AutoModelCache, AutoModelCacheEntry
from simple_syrup.runtime.auto_model_resolver import (
    AutoModelResolver,
    canonical_auto_destination,
    find_model_by_basename,
    relative_model_name,
)
from simple_syrup.runtime.model_catalog import AutoModelArtifact
from simple_syrup.runtime.model_downloads import DownloadRequest, DownloadResult
from test_helpers import FakeFolderPaths


class RecordingDownloader:
    """Downloader double that records requests and writes final files."""

    def __init__(self, fail: bool = False) -> None:
        """Create a downloader with optional failure behavior."""

        self.fail = fail
        self.requests: list[DownloadRequest] = []

    def download(
        self,
        request: DownloadRequest,
        progress: object | None = None,
    ) -> DownloadResult:
        """Record and satisfy one trusted download request."""

        del progress
        self.requests.append(request)
        if self.fail:
            raise ValueError("checksum mismatch")
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        request.destination_path.write_bytes(b"model")
        return DownloadResult(
            path=request.destination_path,
            bytes_downloaded=5,
            skipped_existing=False,
        )


def test_resolver_returns_valid_cached_path(tmp_path: Path) -> None:
    """A valid remembered path is reused without search or download."""

    fake = FakeFolderPaths(tmp_path / "models")
    artifact = _artifact("text_encoders", "model.safetensors")
    cached_path = tmp_path / "models" / "text_encoders" / "qwen" / artifact.filename
    cached_path.parent.mkdir(parents=True)
    cached_path.write_bytes(b"model")
    cache = AutoModelCache(fake)
    cache.save_entry(
        artifact.cache_id,
        AutoModelCacheEntry(
            folder_name=artifact.folder_name,
            filename=artifact.filename,
            path=cached_path,
            source="downloaded",
            sha256=artifact.sha256,
        ),
    )
    downloader = RecordingDownloader()

    resolved = AutoModelResolver(cache, downloader, fake).resolve(artifact)

    assert resolved.path == cached_path
    assert resolved.source == "cached"
    assert downloader.requests == []


def test_resolver_repairs_stale_cache_with_recursive_search(tmp_path: Path) -> None:
    """Missing cached files trigger recursive search and cache update."""

    fake = FakeFolderPaths(tmp_path / "models")
    artifact = _artifact("text_encoders", "model.safetensors")
    found_path = tmp_path / "models" / "text_encoders" / "nested" / artifact.filename
    found_path.parent.mkdir(parents=True)
    found_path.write_bytes(b"model")
    cache = AutoModelCache(fake)
    cache.save_entry(
        artifact.cache_id,
        AutoModelCacheEntry(
            folder_name=artifact.folder_name,
            filename=artifact.filename,
            path=tmp_path / "missing.safetensors",
            source="downloaded",
            sha256=artifact.sha256,
        ),
    )

    resolved = AutoModelResolver(cache, RecordingDownloader(), fake).resolve(artifact)

    assert resolved.path == found_path
    assert resolved.source == "found"
    assert cache.load()[artifact.cache_id].path == found_path


def test_find_model_by_basename_respects_folder_priority(tmp_path: Path) -> None:
    """Recursive search prefers earlier ComfyUI model roots."""

    fake = FakeFolderPaths(tmp_path / "models")
    first = tmp_path / "external" / "text_encoders"
    second = tmp_path / "models" / "text_encoders"
    fake.folder_names_and_paths["text_encoders"] = ([str(first), str(second)], set())
    (first / "a").mkdir(parents=True)
    (second / "b").mkdir(parents=True)
    first_match = first / "a" / "model.safetensors"
    second_match = second / "b" / "model.safetensors"
    first_match.write_bytes(b"first")
    second_match.write_bytes(b"second")

    assert (
        find_model_by_basename("text_encoders", "model.safetensors", fake)
        == first_match
    )


def test_resolver_downloads_to_first_registered_folder(tmp_path: Path) -> None:
    """Missing artifacts download to the canonical subfolder under the first root."""

    fake = FakeFolderPaths(tmp_path / "models")
    first = tmp_path / "external" / "vae"
    second = tmp_path / "models" / "vae"
    fake.folder_names_and_paths["vae"] = ([str(first), str(second)], set())
    artifact = _artifact("vae", "vae.safetensors")
    cache = AutoModelCache(fake)
    downloader = RecordingDownloader()

    resolved = AutoModelResolver(cache, downloader, fake).resolve(artifact)

    expected = first / "qwen" / "vae.safetensors"
    assert resolved.path == expected
    assert downloader.requests[0].destination_path == expected
    assert downloader.requests[0].expected_folder == first
    assert downloader.requests[0].expected_sha256 == artifact.sha256
    assert cache.load()[artifact.cache_id].source == "downloaded"


def test_canonical_destination_rejects_unsafe_subfolder(tmp_path: Path) -> None:
    """Catalog paths cannot escape the model root."""

    fake = FakeFolderPaths(tmp_path / "models")
    artifact = AutoModelArtifact(
        cache_id="bad",
        filename="model.safetensors",
        folder_name="vae",
        canonical_subfolder="..",
        source_url="https://example.invalid/model.safetensors",
        source_repo="example/model",
        description="bad",
        sha256="abc",
    )

    with pytest.raises(ValueError, match="not safe"):
        canonical_auto_destination(artifact, fake)


@pytest.mark.parametrize(
    "unsafe_basename",
    ("nested/model.safetensors", "nested\\model.safetensors"),
)
def test_find_model_by_basename_rejects_relative_paths(
    tmp_path: Path,
    unsafe_basename: str,
) -> None:
    """Search only accepts basenames, not relative paths."""

    with pytest.raises(ValueError, match="not safe"):
        find_model_by_basename(
            "vae",
            unsafe_basename,
            FakeFolderPaths(tmp_path / "models"),
        )


def test_relative_model_name_returns_comfy_relative_path(tmp_path: Path) -> None:
    """Resolved paths can be converted back to ComfyUI relative names."""

    fake = FakeFolderPaths(tmp_path / "models")
    path = tmp_path / "models" / "vae" / "qwen" / "vae.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")

    assert relative_model_name("vae", path, fake) == str(
        Path("qwen") / "vae.safetensors"
    )


def test_resolver_does_not_cache_failed_download(tmp_path: Path) -> None:
    """Failed downloads leave the auto cache unchanged."""

    fake = FakeFolderPaths(tmp_path / "models")
    artifact = _artifact("vae", "vae.safetensors")
    cache = AutoModelCache(fake)

    with pytest.raises(ValueError, match="checksum mismatch"):
        AutoModelResolver(cache, RecordingDownloader(fail=True), fake).resolve(artifact)

    assert artifact.cache_id not in cache.load()


def _artifact(folder_name: str, filename: str) -> AutoModelArtifact:
    """Create a trusted artifact fixture."""

    return AutoModelArtifact(
        cache_id=f"{folder_name}_{filename}",
        filename=filename,
        folder_name=folder_name,
        canonical_subfolder="qwen",
        source_url=f"https://example.invalid/{filename}",
        source_repo="example/model",
        description=f"test {filename}",
        sha256="abc123",
    )
