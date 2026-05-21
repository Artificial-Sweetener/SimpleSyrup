# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for model artifact downloads and progress reporting."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import pytest

from simple_syrup.runtime.model_downloads import DownloadRequest, ModelDownloader


@dataclass
class RecordingProgress:
    """Progress reporter double for download tests."""

    starts: list[tuple[str, int | None]] = field(default_factory=list)
    advances: list[tuple[int, int | None]] = field(default_factory=list)
    finishes: int = 0

    def start(self, label: str, total: int | None) -> None:
        """Record progress start."""

        self.starts.append((label, total))

    def advance(self, current: int, total: int | None) -> None:
        """Record progress update."""

        self.advances.append((current, total))

    def finish(self) -> None:
        """Record progress finish."""

        self.finishes += 1


class FakeResponse:
    """Small streaming response double."""

    def __init__(self, chunks: list[bytes], total: int | None = None) -> None:
        """Create a response from byte chunks."""

        self._chunks = chunks
        self.headers = {}
        if total is not None:
            self.headers["Content-Length"] = str(total)

    def __enter__(self) -> FakeResponse:
        """Enter response context."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit response context."""

    def read(self, _size: int) -> bytes:
        """Return the next chunk."""

        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_downloader_streams_file_and_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloader writes to destination and reports byte progress."""

    content = [b"abc", b"def"]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _url, timeout: FakeResponse(content, total=6),
    )
    progress = RecordingProgress()
    destination = tmp_path / "models" / "sams" / "model.pth"

    result = ModelDownloader().download(
        DownloadRequest(
            source_url="https://example.invalid/model.pth",
            destination_path=destination,
            expected_folder=tmp_path / "models" / "sams",
            description="test model",
        ),
        progress,
    )

    assert destination.read_bytes() == b"abcdef"
    assert result.bytes_downloaded == 6
    assert progress.starts == [("Downloading test model", 6)]
    assert progress.advances[-1] == (6, 6)
    assert progress.finishes == 1


def test_downloader_verifies_expected_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloader verifies the final artifact checksum before publishing."""

    content = [b"abc", b"def"]
    expected_sha256 = hashlib.sha256(b"abcdef").hexdigest()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _url, timeout: FakeResponse(content, total=6),
    )
    destination = tmp_path / "models" / "sams" / "model.pth"

    result = ModelDownloader().download(
        DownloadRequest(
            source_url="https://example.invalid/model.pth",
            destination_path=destination,
            expected_folder=destination.parent,
            description="test model",
            expected_sha256=expected_sha256,
        )
    )

    assert result.path == destination
    assert destination.read_bytes() == b"abcdef"


def test_downloader_removes_partial_file_on_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checksum failures do not leave incomplete download files behind."""

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _url, timeout: FakeResponse([b"wrong"], total=5),
    )
    destination = tmp_path / "models" / "sams" / "model.pth"

    with pytest.raises(ValueError, match="checksum mismatch"):
        ModelDownloader().download(
            DownloadRequest(
                source_url="https://example.invalid/model.pth",
                destination_path=destination,
                expected_folder=destination.parent,
                description="test model",
                expected_sha256=hashlib.sha256(b"expected").hexdigest(),
            )
        )

    assert not destination.exists()
    assert not destination.with_name("model.pth.part").exists()


def test_downloader_skips_existing_file(tmp_path: Path) -> None:
    """Existing complete files are not downloaded again."""

    destination = tmp_path / "models" / "sams" / "model.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    result = ModelDownloader().download(
        DownloadRequest(
            source_url="https://example.invalid/model.pth",
            destination_path=destination,
            expected_folder=destination.parent,
            description="test model",
        )
    )

    assert result.skipped_existing is True
    assert result.bytes_downloaded == 0


def test_downloader_removes_partial_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed downloads clean up their .part file."""

    def fail_urlopen(_url: str, timeout: int) -> FakeResponse:
        raise OSError("network failed")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    destination = tmp_path / "models" / "sams" / "model.pth"

    with pytest.raises(OSError, match="network failed"):
        ModelDownloader().download(
            DownloadRequest(
                source_url="https://example.invalid/model.pth",
                destination_path=destination,
                expected_folder=destination.parent,
                description="test model",
            )
        )

    assert not destination.with_name("model.pth.part").exists()


def test_downloader_rejects_destination_outside_expected_folder(
    tmp_path: Path,
) -> None:
    """Downloader fails closed when destination escapes the model folder."""

    with pytest.raises(ValueError, match="outside"):
        ModelDownloader().download(
            DownloadRequest(
                source_url="https://example.invalid/model.pth",
                destination_path=tmp_path / "elsewhere" / "model.pth",
                expected_folder=tmp_path / "models",
                description="test model",
            )
        )
