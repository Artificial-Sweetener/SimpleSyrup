# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download known model artifacts with ComfyUI progress reporting."""

from __future__ import annotations

import hashlib
import importlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
CHUNK_SIZE = 1024 * 1024


class ProgressReporter(Protocol):
    """Report artifact download progress."""

    def start(self, label: str, total: int | None) -> None:
        """Start reporting one artifact download."""

    def advance(self, current: int, total: int | None) -> None:
        """Report absolute bytes downloaded."""

    def finish(self) -> None:
        """Report completion."""


class _ComfyProgressBar(Protocol):
    """Small protocol for the ComfyUI progress bar API used here."""

    def update_absolute(self, value: int, total: int | None = None) -> None:
        """Update absolute progress."""


class NullProgressReporter:
    """Ignore progress updates."""

    def start(self, label: str, total: int | None) -> None:
        """Ignore download start."""

    def advance(self, current: int, total: int | None) -> None:
        """Ignore download progress."""

    def finish(self) -> None:
        """Ignore download completion."""


class ComfyProgressReporter:
    """Report download progress through ComfyUI's node progress bar."""

    def __init__(self) -> None:
        """Initialize an empty ComfyUI progress reporter."""

        self._progress_bar: object | None = None
        self._total = 1

    def start(self, label: str, total: int | None) -> None:
        """Create a ComfyUI progress bar for one artifact."""

        comfy_utils = importlib.import_module("comfy.utils")
        progress_bar_class = comfy_utils.ProgressBar
        self._total = total if total and total > 0 else 1
        self._progress_bar = progress_bar_class(self._total)
        self.advance(0, total)
        LOGGER.info("download progress started", extra={"label": label, "total": total})

    def advance(self, current: int, total: int | None) -> None:
        """Update the ComfyUI progress bar."""

        if self._progress_bar is None:
            return
        if total and total > 0 and total != self._total:
            self._total = total
        value = (
            current if total and total > 0 else min(current // CHUNK_SIZE, self._total)
        )
        progress_bar = cast(_ComfyProgressBar, self._progress_bar)
        progress_bar.update_absolute(value, self._total)

    def finish(self) -> None:
        """Mark the current ComfyUI progress bar complete."""

        if self._progress_bar is None:
            return
        progress_bar = cast(_ComfyProgressBar, self._progress_bar)
        progress_bar.update_absolute(self._total, self._total)


@dataclass(frozen=True)
class DownloadRequest:
    """A trusted catalog download request."""

    source_url: str
    destination_path: Path
    expected_folder: Path
    description: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    """The result of resolving or downloading an artifact."""

    path: Path
    bytes_downloaded: int
    skipped_existing: bool


class ModelDownloader:
    """Download trusted model artifacts safely into model folders."""

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Download an artifact unless the final file already exists."""

        reporter = progress or NullProgressReporter()
        destination = request.destination_path
        self._validate_destination(destination, request.expected_folder)

        if destination.is_file():
            return DownloadResult(
                path=destination,
                bytes_downloaded=0,
                skipped_existing=True,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_name(f"{destination.name}.part")

        try:
            LOGGER.info(
                "downloading model artifact",
                extra={
                    "url": request.source_url,
                    "destination": str(destination),
                    "description": request.description,
                },
            )
            bytes_downloaded = self._download_to_temporary(
                source_url=request.source_url,
                temporary_path=temporary_path,
                reporter=reporter,
                description=request.description,
            )
            if request.expected_sha256 is not None:
                actual_sha256 = _sha256_file(temporary_path)
                if actual_sha256.lower() != request.expected_sha256.lower():
                    raise ValueError(
                        "Downloaded model artifact checksum mismatch for "
                        f"'{destination}'. Expected {request.expected_sha256}, "
                        f"got {actual_sha256}."
                    )
            temporary_path.replace(destination)
            reporter.finish()
            return DownloadResult(
                path=destination,
                bytes_downloaded=bytes_downloaded,
                skipped_existing=False,
            )
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            LOGGER.exception(
                "model artifact download failed",
                extra={
                    "url": request.source_url,
                    "destination": str(destination),
                    "description": request.description,
                },
            )
            raise

    def _download_to_temporary(
        self,
        source_url: str,
        temporary_path: Path,
        reporter: ProgressReporter,
        description: str,
    ) -> int:
        """Stream a URL to a temporary file."""

        with urllib.request.urlopen(source_url, timeout=30) as response:
            total = _content_length(response)
            reporter.start(f"Downloading {description}", total)
            bytes_downloaded = 0
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    bytes_downloaded += len(chunk)
                    reporter.advance(bytes_downloaded, total)
        return bytes_downloaded

    def _validate_destination(self, destination: Path, expected_folder: Path) -> None:
        """Reject destinations outside the intended model folder."""

        destination_parent = destination.parent.resolve()
        expected_root = expected_folder.resolve()
        try:
            destination_parent.relative_to(expected_root)
        except ValueError as error:
            raise ValueError(
                f"Download destination '{destination}' is outside '{expected_folder}'."
            ) from error


def _content_length(response: BinaryIO) -> int | None:
    """Return response content length when the server supplies it."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
