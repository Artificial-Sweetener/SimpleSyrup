# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for BERT text encoder resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_syrup.runtime.bert_resolver import BertResolver, is_valid_bert_directory
from simple_syrup.runtime.model_downloads import DownloadRequest, DownloadResult
from test_helpers import FakeFolderPaths


class RecordingDownloader:
    """Downloader double that writes requested artifact files."""

    def __init__(self) -> None:
        """Create an empty recording downloader."""

        self.requests: list[DownloadRequest] = []

    def download(
        self,
        request: DownloadRequest,
        progress: object | None = None,
    ) -> DownloadResult:
        """Record and satisfy one download request."""

        self.requests.append(request)
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        request.destination_path.write_bytes(b"artifact")
        return DownloadResult(
            path=request.destination_path,
            bytes_downloaded=8,
            skipped_existing=False,
        )


def test_valid_bert_directory_requires_config_tokenizer_and_weights(
    tmp_path: Path,
) -> None:
    """BERT validation checks the minimum Hugging Face directory shape."""

    bert = tmp_path / "bert"
    bert.mkdir()
    assert not is_valid_bert_directory(bert)

    (bert / "config.json").write_text("{}", encoding="utf-8")
    (bert / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bert / "model.safetensors").write_bytes(b"weights")

    assert is_valid_bert_directory(bert)


def test_resolver_prefers_layerstyle_bert_path(tmp_path: Path) -> None:
    """LayerStyle-compatible BERT wins when present."""

    fake = FakeFolderPaths(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")
    _write_bert(tmp_path / "text_encoders" / "bert")

    resolved = BertResolver(folder_paths_module=fake).resolve(auto_download=False)

    assert resolved.path == tmp_path / "bert-base-uncased"
    assert resolved.downloaded is False


def test_resolver_uses_text_encoder_bert_when_layerstyle_path_absent(
    tmp_path: Path,
) -> None:
    """ComfyUI text_encoders BERT is used as the second local option."""

    fake = FakeFolderPaths(tmp_path)
    _write_bert(tmp_path / "text_encoders" / "bert")

    resolved = BertResolver(folder_paths_module=fake).resolve(auto_download=False)

    assert resolved.path == tmp_path / "text_encoders" / "bert"


def test_resolver_downloads_to_text_encoder_bert(tmp_path: Path) -> None:
    """Missing BERT downloads into models/text_encoders/bert."""

    fake = FakeFolderPaths(tmp_path)
    downloader = RecordingDownloader()

    resolved = BertResolver(
        downloader=downloader,  # type: ignore[arg-type]
        folder_paths_module=fake,
    ).resolve(auto_download=True)

    assert resolved.path == tmp_path / "text_encoders" / "bert"
    assert resolved.downloaded is True
    assert downloader.requests


def test_resolver_errors_when_missing_and_download_disabled(tmp_path: Path) -> None:
    """Missing BERT with downloads disabled fails clearly."""

    fake = FakeFolderPaths(tmp_path)

    with pytest.raises(FileNotFoundError, match="BERT text encoder was not found"):
        BertResolver(folder_paths_module=fake).resolve(auto_download=False)


def _write_bert(path: Path) -> None:
    """Write a minimal valid BERT directory."""

    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
