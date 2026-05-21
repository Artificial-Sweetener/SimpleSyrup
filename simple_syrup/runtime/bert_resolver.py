# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the BERT text encoder used by GroundingDINO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .model_catalog import BERT_ENTRY
from .model_downloads import (
    DownloadRequest,
    ModelDownloader,
    ProgressReporter,
)
from .model_folders import get_primary_model_folder


@dataclass(frozen=True)
class BertResolution:
    """Resolved local BERT directory and how it was obtained."""

    path: Path
    source: str
    downloaded: bool


class BertResolver:
    """Find or download a Hugging Face-style BERT directory."""

    def __init__(
        self,
        downloader: ModelDownloader | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a resolver with injectable runtime dependencies."""

        self._downloader = downloader or ModelDownloader()
        self._folder_paths_module = folder_paths_module

    def resolve(
        self,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> BertResolution:
        """Return a usable BERT directory or raise an actionable error."""

        layerstyle_path = self._models_dir() / "bert-base-uncased"
        if is_valid_bert_directory(layerstyle_path):
            return BertResolution(
                path=layerstyle_path,
                source="models/bert-base-uncased",
                downloaded=False,
            )

        text_encoder_path = self._text_encoder_bert_path()
        if is_valid_bert_directory(text_encoder_path):
            return BertResolution(
                path=text_encoder_path,
                source="models/text_encoders/bert",
                downloaded=False,
            )

        if not auto_download:
            raise FileNotFoundError(
                "BERT text encoder was not found. Checked: "
                f"{layerstyle_path}; {text_encoder_path}. "
                "Enable auto_download or install a Hugging Face "
                "bert-base-uncased snapshot."
            )

        self._download_bert(text_encoder_path, progress)
        if not is_valid_bert_directory(text_encoder_path):
            raise FileNotFoundError(
                f"Downloaded BERT files in '{text_encoder_path}' are incomplete."
            )
        return BertResolution(
            path=text_encoder_path,
            source="downloaded: google-bert/bert-base-uncased",
            downloaded=True,
        )

    def _download_bert(
        self,
        target_directory: Path,
        progress: ProgressReporter | None,
    ) -> None:
        """Download the known BERT artifact set into the target directory."""

        for artifact in BERT_ENTRY.artifacts:
            destination = target_directory / artifact.filename
            self._downloader.download(
                DownloadRequest(
                    source_url=artifact.source_url,
                    destination_path=destination,
                    expected_folder=target_directory,
                    description=artifact.description,
                ),
                progress,
            )

    def _text_encoder_bert_path(self) -> Path:
        """Return SimpleSyrup's BERT path under ComfyUI text encoders."""

        return (
            get_primary_model_folder("text_encoders", self._folder_paths_module)
            / "bert"
        )

    def _models_dir(self) -> Path:
        """Return ComfyUI's models directory."""

        import importlib

        folder_paths = self._folder_paths_module or importlib.import_module(
            "folder_paths"
        )
        return Path(str(folder_paths.models_dir))


def is_valid_bert_directory(path: Path) -> bool:
    """Return whether a directory has enough files to load BERT locally."""

    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_tokenizer = (path / "tokenizer.json").is_file() or (
        path / "vocab.txt"
    ).is_file()
    has_weights = (path / "model.safetensors").is_file() or (
        path / "pytorch_model.bin"
    ).is_file()
    return has_config and has_tokenizer and has_weights
