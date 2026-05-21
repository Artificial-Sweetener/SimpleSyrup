# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load GroundingDINO models and explicit text encoders."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from ..shared.logging import get_logger
from .bert_resolver import is_valid_bert_directory
from .loaded_models import LoadedGroundingDINOModel
from .model_catalog import BERT_ENTRY, ModelEntry, get_grounding_dino_entry
from .model_device_manager import TorchModelDeviceManager
from .model_downloads import DownloadRequest, ModelDownloader, ProgressReporter
from .model_folders import (
    expected_model_file,
    get_primary_model_folder,
    register_required_model_folders,
    resolve_model_file,
)
from .model_instance_cache import ModelInstanceCache

LOGGER = get_logger(__name__)
GROUNDING_DINO_RUNTIME_PACKAGE = "simple_syrup.third_party.groundingdino_runtime"
TEXT_ENCODER_LAYERSTYLE = "LayerStyle bert-base-uncased"
TEXT_ENCODER_COMFY = "text_encoders/bert"
TEXT_ENCODER_AUTO = "BERT base uncased (auto)"
TEXT_ENCODER_CHOICES = (
    TEXT_ENCODER_LAYERSTYLE,
    TEXT_ENCODER_COMFY,
    TEXT_ENCODER_AUTO,
)


@dataclass(frozen=True)
class TextEncoderResolution:
    """Resolved text encoder directory and selection metadata."""

    path: Path
    source: str
    downloaded: bool


@dataclass(frozen=True)
class GroundingDINOModelCacheKey:
    """Identify a loaded GroundingDINO model for process-level reuse."""

    model_id: str
    config_path: Path
    checkpoint_path: Path
    text_encoder_path: Path


_LOADED_GROUNDING_DINO_MODELS: dict[
    GroundingDINOModelCacheKey, LoadedGroundingDINOModel
] = {}


class GroundingDINOLoaderService:
    """Resolve, download, and load GroundingDINO with an explicit text encoder."""

    def __init__(
        self,
        downloader: ModelDownloader | None = None,
        folder_paths_module: ModuleType | None = None,
        device_manager: TorchModelDeviceManager | None = None,
        cache: (
            MutableMapping[GroundingDINOModelCacheKey, LoadedGroundingDINOModel] | None
        ) = None,
    ) -> None:
        """Create a GroundingDINO loader with injectable external boundaries."""

        self._downloader = downloader or ModelDownloader()
        self._folder_paths_module = folder_paths_module
        self._device_manager = device_manager or TorchModelDeviceManager()
        self._cache: ModelInstanceCache[
            GroundingDINOModelCacheKey, LoadedGroundingDINOModel
        ] = ModelInstanceCache(
            cache if cache is not None else _LOADED_GROUNDING_DINO_MODELS
        )

    def load_model(
        self,
        grounding_dino_model: str,
        text_encoder: str,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> LoadedGroundingDINOModel:
        """Load GroundingDINO and return a `GROUNDING_DINO_MODEL` object."""

        register_required_model_folders(self._folder_paths_module)
        entry = get_grounding_dino_entry(grounding_dino_model)
        artifact_paths = self._resolve_artifacts(entry, auto_download, progress)
        text_encoder_resolution = self.resolve_text_encoder(
            text_encoder,
            auto_download,
            progress,
        )
        config_path = _artifact_path_with_suffix(artifact_paths, ".py")
        checkpoint_path = _artifact_path_with_suffix(artifact_paths, ".pth")
        key = GroundingDINOModelCacheKey(
            model_id=entry.entry_id,
            config_path=config_path.resolve(),
            checkpoint_path=checkpoint_path.resolve(),
            text_encoder_path=text_encoder_resolution.path.resolve(),
        )
        already_loaded = key in self._cache.entries
        loaded = self._cache.get_or_load(
            key,
            lambda: self._load_uncached_model(
                entry,
                config_path,
                checkpoint_path,
                text_encoder_resolution,
            ),
        )
        if already_loaded:
            LOGGER.info(
                "GroundingDINO model loaded from process cache",
                extra={
                    "operation": "grounding_dino_loader",
                    "model": entry.entry_id,
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "text_encoder_path": str(text_encoder_resolution.path),
                },
            )
        return loaded

    def _load_uncached_model(
        self,
        entry: ModelEntry,
        config_path: Path,
        checkpoint_path: Path,
        text_encoder_resolution: TextEncoderResolution,
    ) -> LoadedGroundingDINOModel:
        """Load and wrap GroundingDINO after resolution and cache lookup."""

        model = self._load_grounding_dino_model(
            entry,
            config_path,
            checkpoint_path,
            text_encoder_resolution.path,
        )
        managed_model = self._device_manager.manage(
            model,
            model_id=entry.entry_id,
            source=str(checkpoint_path),
        )
        loaded = LoadedGroundingDINOModel(
            model=model,
            text_encoder_path=text_encoder_resolution.path,
            source=text_encoder_resolution.source,
            model_id=entry.entry_id,
            managed_model=managed_model,
        )
        LOGGER.info(
            "GroundingDINO model loaded",
            extra={
                "operation": "grounding_dino_loader",
                "model": entry.entry_id,
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "text_encoder_path": str(text_encoder_resolution.path),
            },
        )
        return loaded

    def resolve_text_encoder(
        self,
        text_encoder: str,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> TextEncoderResolution:
        """Resolve the selected BERT text encoder mode."""

        layerstyle_path = self._models_dir() / "bert-base-uncased"
        comfy_path = self._text_encoder_bert_path()

        if text_encoder == TEXT_ENCODER_LAYERSTYLE:
            return _require_bert_directory(layerstyle_path, text_encoder)
        if text_encoder == TEXT_ENCODER_COMFY:
            return _require_bert_directory(comfy_path, text_encoder)
        if text_encoder != TEXT_ENCODER_AUTO:
            valid = ", ".join(TEXT_ENCODER_CHOICES)
            raise ValueError(f"text_encoder must be one of: {valid}.")

        if is_valid_bert_directory(layerstyle_path):
            return TextEncoderResolution(
                path=layerstyle_path,
                source="models/bert-base-uncased",
                downloaded=False,
            )
        if is_valid_bert_directory(comfy_path):
            return TextEncoderResolution(
                path=comfy_path,
                source="models/text_encoders/bert",
                downloaded=False,
            )
        if not auto_download:
            raise FileNotFoundError(
                "BERT text encoder was not found. Checked: "
                f"{layerstyle_path}; {comfy_path}. Enable auto_download on "
                "GroundingDINO Model Loader or install a Hugging Face "
                "bert-base-uncased snapshot."
            )
        self._download_bert(comfy_path, progress)
        if not is_valid_bert_directory(comfy_path):
            raise FileNotFoundError(
                f"Downloaded BERT files in '{comfy_path}' are incomplete."
            )
        return TextEncoderResolution(
            path=comfy_path,
            source="downloaded: google-bert/bert-base-uncased",
            downloaded=True,
        )

    def _resolve_artifacts(
        self,
        entry: ModelEntry,
        auto_download: bool,
        progress: ProgressReporter | None,
    ) -> dict[str, Path]:
        """Resolve or download GroundingDINO catalog artifacts."""

        artifact_paths: dict[str, Path] = {}
        for artifact in entry.artifacts:
            existing = resolve_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if existing is not None:
                artifact_paths[artifact.artifact_id] = existing
                continue

            destination = expected_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if not auto_download or not entry.auto_download_allowed:
                raise FileNotFoundError(
                    f"GroundingDINO model '{entry.display_name}' is missing and "
                    f"auto_download is disabled. Expected: {destination}. "
                    "Enable auto_download on GroundingDINO Model Loader or install "
                    "the model artifact."
                )
            result = self._downloader.download(
                DownloadRequest(
                    source_url=artifact.source_url,
                    destination_path=destination,
                    expected_folder=destination.parent,
                    description=artifact.description,
                ),
                progress,
            )
            artifact_paths[artifact.artifact_id] = result.path
        return artifact_paths

    def _download_bert(
        self,
        target_directory: Path,
        progress: ProgressReporter | None,
    ) -> None:
        """Download the known BERT artifact set into the selected directory."""

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

    def _load_grounding_dino_model(
        self,
        entry: ModelEntry,
        config_path: Path,
        checkpoint_path: Path,
        text_encoder_path: Path,
    ) -> object:
        """Load GroundingDINO from a known config/checkpoint pair."""

        importlib.invalidate_caches()
        try:
            slconfig_module = importlib.import_module(
                f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.slconfig"
            )
            utils_module = importlib.import_module(
                f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.utils"
            )
            models_module = importlib.import_module(
                f"{GROUNDING_DINO_RUNTIME_PACKAGE}.models"
            )
        except ImportError as error:
            raise RuntimeError(
                "SimpleSyrup's GroundingDINO runtime is unavailable. Reinstall "
                "SimpleSyrup or restore "
                "simple_syrup.third_party.groundingdino_runtime. "
                f"Import failed: {error}."
            ) from error

        args = slconfig_module.SLConfig.fromfile(str(config_path))
        if getattr(args, "text_encoder_type", "") == "bert-base-uncased":
            args.text_encoder_type = str(text_encoder_path)

        model = models_module.build_model(args)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        clean_state_dict = utils_module.clean_state_dict
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.model_name = entry.entry_id
        return model

    def _text_encoder_bert_path(self) -> Path:
        """Return SimpleSyrup's BERT path under ComfyUI text encoders."""

        return (
            get_primary_model_folder("text_encoders", self._folder_paths_module)
            / "bert"
        )

    def _models_dir(self) -> Path:
        """Return ComfyUI's models directory."""

        import importlib as _importlib

        folder_paths = self._folder_paths_module or _importlib.import_module(
            "folder_paths"
        )
        return Path(str(folder_paths.models_dir))


def _require_bert_directory(path: Path, label: str) -> TextEncoderResolution:
    """Return an explicit BERT directory or fail with an actionable message."""

    if not is_valid_bert_directory(path):
        raise FileNotFoundError(
            f"Text encoder '{label}' is incomplete. Expected files under {path}: "
            "config.json, tokenizer.json or vocab.txt, and model.safetensors or "
            "pytorch_model.bin."
        )
    return TextEncoderResolution(path=path, source=label, downloaded=False)


def _artifact_path_with_suffix(artifact_paths: dict[str, Path], suffix: str) -> Path:
    """Return the first artifact path with a suffix."""

    for path in artifact_paths.values():
        if path.suffix.lower() == suffix:
            return path
    raise FileNotFoundError(f"GroundingDINO model does not have a '{suffix}' artifact.")
