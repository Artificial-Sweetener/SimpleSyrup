# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build settings-aware model dropdown choices for loader nodes."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol

from .model_catalog import (
    GROUNDING_DINO_ENTRIES,
    SAM_ENTRIES,
    VITMATTE_ENTRIES,
    WD14_TAGGER_ENTRIES,
    ModelEntry,
    grounding_dino_choices,
    sam_choices,
    vitmatte_choices,
    wd14_tagger_choices,
)
from .model_folders import resolve_model_file
from .settings import SimpleSyrupSettings, SimpleSyrupSettingsRepository
from .vitmatte_loader import ViTMatteLoaderService

NO_LOCAL_SAM_MODELS = "No local SAM models found"
NO_LOCAL_GROUNDING_DINO_MODELS = "No local GroundingDINO models found"
NO_LOCAL_VITMATTE_MODELS = "No local ViTMatte models found"
NO_LOCAL_WD14_TAGGER_MODELS = "No local WD14 tagger models found"


class SettingsProvider(Protocol):
    """Settings dependency used by model choice policy."""

    def load(self) -> SimpleSyrupSettings:
        """Return current SimpleSyrup settings."""


class ModelChoiceService:
    """Build model loader dropdown choices from settings and local availability."""

    def __init__(
        self,
        settings_repository: SettingsProvider | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create the choice service with injectable external boundaries."""

        self._settings_repository = (
            settings_repository or SimpleSyrupSettingsRepository()
        )
        self._folder_paths_module = folder_paths_module
        self._vitmatte_loader = ViTMatteLoaderService(
            folder_paths_module=folder_paths_module
        )

    def sam_choices(self) -> list[str]:
        """Return settings-aware SAM dropdown choices."""

        if self._show_downloadable_models():
            return sam_choices()

        choices = [
            entry.display_name
            for entry in SAM_ENTRIES
            if self._entry_artifacts_are_local(entry)
        ]
        return choices or [NO_LOCAL_SAM_MODELS]

    def grounding_dino_choices(self) -> list[str]:
        """Return settings-aware GroundingDINO dropdown choices."""

        if self._show_downloadable_models():
            return grounding_dino_choices()

        choices = [
            entry.display_name
            for entry in GROUNDING_DINO_ENTRIES
            if self._entry_artifacts_are_local(entry)
        ]
        return choices or [NO_LOCAL_GROUNDING_DINO_MODELS]

    def vitmatte_choices(self) -> list[str]:
        """Return settings-aware ViTMatte dropdown choices."""

        if self._show_downloadable_models():
            return vitmatte_choices()

        choices = [
            entry.display_name
            for entry in VITMATTE_ENTRIES
            if self._vitmatte_entry_is_local(entry)
        ]
        return choices or [NO_LOCAL_VITMATTE_MODELS]

    def wd14_tagger_choices(self) -> list[str]:
        """Return settings-aware WD14 tagger dropdown choices."""

        if self._show_downloadable_models():
            return wd14_tagger_choices()

        choices = [
            entry.display_name
            for entry in WD14_TAGGER_ENTRIES
            if self._entry_artifacts_are_local(entry)
        ]
        return choices or [NO_LOCAL_WD14_TAGGER_MODELS]

    def reject_sentinel(self, selection: str) -> None:
        """Reject placeholder dropdown selections before loader work begins."""

        if selection == NO_LOCAL_SAM_MODELS:
            raise ValueError(
                "No local SAM models are available. Enable 'Show downloadable "
                "models in loader dropdowns' in SimpleSyrup settings or install "
                "a SAM model in the sams folder."
            )
        if selection == NO_LOCAL_GROUNDING_DINO_MODELS:
            raise ValueError(
                "No local GroundingDINO models are available. Enable 'Show "
                "downloadable models in loader dropdowns' in SimpleSyrup "
                "settings or install a complete GroundingDINO model in the "
                "grounding-dino folder."
            )
        if selection == NO_LOCAL_VITMATTE_MODELS:
            raise ValueError(
                "No local ViTMatte models are available. Enable 'Show "
                "downloadable models in loader dropdowns' in SimpleSyrup "
                "settings or install a ViTMatte model in the vitmatte folder."
            )
        if selection == NO_LOCAL_WD14_TAGGER_MODELS:
            raise ValueError(
                "No local WD14 tagger models are available. Enable 'Show "
                "downloadable models in loader dropdowns' in SimpleSyrup "
                "settings or install a WD14 ONNX model and CSV in the "
                "wd14_tagger folder."
            )

    def _show_downloadable_models(self) -> bool:
        """Return whether known downloadable catalog entries should be visible."""

        return self._settings_repository.load().show_downloadable_models

    def _entry_artifacts_are_local(self, entry: ModelEntry) -> bool:
        """Return whether every catalog artifact exists locally."""

        return all(
            resolve_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            is not None
            for artifact in entry.artifacts
        )

    def _vitmatte_entry_is_local(self, entry: ModelEntry) -> bool:
        """Return whether a valid ViTMatte directory exists locally."""

        try:
            self._vitmatte_loader.resolve_model_directory(entry, auto_download=False)
        except FileNotFoundError:
            return False
        return True


def default_choice(choices: list[str], preferred: str) -> str:
    """Return the preferred default when visible, otherwise the first choice."""

    if preferred in choices:
        return preferred
    return choices[0]
