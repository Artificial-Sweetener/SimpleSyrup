# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build settings-aware model dropdown choices for loader nodes."""

from __future__ import annotations

from typing import Protocol

from .model_catalog import (
    grounding_dino_choices,
    sam_choices,
    ultralytics_choices,
    vitmatte_choices,
    wd14_tagger_choices,
)
from .settings import SimpleSyrupSettings
from .settings_repository import SimpleSyrupSettingsRepository

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
    ) -> None:
        """Create the choice service with an injectable settings boundary."""

        self._settings_repository = (
            settings_repository or SimpleSyrupSettingsRepository()
        )

    def sam_choices(self) -> list[str]:
        """Return curated SAM choices when catalog models are visible."""

        if self._show_downloadable_models():
            return sam_choices()

        return [NO_LOCAL_SAM_MODELS]

    def grounding_dino_choices(self) -> list[str]:
        """Return curated GroundingDINO choices when catalog models are visible."""

        if self._show_downloadable_models():
            return grounding_dino_choices()

        return [NO_LOCAL_GROUNDING_DINO_MODELS]

    def vitmatte_choices(self) -> list[str]:
        """Return curated ViTMatte choices when catalog models are visible."""

        if self._show_downloadable_models():
            return vitmatte_choices()

        return [NO_LOCAL_VITMATTE_MODELS]

    def wd14_tagger_choices(self) -> list[str]:
        """Return curated WD14 tagger choices when catalog models are visible."""

        if self._show_downloadable_models():
            return wd14_tagger_choices()

        return [NO_LOCAL_WD14_TAGGER_MODELS]

    def ultralytics_choices(self) -> list[str]:
        """Return curated Ultralytics choices when catalog models are visible."""

        if self._show_downloadable_models():
            return ultralytics_choices()

        return []

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


def default_choice(choices: list[str], preferred: str) -> str:
    """Return the preferred default when visible, otherwise the first choice."""

    if preferred in choices:
        return preferred
    return choices[0]
