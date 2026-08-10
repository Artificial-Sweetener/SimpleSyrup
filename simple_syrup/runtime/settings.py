# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent backend settings for SimpleSyrup runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.external_llm import (
    ExternalLLMConfigError,
    normalize_base_url,
    normalize_model_ids,
)
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
DEFAULT_QUANT_CACHE_LIMIT_GIB = 20
MIN_QUANT_CACHE_LIMIT_GIB = 1
MAX_QUANT_CACHE_LIMIT_GIB = 2048


class SimpleSyrupSettingsError(ValueError):
    """Raised when SimpleSyrup settings data is malformed."""


@dataclass(frozen=True)
class ExternalLLMSettings:
    """Non-secret external LLM provider settings persisted in Comfy's user data."""

    base_url: str = ""
    cached_models: tuple[str, ...] = ()
    default_model: str = ""

    def to_payload(self) -> dict[str, object]:
        """Return the validated external LLM JSON payload shape."""

        return {
            "base_url": self.base_url,
            "cached_models": list(self.cached_models),
            "default_model": self.default_model,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ExternalLLMSettings:
        """Create external LLM settings from a decoded JSON payload."""

        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm settings must be a JSON object."
            )

        base_url = payload.get("base_url", "")
        if not isinstance(base_url, str):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm.base_url must be a string."
            )

        default_model = payload.get("default_model", "")
        if not isinstance(default_model, str):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm.default_model must be a string."
            )

        try:
            cached_models = normalize_model_ids(payload.get("cached_models", []))
        except ExternalLLMConfigError as error:
            raise SimpleSyrupSettingsError(str(error)) from error

        default = default_model.strip()
        if default and cached_models and default not in cached_models:
            default = cached_models[0]

        normalized_url = ""
        if base_url.strip():
            try:
                normalized_url = normalize_base_url(base_url)
            except ExternalLLMConfigError as error:
                raise SimpleSyrupSettingsError(str(error)) from error

        return cls(
            base_url=normalized_url,
            cached_models=cached_models,
            default_model=default,
        )


@dataclass(frozen=True)
class SimpleSyrupSettings:
    """User-configurable SimpleSyrup runtime settings."""

    show_downloadable_models: bool = True
    quant_cache_limit_gib: int = DEFAULT_QUANT_CACHE_LIMIT_GIB
    external_llm: ExternalLLMSettings = field(default_factory=ExternalLLMSettings)

    def to_payload(self) -> dict[str, object]:
        """Return the validated JSON payload shape."""

        return {
            "show_downloadable_models": self.show_downloadable_models,
            "quant_cache_limit_gib": self.quant_cache_limit_gib,
            "external_llm": self.external_llm.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> SimpleSyrupSettings:
        """Create settings from a decoded JSON payload."""

        if not isinstance(payload, dict):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup settings payload must be a JSON object."
            )

        value = payload.get("show_downloadable_models")
        if not isinstance(value, bool):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup settings payload is invalid. Expected "
                "show_downloadable_models to be a boolean."
            )

        quant_cache_limit = payload.get(
            "quant_cache_limit_gib", DEFAULT_QUANT_CACHE_LIMIT_GIB
        )
        if (
            not isinstance(quant_cache_limit, int)
            or isinstance(quant_cache_limit, bool)
            or not MIN_QUANT_CACHE_LIMIT_GIB
            <= quant_cache_limit
            <= MAX_QUANT_CACHE_LIMIT_GIB
        ):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup settings payload is invalid. Expected "
                f"quant_cache_limit_gib to be an integer from "
                f"{MIN_QUANT_CACHE_LIMIT_GIB} to {MAX_QUANT_CACHE_LIMIT_GIB}."
            )

        try:
            external_llm = ExternalLLMSettings.from_payload(payload.get("external_llm"))
        except SimpleSyrupSettingsError as error:
            LOGGER.warning(
                "using default external llm settings after failed load",
                extra={"reason": str(error)},
            )
            external_llm = ExternalLLMSettings()

        return cls(
            show_downloadable_models=value,
            quant_cache_limit_gib=quant_cache_limit,
            external_llm=external_llm,
        )
