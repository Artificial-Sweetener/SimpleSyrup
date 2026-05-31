# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for external LLM prompt generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from ..domain.external_llm import (
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    ExternalLLMChatRequest,
    ExternalLLMChatResponse,
    ExternalLLMConfigError,
    ExternalLLMProviderError,
    normalize_base_url,
)
from ..runtime.external_llm_client import ExternalLLMClient
from ..runtime.external_llm_images import ExternalLLMImageEncoder
from ..runtime.external_llm_keyring import ExternalLLMKeyringStore
from ..runtime.settings import (
    ExternalLLMSettings,
    SimpleSyrupSettings,
    SimpleSyrupSettingsRepository,
)

CONFIGURE_EXTERNAL_LLM = "Configure external LLM endpoint"


class SettingsRepository(Protocol):
    """Settings persistence boundary used by external LLM services."""

    def load(self) -> SimpleSyrupSettings:
        """Load current SimpleSyrup settings."""

    def save(self, settings: SimpleSyrupSettings) -> SimpleSyrupSettings:
        """Persist current SimpleSyrup settings."""


class KeyStore(Protocol):
    """Credential storage boundary used by external LLM services."""

    def has_api_key(self, base_url: str) -> bool:
        """Return whether an API key exists for an endpoint."""

    def get_api_key(self, base_url: str) -> str:
        """Return an API key for an endpoint."""

    def save_api_key(self, base_url: str, api_key: str) -> None:
        """Store an API key for an endpoint."""

    def delete_api_key(self, base_url: str) -> None:
        """Delete an API key for an endpoint."""


class ProviderClient(Protocol):
    """External LLM provider client boundary."""

    def list_models(self, base_url: str, api_key: str) -> tuple[str, ...]:
        """Return provider model ids."""

    def create_chat_completion(
        self,
        base_url: str,
        api_key: str,
        request: ExternalLLMChatRequest,
    ) -> ExternalLLMChatResponse:
        """Return provider chat completion content wrapper."""


class ImageEncoder(Protocol):
    """Image encoding boundary for external LLM vision inputs."""

    def encode_first_image_as_data_url(self, image: object) -> str:
        """Return the first ComfyUI IMAGE tensor item as a data URL."""


class ExternalLLMPromptService:
    """Coordinate external LLM settings, credentials, and provider calls."""

    def __init__(
        self,
        settings_repository: SettingsRepository | None = None,
        key_store: KeyStore | None = None,
        client: ProviderClient | None = None,
        image_encoder: ImageEncoder | None = None,
    ) -> None:
        """Create the service with injectable runtime boundaries."""

        self._settings_repository = (
            settings_repository or SimpleSyrupSettingsRepository()
        )
        self._key_store = key_store or ExternalLLMKeyringStore()
        self._client = client or ExternalLLMClient()
        self._image_encoder = image_encoder or ExternalLLMImageEncoder()

    def model_choices(self) -> list[str]:
        """Return cached provider model choices for Comfy dropdowns."""

        models = self._settings_repository.load().external_llm.cached_models
        return list(models) if models else [CONFIGURE_EXTERNAL_LLM]

    def provider_is_configured(self) -> bool:
        """Return whether endpoint and API key are configured."""

        external = self._settings_repository.load().external_llm
        return bool(external.base_url) and self._key_store.has_api_key(
            external.base_url
        )

    def settings_payload(self) -> dict[str, object]:
        """Return non-secret external LLM settings for frontend routes."""

        external = self._settings_repository.load().external_llm
        has_api_key = (
            self._key_store.has_api_key(external.base_url)
            if external.base_url
            else False
        )
        return {
            "base_url": external.base_url,
            "cached_models": list(external.cached_models),
            "default_model": external.default_model,
            "has_api_key": has_api_key,
        }

    def refresh_models(self) -> tuple[str, ...]:
        """Refresh cached provider models when endpoint credentials exist."""

        settings = self._settings_repository.load()
        external = settings.external_llm
        if not external.base_url:
            return external.cached_models

        if not self._key_store.has_api_key(external.base_url):
            return external.cached_models

        api_key = self._key_store.get_api_key(external.base_url)
        models = self._client.list_models(external.base_url, api_key)
        default_model = external.default_model
        if models and default_model not in models:
            default_model = models[0]
        if not models:
            default_model = ""

        saved = self._settings_repository.save(
            replace(
                settings,
                external_llm=replace(
                    external,
                    cached_models=models,
                    default_model=default_model,
                ),
            )
        )
        return saved.external_llm.cached_models

    def save_config(
        self, base_url: str, default_model: str = ""
    ) -> SimpleSyrupSettings:
        """Persist non-secret external LLM endpoint settings."""

        settings = self._settings_repository.load()
        normalized_url = normalize_base_url(base_url) if base_url.strip() else ""
        endpoint_changed = normalized_url != settings.external_llm.base_url
        cached_models = () if endpoint_changed else settings.external_llm.cached_models
        default = default_model.strip()
        if endpoint_changed:
            default = ""
        if default and cached_models and default not in cached_models:
            raise ExternalLLMConfigError(
                "Default external LLM model must be one of the cached models."
            )

        self._settings_repository.save(
            replace(
                settings,
                external_llm=ExternalLLMSettings(
                    base_url=normalized_url,
                    cached_models=cached_models,
                    default_model=default,
                ),
            )
        )
        try:
            self.refresh_models()
        except ExternalLLMProviderError:
            pass
        return self._settings_repository.load()

    def save_api_key(self, api_key: str) -> SimpleSyrupSettings:
        """Store the API key for the configured endpoint."""

        settings = self._settings_repository.load()
        base_url = settings.external_llm.base_url
        if not base_url:
            raise ExternalLLMConfigError(
                "Configure an external LLM endpoint in SimpleSyrup settings before "
                "adding an API key."
            )
        self._key_store.save_api_key(base_url, api_key)
        try:
            self.refresh_models()
        except ExternalLLMProviderError:
            pass
        return self._settings_repository.load()

    def delete_api_key(self) -> SimpleSyrupSettings:
        """Delete the API key for the configured endpoint."""

        settings = self._settings_repository.load()
        base_url = settings.external_llm.base_url
        if base_url:
            self._key_store.delete_api_key(base_url)
        return settings

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
        image: object | None = None,
    ) -> str:
        """Generate an assistant response for the supplied prompt pair."""

        return self.generate_with_image_data_url(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            image_data_url=(
                None
                if image is None
                else self._image_encoder.encode_first_image_as_data_url(image)
            ),
        )

    def generate_with_image_data_url(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
        image_data_url: str | None = None,
    ) -> str:
        """Generate an assistant response with a pre-encoded optional image."""

        selected_model = self._resolve_model_for_execution(model)

        external = self._settings_repository.load().external_llm
        if not external.base_url:
            raise ExternalLLMConfigError(
                "Configure an external LLM endpoint in SimpleSyrup settings before "
                "using this node."
            )

        api_key = self._key_store.get_api_key(external.base_url)
        if not api_key:
            raise ExternalLLMConfigError(
                "Add an API key for the configured external LLM endpoint in "
                "SimpleSyrup settings."
            )

        request = ExternalLLMChatRequest(
            model=selected_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            image_data_url=image_data_url,
        )
        response = self._client.create_chat_completion(
            external.base_url,
            api_key,
            request,
        )
        return response.content

    def _resolve_model_for_execution(self, model: str) -> str:
        """Return a concrete provider model, refreshing stale sentinel choices."""

        selected_model = model.strip()
        if selected_model != CONFIGURE_EXTERNAL_LLM:
            return selected_model

        settings = self._settings_repository.load()
        external = settings.external_llm
        if not external.base_url:
            raise ExternalLLMConfigError(
                "Configure an external LLM endpoint in SimpleSyrup settings before "
                "using this node."
            )
        if not self._key_store.has_api_key(external.base_url):
            raise ExternalLLMConfigError(
                "Add an API key for the configured external LLM endpoint in "
                "SimpleSyrup settings."
            )

        models = external.cached_models or self.refresh_models()
        refreshed_external = self._settings_repository.load().external_llm
        default_model = refreshed_external.default_model
        if default_model and default_model in models:
            return default_model
        if models:
            return models[0]
        raise ExternalLLMConfigError(
            "The configured external LLM endpoint did not report any models. "
            "Check the endpoint URL and API key in SimpleSyrup settings."
        )
