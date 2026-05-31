# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external LLM prompt service orchestration."""

from __future__ import annotations

import pytest

from simple_syrup.domain.external_llm import (
    ExternalLLMChatRequest,
    ExternalLLMChatResponse,
    ExternalLLMConfigError,
    ExternalLLMProviderError,
)
from simple_syrup.runtime.settings import ExternalLLMSettings, SimpleSyrupSettings
from simple_syrup.services.external_llm_prompt_service import (
    CONFIGURE_EXTERNAL_LLM,
    ExternalLLMPromptService,
)


class FakeSettingsRepository:
    """In-memory settings repository."""

    def __init__(self, settings: SimpleSyrupSettings) -> None:
        """Store initial settings."""

        self.settings = settings

    def load(self) -> SimpleSyrupSettings:
        """Return current settings."""

        return self.settings

    def save(self, settings: SimpleSyrupSettings) -> SimpleSyrupSettings:
        """Persist current settings."""

        self.settings = settings
        return settings


class FakeKeyStore:
    """Credential store double."""

    def __init__(self, api_key: str = "") -> None:
        """Store a fake API key."""

        self.api_key = api_key

    def has_api_key(self, _base_url: str) -> bool:
        """Return whether a key exists."""

        return bool(self.api_key)

    def get_api_key(self, _base_url: str) -> str:
        """Return the fake key."""

        return self.api_key

    def save_api_key(self, _base_url: str, api_key: str) -> None:
        """Save the fake key."""

        self.api_key = api_key

    def delete_api_key(self, _base_url: str) -> None:
        """Delete the fake key."""

        self.api_key = ""


class FakeClient:
    """Provider client double."""

    def __init__(self, models: tuple[str, ...] = ("model-a",)) -> None:
        """Store provider models."""

        self.models = models
        self.requests: list[ExternalLLMChatRequest] = []
        self.list_model_calls = 0

    def list_models(self, _base_url: str, _api_key: str) -> tuple[str, ...]:
        """Return configured models."""

        self.list_model_calls += 1
        return self.models

    def create_chat_completion(
        self,
        _base_url: str,
        _api_key: str,
        request: ExternalLLMChatRequest,
    ) -> ExternalLLMChatResponse:
        """Capture request and return content."""

        self.requests.append(request)
        return ExternalLLMChatResponse("assistant response")


class FailingModelClient(FakeClient):
    """Provider client double that cannot refresh models."""

    def list_models(self, _base_url: str, _api_key: str) -> tuple[str, ...]:
        """Raise a provider failure during model refresh."""

        self.list_model_calls += 1
        raise ExternalLLMProviderError("The external LLM provider rejected /models.")


class FakeImageEncoder:
    """Image encoder double."""

    def encode_first_image_as_data_url(self, image: object) -> str:
        """Return a deterministic data URL."""

        assert image == "image"
        return "data:image/png;base64,abc"


def test_model_choices_return_sentinel_without_cached_models() -> None:
    """No cached models produces a clear sentinel choice."""

    service = ExternalLLMPromptService(
        FakeSettingsRepository(SimpleSyrupSettings()),
        FakeKeyStore(),
        FakeClient(),
    )

    assert service.model_choices() == [CONFIGURE_EXTERNAL_LLM]


def test_refresh_models_skips_when_unconfigured() -> None:
    """Refresh does no network work without endpoint credentials."""

    repository = FakeSettingsRepository(SimpleSyrupSettings())
    client = FakeClient(("model-a",))
    service = ExternalLLMPromptService(repository, FakeKeyStore(), client)

    assert service.refresh_models() == ()
    assert repository.load().external_llm.cached_models == ()


def test_refresh_models_persists_provider_models() -> None:
    """Refresh stores sanitized provider model choices."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://provider.example/v1",
                cached_models=("old-model",),
                default_model="old-model",
            )
        )
    )
    service = ExternalLLMPromptService(
        repository,
        FakeKeyStore("secret"),
        FakeClient(("model-a", "model-b")),
    )

    assert service.refresh_models() == ("model-a", "model-b")
    assert repository.load().external_llm.default_model == "model-a"


def test_save_config_clears_cached_models_when_endpoint_changes() -> None:
    """Provider model cache belongs to the configured endpoint."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://old-provider.example/v1",
                cached_models=("old-model",),
                default_model="old-model",
            )
        )
    )
    service = ExternalLLMPromptService(repository, FakeKeyStore(), FakeClient())

    saved = service.save_config("https://new-provider.example/v1")

    assert saved.external_llm.base_url == "https://new-provider.example/v1"
    assert saved.external_llm.cached_models == ()
    assert saved.external_llm.default_model == ""


def test_save_config_preserves_cached_models_for_same_endpoint() -> None:
    """Saving the existing endpoint keeps its model cache and default choice."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://provider.example/v1",
                cached_models=("model-a", "model-b"),
                default_model="model-a",
            )
        )
    )
    service = ExternalLLMPromptService(repository, FakeKeyStore(), FakeClient())

    saved = service.save_config(
        "https://provider.example/v1/",
        default_model="model-b",
    )

    assert saved.external_llm.cached_models == ("model-a", "model-b")
    assert saved.external_llm.default_model == "model-b"


def test_save_config_refreshes_models_when_endpoint_has_key() -> None:
    """Saving an endpoint updates the available model cache when possible."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://provider.example/v1",
                cached_models=("old-model",),
                default_model="old-model",
            )
        )
    )
    client = FakeClient(("model-a", "model-b"))
    service = ExternalLLMPromptService(repository, FakeKeyStore("secret"), client)

    saved = service.save_config("https://provider.example/v1")

    assert client.list_model_calls == 1
    assert saved.external_llm.cached_models == ("model-a", "model-b")
    assert saved.external_llm.default_model == "model-a"


def test_save_config_succeeds_when_model_refresh_fails() -> None:
    """Endpoint storage is not blocked by provider model discovery failures."""

    repository = FakeSettingsRepository(SimpleSyrupSettings())
    service = ExternalLLMPromptService(
        repository,
        FakeKeyStore("secret"),
        FailingModelClient(),
    )

    saved = service.save_config("https://provider.example/v1")

    assert saved.external_llm.base_url == "https://provider.example/v1"
    assert saved.external_llm.cached_models == ()


def test_save_api_key_persists_key_when_model_refresh_fails() -> None:
    """Credential storage is independent from provider model discovery."""

    key_store = FakeKeyStore()
    service = ExternalLLMPromptService(
        FakeSettingsRepository(
            SimpleSyrupSettings(
                external_llm=ExternalLLMSettings(
                    base_url="https://provider.example/v1",
                )
            )
        ),
        key_store,
        FailingModelClient(),
    )

    saved = service.save_api_key("secret")

    assert key_store.has_api_key(saved.external_llm.base_url) is True
    assert saved.external_llm.cached_models == ()


def test_generate_rejects_sentinel_when_unconfigured() -> None:
    """The setup sentinel still requires endpoint configuration."""

    service = ExternalLLMPromptService(
        FakeSettingsRepository(SimpleSyrupSettings()),
        FakeKeyStore(),
        FakeClient(),
    )

    with pytest.raises(ExternalLLMConfigError, match="Configure an external LLM"):
        service.generate(CONFIGURE_EXTERNAL_LLM, "", "prompt")


def test_generate_refreshes_stale_sentinel_and_uses_default_model() -> None:
    """Execution recovers when the graph still holds the setup sentinel."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://provider.example/v1",
            )
        )
    )
    client = FakeClient(("model-a", "model-b"))
    service = ExternalLLMPromptService(repository, FakeKeyStore("secret"), client)

    assert service.generate(CONFIGURE_EXTERNAL_LLM, "system", "user") == (
        "assistant response"
    )
    assert client.list_model_calls == 1
    assert client.requests[0].model == "model-a"
    assert repository.load().external_llm.cached_models == ("model-a", "model-b")


def test_generate_uses_cached_default_when_sentinel_is_stale() -> None:
    """A stale sentinel resolves to the saved default without provider refresh."""

    repository = FakeSettingsRepository(
        SimpleSyrupSettings(
            external_llm=ExternalLLMSettings(
                base_url="https://provider.example/v1",
                cached_models=("model-a", "model-b"),
                default_model="model-b",
            )
        )
    )
    client = FakeClient(("model-c",))
    service = ExternalLLMPromptService(repository, FakeKeyStore("secret"), client)

    assert service.generate(CONFIGURE_EXTERNAL_LLM, "system", "user") == (
        "assistant response"
    )
    assert client.list_model_calls == 0
    assert client.requests[0].model == "model-b"


def test_generate_rejects_empty_user_prompt() -> None:
    """User prompt validation happens before provider work."""

    service = configured_service()

    with pytest.raises(ExternalLLMConfigError, match="User prompt"):
        service.generate("model-a", "", " ")


def test_generate_returns_provider_response() -> None:
    """Prompt generation delegates to the provider client."""

    service = configured_service()

    assert service.generate("model-a", "system", "user") == "assistant response"


def test_generate_forwards_max_tokens() -> None:
    """Prompt generation includes the requested response token limit."""

    client = FakeClient()
    service = configured_service(client=client)

    assert service.generate("model-a", "system", "user", max_tokens=128) == (
        "assistant response"
    )
    assert client.requests[0].max_tokens == 128


def test_generate_forwards_reasoning_effort() -> None:
    """Prompt generation includes the requested reasoning behavior."""

    client = FakeClient()
    service = configured_service(client=client)

    assert (
        service.generate(
            "model-a",
            "system",
            "user",
            reasoning_effort="off",
        )
        == "assistant response"
    )
    assert client.requests[0].reasoning_effort == "off"


def test_generate_rejects_invalid_max_tokens() -> None:
    """Response token limit must be a positive integer."""

    service = configured_service()

    with pytest.raises(ExternalLLMConfigError, match="max tokens"):
        service.generate("model-a", "system", "user", max_tokens=0)


def test_generate_rejects_invalid_reasoning_effort() -> None:
    """Reasoning effort must be one of the node choices."""

    service = configured_service()

    with pytest.raises(ExternalLLMConfigError, match="reasoning effort"):
        service.generate("model-a", "system", "user", reasoning_effort="none")


def test_generate_attaches_encoded_image_when_supplied() -> None:
    """Optional image inputs are encoded before provider chat requests."""

    client = FakeClient()
    service = configured_service(client=client, image_encoder=FakeImageEncoder())

    assert (
        service.generate("model-a", "system", "user", image="image")
        == "assistant response"
    )
    assert client.requests[0].image_data_url == "data:image/png;base64,abc"


def test_generate_with_image_data_url_forwards_preencoded_image() -> None:
    """SEG callers can supply a prebuilt image data URL."""

    client = FakeClient()
    service = configured_service(client=client)

    assert (
        service.generate_with_image_data_url(
            model="model-a",
            system_prompt="system",
            user_prompt="user",
            image_data_url="data:image/png;base64,seg",
        )
        == "assistant response"
    )
    assert client.requests[0].image_data_url == "data:image/png;base64,seg"


def configured_service(
    client: FakeClient | None = None,
    image_encoder: FakeImageEncoder | None = None,
) -> ExternalLLMPromptService:
    """Create a service with endpoint, cached model, key, and fake client."""

    return ExternalLLMPromptService(
        FakeSettingsRepository(
            SimpleSyrupSettings(
                external_llm=ExternalLLMSettings(
                    base_url="https://provider.example/v1",
                    cached_models=("model-a",),
                    default_model="model-a",
                )
            )
        ),
        FakeKeyStore("secret"),
        client or FakeClient(),
        image_encoder=image_encoder,
    )
