# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external LLM keyring credential storage."""

from __future__ import annotations

import pytest

from simple_syrup.domain.external_llm import ExternalLLMConfigError
from simple_syrup.runtime.external_llm_keyring import (
    ExternalLLMKeyringError,
    ExternalLLMKeyringStore,
    credential_username,
)


class FakeKeyring:
    """In-memory keyring double."""

    def __init__(self) -> None:
        """Create empty credential storage."""

        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return a stored password."""

        return self.passwords.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Store a password."""

        self.passwords[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete a password."""

        self.passwords.pop((service_name, username), None)


class FailingKeyring(FakeKeyring):
    """Keyring double that fails all operations."""

    def get_password(self, service_name: str, username: str) -> str | None:
        """Raise on read."""

        raise RuntimeError("backend failed")

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Raise on write."""

        raise RuntimeError("backend failed")

    def delete_password(self, service_name: str, username: str) -> None:
        """Raise on delete."""

        raise RuntimeError("backend failed")


def test_keyring_store_saves_reads_checks_and_deletes_api_key() -> None:
    """API keys are stored under the normalized endpoint credential name."""

    keyring = FakeKeyring()
    store = ExternalLLMKeyringStore(keyring)

    store.save_api_key("https://provider.example/v1/", " secret ")

    assert store.has_api_key("https://provider.example/v1") is True
    assert store.get_api_key("https://provider.example/v1") == "secret"
    assert credential_username("https://provider.example/v1/").endswith(
        "https://provider.example/v1"
    )

    store.delete_api_key("https://provider.example/v1")

    assert store.has_api_key("https://provider.example/v1") is False


def test_keyring_store_rejects_empty_api_key() -> None:
    """Empty API keys fail before touching credential storage."""

    store = ExternalLLMKeyringStore(FakeKeyring())

    with pytest.raises(ExternalLLMConfigError, match="must not be empty"):
        store.save_api_key("https://provider.example/v1", " ")


def test_keyring_failures_do_not_expose_api_key() -> None:
    """Credential backend failures are wrapped without leaking secrets."""

    store = ExternalLLMKeyringStore(FailingKeyring())

    with pytest.raises(ExternalLLMKeyringError) as error:
        store.save_api_key("https://provider.example/v1", "secret-value")

    assert "secret-value" not in str(error.value)
