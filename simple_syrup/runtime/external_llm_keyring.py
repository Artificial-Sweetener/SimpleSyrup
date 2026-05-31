# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OS credential storage adapter for external LLM API keys."""

from __future__ import annotations

import importlib
from typing import Protocol, cast

from ..domain.external_llm import ExternalLLMConfigError, normalize_base_url

SERVICE_NAME = "SimpleSyrup"


class KeyringProtocol(Protocol):
    """Subset of the keyring API used by SimpleSyrup."""

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return a stored password or None."""

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Store a password."""

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete a stored password."""


class ExternalLLMKeyringError(RuntimeError):
    """Raised when OS credential storage cannot complete an operation."""


class ExternalLLMKeyringStore:
    """Store external LLM API keys in the OS credential backend."""

    def __init__(self, keyring_module: KeyringProtocol | None = None) -> None:
        """Create a keyring store with an injectable credential backend."""

        self._keyring = keyring_module

    def has_api_key(self, base_url: str) -> bool:
        """Return whether an API key exists for the normalized endpoint."""

        return self.get_api_key(base_url) != ""

    def get_api_key(self, base_url: str) -> str:
        """Return the API key for the endpoint or an empty string."""

        username = credential_username(base_url)
        try:
            return self._backend().get_password(SERVICE_NAME, username) or ""
        except Exception as error:
            raise ExternalLLMKeyringError(
                "Could not read the external LLM API key from OS credential storage."
            ) from error

    def save_api_key(self, base_url: str, api_key: str) -> None:
        """Store an API key for the normalized endpoint."""

        secret = api_key.strip()
        if not secret:
            raise ExternalLLMConfigError("External LLM API key must not be empty.")

        username = credential_username(base_url)
        try:
            self._backend().set_password(SERVICE_NAME, username, secret)
        except Exception as error:
            raise ExternalLLMKeyringError(
                "Could not save the external LLM API key in OS credential storage."
            ) from error

    def delete_api_key(self, base_url: str) -> None:
        """Delete the API key for the normalized endpoint when present."""

        username = credential_username(base_url)
        try:
            self._backend().delete_password(SERVICE_NAME, username)
        except Exception as error:
            raise ExternalLLMKeyringError(
                "Could not delete the external LLM API key from OS credential storage."
            ) from error

    def _backend(self) -> KeyringProtocol:
        """Return the configured keyring backend."""

        if self._keyring is not None:
            return self._keyring

        try:
            module = importlib.import_module("keyring")
        except ModuleNotFoundError as error:
            raise ExternalLLMKeyringError(
                "The keyring package is required to store external LLM API keys."
            ) from error
        return cast(KeyringProtocol, module)


def credential_username(base_url: str) -> str:
    """Return the stable keyring username for an endpoint."""

    return f"external-llm:{normalize_base_url(base_url)}"
