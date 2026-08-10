# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt persisted SimpleSyrup settings to the quant cache byte budget."""

from __future__ import annotations

from .settings_repository import SimpleSyrupSettingsRepository

BYTES_PER_GIB = 1024**3


class SettingsQuantCacheLimitProvider:
    """Read the authoritative global quant cache limit from backend settings."""

    def __init__(
        self,
        repository: SimpleSyrupSettingsRepository | None = None,
    ) -> None:
        """Create a provider with injectable settings persistence."""

        self._repository = repository or SimpleSyrupSettingsRepository()

    def limit_bytes(self) -> int:
        """Return the configured integer GiB limit in bytes."""

        return self._repository.load().quant_cache_limit_gib * BYTES_PER_GIB
