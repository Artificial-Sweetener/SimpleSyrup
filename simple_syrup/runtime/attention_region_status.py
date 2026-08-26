# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish non-blocking attention-region status through Comfy's progress channel."""

from __future__ import annotations

import logging
from importlib import import_module

LOGGER = logging.getLogger(__name__)


class AttentionRegionStatusPublisher:
    """Send concise node status without making UI transport a runtime dependency."""

    def publish(self, node_id: str, message: str) -> None:
        """Send progress text when Comfy's PromptServer is available."""

        if not node_id or not message:
            return
        try:
            server_module = import_module("server")
            prompt_server = getattr(server_module, "PromptServer", None)
            instance = getattr(prompt_server, "instance", None)
            sender = getattr(instance, "send_progress_text", None)
            if callable(sender):
                sender(message, node_id)
        except (ImportError, AttributeError, RuntimeError):
            LOGGER.debug(
                "Attention-region status transport is unavailable",
                extra={"node_id": node_id, "status": message},
                exc_info=True,
            )


ATTENTION_REGION_STATUS_PUBLISHER = AttentionRegionStatusPublisher()
