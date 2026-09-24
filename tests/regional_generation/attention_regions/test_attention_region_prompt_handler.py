# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test safe one-time attention-region prompt handler registration."""

from __future__ import annotations

import sys
from typing import Any

import simple_syrup.runtime.attention_region_prompt_handler as handler_module


class _PromptServer:
    """Record prompt handlers without importing Comfy's HTTP server."""

    def __init__(self) -> None:
        """Create an empty handler list."""

        self.handlers: list[object] = []

    def add_on_prompt_handler(self, handler: object) -> None:
        """Record one registered handler."""

        self.handlers.append(handler)


def test_prompt_handler_registers_once(monkeypatch: Any) -> None:
    """Prevent duplicate graph rewrites across extension reload paths."""

    instance = _PromptServer()
    fake_module = type("ServerModule", (), {})()
    fake_module.PromptServer = type("PromptServerType", (), {"instance": instance})
    monkeypatch.setitem(sys.modules, "server", fake_module)

    handler_module.register_attention_region_prompt_handler()
    handler_module.register_attention_region_prompt_handler()

    assert len(instance.handlers) == 1
