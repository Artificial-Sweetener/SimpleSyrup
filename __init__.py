# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SimpleSyrup ComfyUI extension entry point."""

from __future__ import annotations

import sys

from . import simple_syrup as _simple_syrup_package

sys.modules.setdefault("simple_syrup", _simple_syrup_package)

from .simple_syrup.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from .simple_syrup.runtime.settings_routes import register_settings_routes  # noqa: E402

WEB_DIRECTORY = "./web/dist"


async def comfy_entrypoint() -> object:
    """Return Comfy v3 extension nodes without importing optional integrations."""

    from comfy_api.latest import ComfyExtension

    class SimpleSyrupExtension(ComfyExtension):
        """Expose SimpleSyrup's Comfy v3 nodes."""

        async def get_node_list(self) -> list[type[object]]:
            """Return v3 nodes after Comfy asks for them."""

            from .simple_syrup.nodes_v3 import get_nodes

            return get_nodes()

    return SimpleSyrupExtension()


register_settings_routes()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]
