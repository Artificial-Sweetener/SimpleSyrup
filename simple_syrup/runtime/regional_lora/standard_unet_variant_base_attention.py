# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare graph-local packed attn2 callbacks for standard-UNet base output."""

from __future__ import annotations

from ..attention_coupling.unet_attn2_execution_resolver import (
    UnetAttn2ExecutionResolver,
)
from ..attention_coupling.unet_attn2_patch import UnetAttn2PatchPair

_INPUT_PATCH_KEY = "attn2_patch"
_OUTPUT_PATCH_KEY = "attn2_output_patch"


class StandardUnetVariantBaseAttention:
    """Install Attention Couple only on graph-local unpatched base execution."""

    def __init__(self, resolver: UnetAttn2ExecutionResolver) -> None:
        """Retain one paired callback authority for the active request state."""

        if not isinstance(resolver, UnetAttn2ExecutionResolver):
            raise TypeError("Standard UNet base attention requires a resolver.")
        self._patches = UnetAttn2PatchPair(resolver)

    def prepare(self, source: dict[str, object]) -> dict[str, object]:
        """Return isolated transformer options with one collision-free pair."""

        if not isinstance(source, dict):
            raise TypeError(
                "Standard UNet base attention options must be a dictionary."
            )
        prepared = source.copy()
        source_patches = source.get("patches")
        if source_patches is None:
            patches: dict[object, object] = {}
        elif isinstance(source_patches, dict):
            patches = source_patches.copy()
        else:
            raise TypeError("Standard UNet transformer patches must be a dictionary.")
        for key in (_INPUT_PATCH_KEY, _OUTPUT_PATCH_KEY):
            if key in patches:
                raise ValueError(f"Standard UNet base graph already contains {key!r}.")
        patches[_INPUT_PATCH_KEY] = [self._patches.input_patch]
        patches[_OUTPUT_PATCH_KEY] = [self._patches.output_patch]
        prepared["patches"] = patches
        return prepared
