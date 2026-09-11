# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare graph-local packed attn2 callbacks for standard-UNet base output."""

from __future__ import annotations

from ..attention_coupling.unet_attn2_execution_resolver import (
    UnetAttn2ExecutionResolver,
)
from ..attention_coupling.unet_attn2_patch import UnetAttn2PatchPair
from ..ppm_negpip_interop import PpmNegpipInterop

_INPUT_PATCH_KEY = "attn2_patch"
_OUTPUT_PATCH_KEY = "attn2_output_patch"


class StandardUnetVariantBaseAttention:
    """Install Attention Couple only on graph-local unpatched base execution."""

    def __init__(
        self,
        resolver: UnetAttn2ExecutionResolver,
        *,
        negpip: PpmNegpipInterop | None = None,
    ) -> None:
        """Retain one paired callback authority for the active request state."""

        if not isinstance(resolver, UnetAttn2ExecutionResolver):
            raise TypeError("Standard UNet base attention requires a resolver.")
        self._patches = UnetAttn2PatchPair(resolver)
        if negpip is not None and not isinstance(negpip, PpmNegpipInterop):
            raise TypeError("Standard UNet base attention NegPiP state is invalid.")
        self._negpip = negpip

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
        expected_input = [] if self._negpip is None else [self._negpip.attention_patch]
        if (
            self._negpip is None
            and _INPUT_PATCH_KEY in patches
            or self._negpip is not None
            and patches.get(_INPUT_PATCH_KEY) != expected_input
        ):
            raise ValueError(
                "Standard UNet base graph already contains an unadmitted attn2 "
                "input patch."
            )
        if _OUTPUT_PATCH_KEY in patches:
            raise ValueError(
                "Standard UNet base graph already contains 'attn2_output_patch'."
            )
        patches[_INPUT_PATCH_KEY] = [self._patches.input_patch, *expected_input]
        patches[_OUTPUT_PATCH_KEY] = [self._patches.output_patch]
        prepared["patches"] = patches
        return prepared
