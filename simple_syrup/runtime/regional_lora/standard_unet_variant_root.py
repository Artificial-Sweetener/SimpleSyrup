# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create one patcher-visible diffusion root for persistent regional variants."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass

from torch import nn

_VARIANT_CONTAINER_NAME = "simple_syrup_regional_variants"


@dataclass(frozen=True, slots=True)
class StandardUnetVariantRoot:
    """Retain one base-compatible root and its registered variant container."""

    module: nn.Module
    variants: nn.ModuleDict

    def install_forward(self, forward: Callable[..., object]) -> None:
        """Bind the sole regional forward owner to the private root clone."""

        if not callable(forward):
            raise TypeError("Standard UNet variant forward must be callable.")
        self.module.forward = forward


class StandardUnetVariantRootBuilder:
    """Clone root-owned registries while sharing the globally patched base graph."""

    @staticmethod
    def build(source: object) -> StandardUnetVariantRoot:
        """Return an uninstalled root with an empty managed variant container."""

        if not isinstance(source, nn.Module):
            raise TypeError("Standard UNet variant root requires a module.")
        if hasattr(source, _VARIANT_CONTAINER_NAME):
            raise ValueError("Standard UNet source already owns regional variants.")
        root = copy(source)
        root._modules = source._modules.copy()
        root._parameters = source._parameters.copy()
        root._buffers = source._buffers.copy()
        root._non_persistent_buffers_set = source._non_persistent_buffers_set.copy()
        variants = nn.ModuleDict()
        root.add_module(_VARIANT_CONTAINER_NAME, variants)
        return StandardUnetVariantRoot(root, variants)


STANDARD_UNET_VARIANT_ROOT_BUILDER = StandardUnetVariantRootBuilder()
