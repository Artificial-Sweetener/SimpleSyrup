# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the source-visible Anima surface from one Comfy MODEL patcher."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Protocol, cast

from comfy.ldm.anima.model import Anima
from torch import nn

from .anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
)


class AnimaSurfaceDiscovery(Protocol):
    """Validate one installed Anima module graph."""

    def discover(self, diffusion_model: object) -> AnimaModuleSurface:
        """Return the exact validated Anima patch surface."""


class AnimaModelPatcherSurfaceResolver:
    """Adapt Comfy's source-visible object boundary to Anima discovery."""

    def __init__(
        self,
        discovery: AnimaSurfaceDiscovery = ANIMA_MODULE_SURFACE_DISCOVERY,
    ) -> None:
        """Retain the installed-surface validation authority."""

        self._discovery = discovery

    def resolve(self, model: object) -> AnimaModuleSurface:
        """Return originals even while another clone's replacements are live."""

        getter = getattr(model, "get_model_object", None)
        if not callable(getter):
            raise TypeError("Anima surface resolution requires MODEL get_model_object.")
        base_model = getattr(model, "model", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None:
            raise TypeError("Anima surface resolution requires MODEL diffusion_model.")
        blocks = getattr(diffusion_model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            return self._discovery.discover(diffusion_model)

        source_blocks = tuple(
            self._source_block(getter, block_index)
            for block_index in range(len(blocks))
        )
        if all(
            source is installed
            for source, installed in zip(source_blocks, blocks, strict=True)
        ):
            return self._discovery.discover(diffusion_model)

        validation_view = self._validation_view(diffusion_model, blocks, source_blocks)
        surface = self._discovery.discover(validation_view)
        if type(diffusion_model) is not Anima:
            raise TypeError("Validated Anima surface root changed unexpectedly.")
        return replace(surface, diffusion_model=cast(Anima, diffusion_model))

    @staticmethod
    def _source_block(getter: object, block_index: int) -> nn.Module:
        """Return one block through Comfy's source-visible object lookup."""

        if not callable(getter):
            raise AssertionError("Validated MODEL object getter became unavailable.")
        path = f"diffusion_model.blocks.{block_index}"
        block = getter(path)
        if not isinstance(block, nn.Module):
            raise TypeError(f"MODEL source-visible object {path!r} must be a module.")
        return block

    @staticmethod
    def _validation_view(
        diffusion_model: object,
        installed_blocks: nn.ModuleList,
        source_blocks: tuple[nn.Module, ...],
    ) -> object:
        """Copy registration shells without registering or mutating host objects."""

        if not isinstance(diffusion_model, nn.Module):
            raise TypeError("Anima diffusion_model must be a PyTorch module.")
        model_view = copy(diffusion_model)
        model_view._modules = diffusion_model._modules.copy()
        blocks_view = copy(installed_blocks)
        blocks_view._modules = installed_blocks._modules.copy()
        for block_index, source_block in enumerate(source_blocks):
            blocks_view[block_index] = source_block
        nn.Module.__setattr__(model_view, "blocks", blocks_view)
        return model_view


ANIMA_MODEL_PATCHER_SURFACE_RESOLVER = AnimaModelPatcherSurfaceResolver()
