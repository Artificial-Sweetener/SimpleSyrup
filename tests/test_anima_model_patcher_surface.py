# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove source-visible Anima discovery across installed clone replacements."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anima_module_surface_fixtures import installed_meta_anima
from torch import nn

from simple_syrup.runtime.model_patcher_mutations import (
    ModelExactObjectPatchMutation,
)
from simple_syrup.runtime.regional_lora.anima_model_patcher_surface import (
    AnimaModelPatcherSurfaceResolver,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
    AnimaModuleSurfaceDiscovery,
    AnimaModuleSurfaceError,
)
from simple_syrup.runtime.regional_lora.anima_targets import ANIMA_BLOCK_COUNT


class _SourceVisiblePatcher:
    """Expose original objects while the shared live graph remains contaminated."""

    def __init__(self, diffusion_model: nn.Module) -> None:
        """Retain one host root and its original block/target inventory."""

        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.object_patches: dict[object, object] = {}
        self.object_patches_backup: dict[object, object] = {}
        self._objects = dict(diffusion_model.named_modules())

    def get_model_object(self, name: str) -> nn.Module:
        """Return the source-visible object for one canonical model path."""

        relative = name.removeprefix("diffusion_model.")
        value = self._objects.get(relative)
        if not isinstance(value, nn.Module):
            raise KeyError(name)
        return value

    def add_object_patch(self, name: str, obj: object) -> None:
        """Record one exact mutation without touching the host graph."""

        self.object_patches[name] = obj


class _RecordingDiscovery:
    """Record which root receives installed-surface validation."""

    def __init__(self) -> None:
        """Create an empty call ledger."""

        self.roots: list[object] = []

    def discover(self, diffusion_model: object) -> AnimaModuleSurface:
        """Record and delegate one installed-surface discovery."""

        self.roots.append(diffusion_model)
        return ANIMA_MODULE_SURFACE_DISCOVERY.discover(diffusion_model)


def test_resolves_original_surface_while_prior_clone_replacement_is_installed() -> None:
    """Build mutations from source backups rather than a prior live replacement."""

    diffusion_model = installed_meta_anima()
    patcher = _SourceVisiblePatcher(diffusion_model)
    original_blocks = tuple(diffusion_model.blocks)
    diffusion_model.blocks[0] = nn.Identity()

    with pytest.raises(AnimaModuleSurfaceError, match="blocks.0"):
        ANIMA_MODULE_SURFACE_DISCOVERY.discover(diffusion_model)

    surface = AnimaModelPatcherSurfaceResolver().resolve(patcher)

    assert surface.diffusion_model is diffusion_model
    assert tuple(block.block for block in surface.blocks) == original_blocks
    assert len(surface.blocks) == ANIMA_BLOCK_COUNT
    assert len(surface.lora_targets) == 448
    target = surface.lora_targets[0]
    replacement = nn.Identity()
    ModelExactObjectPatchMutation(
        target.target_name,
        target.module,
        replacement,
    ).apply(patcher)
    assert patcher.object_patches[target.target_name] is replacement


def test_clean_surface_uses_live_root_without_constructing_a_validation_view() -> None:
    """Retain the direct identity path when no previous object patches are live."""

    diffusion_model = installed_meta_anima()
    patcher = _SourceVisiblePatcher(diffusion_model)
    discovery = _RecordingDiscovery()

    surface = AnimaModelPatcherSurfaceResolver(discovery).resolve(patcher)

    assert discovery.roots == [diffusion_model]
    assert surface.diffusion_model is diffusion_model


@pytest.mark.parametrize(
    "model",
    (object(), SimpleNamespace(model=object())),
    ids=("no-getter", "no-diffusion-model"),
)
def test_resolver_rejects_missing_model_patcher_surface(model: object) -> None:
    """Fail closed when the supplied MODEL omits required host boundaries."""

    with pytest.raises(TypeError, match="MODEL|get_model_object|diffusion_model"):
        AnimaModelPatcherSurfaceResolver().resolve(model)


def test_resolver_rejects_foreign_source_visible_block() -> None:
    """Delegate malformed backup objects to exact installed-surface validation."""

    diffusion_model = installed_meta_anima()
    patcher = _SourceVisiblePatcher(diffusion_model)
    patcher._objects["blocks.0"] = nn.Identity()

    with pytest.raises(AnimaModuleSurfaceError, match="blocks.0"):
        AnimaModelPatcherSurfaceResolver(AnimaModuleSurfaceDiscovery()).resolve(patcher)
