# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact read-only discovery of the installed Anima patch surface."""

from __future__ import annotations

from types import MethodType

import pytest
from anima_module_surface_fixtures import installed_meta_anima
from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block, GPT2FeedForward
from torch import nn

from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurfaceError,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_BLOCK_COUNT,
    AnimaLoraTargetFamily,
    anima_lora_target_name,
    expected_anima_lora_features,
)


@pytest.fixture(scope="module")
def installed_anima() -> nn.Module:
    """Build the real installed Anima graph with allocation-free meta weights."""

    model = installed_meta_anima()
    if not isinstance(model, nn.Module):
        raise AssertionError("Installed Anima must be a torch module")
    return model


def test_discovers_all_installed_blocks_and_lora_targets_read_only(
    installed_anima: nn.Module,
) -> None:
    """Bind all 448 canonical targets without changing graph or global methods."""

    named_modules_before = tuple(installed_anima.named_modules())
    state_keys_before = tuple(installed_anima.state_dict().keys())
    class_forwards_before = (
        Anima.forward,
        Block.forward,
        Attention.forward,
        GPT2FeedForward.forward,
    )

    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)

    assert surface.diffusion_model is installed_anima
    assert len(surface.blocks) == ANIMA_BLOCK_COUNT
    assert len(surface.lora_targets) == 448
    assert [target.target_name for target in surface.lora_targets] == [
        anima_lora_target_name(block_index, family)
        for block_index in range(ANIMA_BLOCK_COUNT)
        for family in AnimaLoraTargetFamily
    ]
    assert tuple(installed_anima.named_modules()) == named_modules_before
    assert tuple(installed_anima.state_dict().keys()) == state_keys_before
    assert (
        Anima.forward,
        Block.forward,
        Attention.forward,
        GPT2FeedForward.forward,
    ) == class_forwards_before


def test_descriptors_retain_exact_installed_owners_and_shapes(
    installed_anima: nn.Module,
) -> None:
    """Retain every block owner and target module by exact object identity."""

    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)

    for block_surface in surface.blocks:
        block = block_surface.block
        assert block_surface.self_attention is block.self_attn
        assert block_surface.cross_attention is block.cross_attn
        assert block_surface.mlp is block.mlp
        assert block_surface.adaln_self_attention is block.adaln_modulation_self_attn
        assert block_surface.adaln_cross_attention is block.adaln_modulation_cross_attn
        assert block_surface.adaln_mlp is block.adaln_modulation_mlp
        assert len(block_surface.lora_targets) == len(AnimaLoraTargetFamily)
        for target in block_surface.lora_targets:
            assert target.module is _resolve_target(block, target.family)
            assert (target.input_features, target.output_features) == (
                expected_anima_lora_features(target.family)
            )
            assert target.forward_signature


def test_rejects_foreign_diffusion_module_before_graph_inspection() -> None:
    """Keep unverified Cosmos variants outside the Anima patch contract."""

    with pytest.raises(AnimaModuleSurfaceError, match="exact installed.*Anima"):
        ANIMA_MODULE_SURFACE_DISCOVERY.discover(nn.Identity())


def test_aggregates_configuration_and_block_count_drift(
    installed_anima: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report every incompatible model-level setting in one pre-patch failure."""

    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)
    monkeypatch.setattr(installed_anima, "model_channels", 1024)
    monkeypatch.setattr(installed_anima, "num_heads", 8)
    monkeypatch.setattr(installed_anima, "use_adaln_lora", False)
    monkeypatch.setattr(
        installed_anima,
        "blocks",
        nn.ModuleList([item.block for item in surface.blocks[:-1]]),
    )

    with pytest.raises(AnimaModuleSurfaceError) as captured:
        ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)

    message = str(captured.value)
    assert "diffusion_model.model_channels" in message
    assert "diffusion_model.num_heads" in message
    assert "diffusion_model.use_adaln_lora" in message
    assert "expected 28 blocks, observed 27" in message


def test_aggregates_block_owner_adaln_and_target_shape_drift(
    installed_anima: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject mixed block owners, incomplete AdaLN paths, and wrong linear shapes."""

    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)
    monkeypatch.setattr(surface.blocks[0].block, "self_attn", nn.Identity())
    monkeypatch.setattr(
        surface.blocks[1].block,
        "adaln_modulation_cross_attn",
        nn.Sequential(nn.SiLU()),
    )
    monkeypatch.setattr(
        surface.blocks[2].block.mlp,
        "layer1",
        nn.Linear(32, 64, bias=False, device="meta"),
    )
    monkeypatch.setattr(
        surface.blocks[3].self_attention,
        "k_proj",
        surface.blocks[3].self_attention.q_proj,
    )

    with pytest.raises(AnimaModuleSurfaceError) as captured:
        ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)

    message = str(captured.value)
    assert "diffusion_model.blocks.0.self_attn" in message
    assert "diffusion_model.blocks.1.adaln_modulation_cross_attn" in message
    assert "diffusion_model.blocks.2.mlp.layer1" in message
    assert "input=2048, output=8192" in message
    assert "diffusion_model.blocks.3.self_attn.k_proj" in message
    assert "aliases the distinct canonical target" in message


def test_aggregates_every_patch_relevant_forward_signature_drift(
    installed_anima: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject forward drift at diffusion, block, attention, MLP, and target layers."""

    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)
    monkeypatch.setattr(
        installed_anima,
        "forward",
        MethodType(_incompatible_forward, installed_anima),
    )
    monkeypatch.setattr(
        surface.blocks[0].block,
        "forward",
        MethodType(_incompatible_forward, surface.blocks[0].block),
    )
    monkeypatch.setattr(
        surface.blocks[1].self_attention,
        "forward",
        MethodType(_incompatible_forward, surface.blocks[1].self_attention),
    )
    monkeypatch.setattr(
        surface.blocks[2].mlp,
        "forward",
        MethodType(_incompatible_forward, surface.blocks[2].mlp),
    )
    monkeypatch.setattr(
        surface.blocks[3].lora_targets[0].module,
        "forward",
        MethodType(_no_input_forward, surface.blocks[3].lora_targets[0].module),
    )

    with pytest.raises(AnimaModuleSurfaceError) as captured:
        ANIMA_MODULE_SURFACE_DISCOVERY.discover(installed_anima)

    message = str(captured.value)
    assert "diffusion_model.forward" in message
    assert "diffusion_model.blocks.0.forward" in message
    assert "diffusion_model.blocks.1.self_attn.forward" in message
    assert "diffusion_model.blocks.2.mlp.forward" in message
    assert "diffusion_model.blocks.3.self_attn.q_proj.forward" in message


def _resolve_target(block: Block, family: AnimaLoraTargetFamily) -> nn.Module:
    """Resolve one expected target independently for descriptor identity checks."""

    current: object = block
    for part in family.value.split("."):
        if part.isdecimal() and isinstance(current, nn.Sequential):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise AssertionError("Anima target did not resolve to a module")
    return current


def _incompatible_forward(self: object, unexpected: object) -> object:
    """Provide an intentionally incompatible bound forward for guard tests."""

    del self
    return unexpected


def _no_input_forward(self: object) -> None:
    """Provide an intentionally input-free target forward for guard tests."""

    del self
