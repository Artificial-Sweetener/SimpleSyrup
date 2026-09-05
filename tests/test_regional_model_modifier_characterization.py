# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize installed cache, NegPiP, and attention-modifier surfaces."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
import torch
from comfy.ldm.cosmos.predict2 import Attention
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_easycache import (  # type: ignore[import-not-found]
    EasyCacheHolder,
    EasyCacheNode,
    LazyCacheHolder,
    LazyCacheNode,
    easycache_calc_cond_batch_wrapper,
    easycache_forward_wrapper,
    easycache_sample_wrapper,
    lazycache_predict_noise_wrapper,
)

from simple_syrup.runtime.model_patcher_mutations import (
    ModelDiffusionWrapperMutation,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE

CUSTOM_NODES_ROOT = Path(__file__).resolve().parents[2]


class _CacheFixtureModel(torch.nn.Module):
    """Expose only the host state required by the installed cache nodes."""

    def __init__(self) -> None:
        """Create one parameter and a four-channel latent descriptor."""

        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.latent_format = SimpleNamespace(latent_channels=4)


def test_core_cache_nodes_preserve_exact_state_through_attention_derivation() -> None:
    """Retain installed cache owners while appending one inner wrapper."""

    source = _patcher(_CacheFixtureModel())
    easy = cast(
        ModelPatcher,
        EasyCacheNode.execute(source, 0.2, 0.15, 0.95, False).result[0],
    )
    lazy = cast(
        ModelPatcher,
        LazyCacheNode.execute(source, 0.2, 0.15, 0.95, False).result[0],
    )

    def attention_wrapper(
        executor: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Delegate one characterized inner diffusion wrapper."""

        return executor(*args, **kwargs)

    easy_derived = PATCHER_LIFECYCLE.derive_model(
        easy,
        (
            ModelDiffusionWrapperMutation(
                "simple_syrup.characterization",
                attention_wrapper,
            ),
        ),
        operation="EasyCache interop characterization",
    )
    lazy_derived = PATCHER_LIFECYCLE.derive_model(
        lazy,
        (
            ModelDiffusionWrapperMutation(
                "simple_syrup.characterization",
                attention_wrapper,
            ),
        ),
        operation="LazyCache interop characterization",
    )

    assert source.wrappers == {}
    assert source.model_options == {"transformer_options": {}}
    assert easy.parent is source
    assert lazy.parent is source
    assert isinstance(
        easy.model_options["transformer_options"]["easycache"],
        EasyCacheHolder,
    )
    assert isinstance(
        lazy.model_options["transformer_options"]["easycache"],
        LazyCacheHolder,
    )
    assert easy.get_wrappers(WrappersMP.OUTER_SAMPLE, "easycache") == [
        easycache_sample_wrapper
    ]
    assert easy.get_wrappers(WrappersMP.CALC_COND_BATCH, "easycache") == [
        easycache_calc_cond_batch_wrapper
    ]
    assert easy_derived.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == [
        easycache_forward_wrapper,
        attention_wrapper,
    ]
    assert lazy.get_wrappers(WrappersMP.OUTER_SAMPLE, "lazycache") == [
        easycache_sample_wrapper
    ]
    assert lazy.get_wrappers(WrappersMP.PREDICT_NOISE, "lazycache") == [
        lazycache_predict_noise_wrapper
    ]
    assert lazy_derived.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == [
        attention_wrapper
    ]


@pytest.mark.external_artifact
def test_installed_negpip_mask_broadcasts_one_base_mask_to_regional_branches() -> None:
    """Expose why ordinary-batch NegPiP cannot enter branch-packed attention."""

    negpip = _load_installed_anima_negpip()
    patch = cast(
        Callable[..., dict[str, torch.Tensor | None]],
        negpip.cosmos_attn2_negpip,
    )
    query = torch.zeros((3, 4, 2), dtype=torch.float32)
    key = torch.zeros((3, 5, 2), dtype=torch.float32)
    value = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2)
    base_mask = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0], [1.0]]])

    result = patch(
        query,
        key,
        value,
        extra_options={negpip.NEGPIP_MASK_KEY: base_mask},
    )

    expected = value * base_mask
    patched_value = result["v"]
    assert base_mask.shape[0] == 1
    assert value.shape[0] == 3
    assert isinstance(patched_value, torch.Tensor)
    assert torch.equal(patched_value, expected)
    assert torch.equal(expected[0], value[0] * base_mask[0])
    assert torch.equal(expected[1], value[1] * base_mask[0])
    assert torch.equal(expected[2], value[2] * base_mask[0])


def test_installed_anima_attention_passes_branch_batch_to_attention_override() -> None:
    """Preserve one optimized-attention override after regional branch packing."""

    attention = Attention(
        query_dim=4,
        context_dim=4,
        n_heads=1,
        head_dim=4,
        operations=torch.nn,
    )
    observed: list[tuple[int, ...]] = []

    def override(
        original: Callable[..., torch.Tensor],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Record the projected branch batch and delegate exact attention math."""

        query = args[0]
        assert isinstance(query, torch.Tensor)
        observed.append(tuple(query.shape))
        return original(*args, **kwargs)

    output = attention(
        torch.randn((3, 4, 4)),
        torch.randn((3, 5, 4)),
        transformer_options={"optimized_attention_override": override},
    )

    assert output.shape == (3, 4, 4)
    assert observed == [(3, 1, 4, 4)]


def test_easycache_identity_cannot_distinguish_equal_shaped_spatial_views() -> None:
    """Show the installed UUID/shape state reuses another view's cached delta."""

    holder = EasyCacheHolder(0.2, 0.15, 0.95, 1, False, output_channels=4)
    conditioning_uuid = uuid4()
    first_view = torch.zeros((1, 4, 2, 2))
    second_view = torch.full_like(first_view, 10.0)
    holder.first_cond_uuid = conditioning_uuid
    holder.update_cache_diff(
        torch.full_like(first_view, 5.0),
        first_view,
        [conditioning_uuid],
    )

    assert holder.check_metadata(first_view)
    assert holder.check_metadata(second_view)
    assert holder.can_apply_cache_diff([conditioning_uuid])
    reused = holder.apply_cache_diff(second_view.clone(), [conditioning_uuid])

    torch.testing.assert_close(reused, torch.full_like(second_view, 15.0))


def test_lazycache_identity_cannot_distinguish_inner_regional_schedule_state() -> None:
    """Show the installed tensor state has no regional schedule identity input."""

    holder = LazyCacheHolder(0.2, 0.15, 0.95, 1, False, output_channels=4)
    latent = torch.zeros((1, 4, 2, 2))
    holder.update_cache_diff(torch.full_like(latent, 3.0), latent)

    assert holder.check_metadata(latent)
    assert holder.has_cache_diff()
    reused_after_inner_schedule_change = holder.apply_cache_diff(latent.clone())

    torch.testing.assert_close(
        reused_after_inner_schedule_change,
        torch.full_like(latent, 3.0),
    )


def _patcher(model: torch.nn.Module) -> ModelPatcher:
    """Create one real installed CPU patcher around a focused fixture model."""

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)


def _load_installed_anima_negpip() -> ModuleType:
    """Load the exact installed standalone PPM Anima NegPiP module."""

    path = CUSTOM_NODES_ROOT / "comfyui-ppm" / "src" / "negpip" / "anima_negpip.py"
    spec = importlib.util.spec_from_file_location(
        "simple_syrup_installed_anima_negpip",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Installed Anima NegPiP source could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
