# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify centralized MODEL modifier admission for Attention Coupling."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_easycache import (  # type: ignore[import-not-found]
    EasyCacheNode,
    LazyCacheNode,
)

from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from simple_syrup.runtime.regional_model_patch_interop import (
    REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR,
    RegionalPreservedModelModifier,
)


class _FixtureModel(torch.nn.Module):
    """Expose one parameter and the cache node's latent descriptor."""

    def __init__(self) -> None:
        """Initialize focused host state without diffusion execution."""

        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.latent_format = SimpleNamespace(latent_channels=4)


def test_validator_preserves_easycache_and_unrelated_model_state() -> None:
    """Accept EasyCache while retaining every collaborator-owned surface."""

    source = _patcher()

    def model_wrapper(
        apply_model: Callable[..., object],
        args: dict[str, object],
    ) -> object:
        """Delegate one upstream model-function wrapper."""

        return apply_model(args["input"], args["timestep"])

    def diffusion_wrapper(
        executor: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Delegate one upstream diffusion wrapper."""

        return executor(*args, **kwargs)

    def attention_override(
        original: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Delegate one optimized-attention override."""

        return original(*args, **kwargs)

    source.set_model_unet_function_wrapper(model_wrapper)
    source.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "upstream.diffusion",
        diffusion_wrapper,
    )
    source.set_model_post_input_patch(lambda value: value)
    source.add_object_patch("projection", torch.nn.Identity())
    source.model_options["transformer_options"]["optimized_attention_override"] = (
        attention_override
    )
    cached = cast(
        ModelPatcher,
        EasyCacheNode.execute(source, 0.2, 0.15, 0.95, False).result[0],
    )
    before_options = cached.model_options.copy()
    before_wrappers = {
        wrapper_type: {key: callbacks.copy() for key, callbacks in keyed.items()}
        for wrapper_type, keyed in cached.wrappers.items()
    }
    before_objects = cached.object_patches.copy()

    report = REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
        cached,
        _capabilities(RegionalModelFamily.ANIMA),
    )

    assert report.model_family is RegionalModelFamily.ANIMA
    assert set(report.preserved_modifiers) == {
        RegionalPreservedModelModifier.MODEL_FUNCTION_WRAPPER,
        RegionalPreservedModelModifier.DIFFUSION_MODEL_WRAPPER,
        RegionalPreservedModelModifier.OPTIMIZED_ATTENTION_OVERRIDE,
        RegionalPreservedModelModifier.OBJECT_PATCH,
        RegionalPreservedModelModifier.EASYCACHE,
    }
    assert cached.model_options == before_options
    assert cached.wrappers == before_wrappers
    assert cached.object_patches == before_objects


def test_validator_rejects_lazycache_without_mutating_it() -> None:
    """Reject whole-denoiser reuse that cannot execute exact skipped-step LoRA math."""

    cached = cast(
        ModelPatcher,
        LazyCacheNode.execute(_patcher(), 0.2, 0.15, 0.95, False).result[0],
    )
    before_options = cached.model_options.copy()
    before_wrappers = cached.wrappers.copy()

    with pytest.raises(
        ValueError,
        match="does not support LazyCache.*complete denoiser evaluations",
    ):
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            cached,
            _capabilities(RegionalModelFamily.ANIMA),
        )

    assert cached.model_options == before_options
    assert cached.wrappers == before_wrappers


def test_validator_rejects_both_core_caches_without_mutating_them() -> None:
    """Prevent one cache holder from silently serving two incompatible wrappers."""

    easy = cast(
        ModelPatcher,
        EasyCacheNode.execute(_patcher(), 0.2, 0.15, 0.95, False).result[0],
    )
    combined = cast(
        ModelPatcher,
        LazyCacheNode.execute(easy, 0.2, 0.15, 0.95, False).result[0],
    )
    before_options = combined.model_options.copy()
    before_wrappers = combined.wrappers.copy()

    with pytest.raises(
        ValueError,
        match="EasyCache and LazyCache cannot both own one MODEL",
    ):
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            combined,
            _capabilities(RegionalModelFamily.ANIMA),
        )

    assert combined.model_options == before_options
    assert combined.wrappers == before_wrappers


@pytest.mark.parametrize(
    "family",
    [RegionalModelFamily.ANIMA, RegionalModelFamily.STANDARD_UNET],
)
def test_validator_rejects_named_negpip_before_generic_attn2_collision(
    family: RegionalModelFamily,
) -> None:
    """Report the installed modifier and regional mask misalignment by name."""

    model = _patcher()
    model.model_options["ppm_negpip"] = True
    model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "ppm_negpip_anima",
        lambda executor, *args, **kwargs: executor(*args, **kwargs),
    )
    model.set_model_attn2_patch(lambda q, k, v, **kwargs: {"q": q, "k": k, "v": v})

    with pytest.raises(
        ValueError,
        match="NegPiP.*ordinary conditioning batch.*regional branch batch",
    ):
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            model,
            _capabilities(family),
        )


@pytest.mark.parametrize(
    ("family", "patch_names"),
    [
        (RegionalModelFamily.ANIMA, ("attn2_patch",)),
        (
            RegionalModelFamily.STANDARD_UNET,
            ("attn2_patch", "attn2_output_patch"),
        ),
    ],
)
def test_validator_rejects_every_family_owned_attn2_surface(
    family: RegionalModelFamily,
    patch_names: tuple[str, ...],
) -> None:
    """Require exclusive branch-aware attention ownership before derivation."""

    model = _patcher()
    for patch_name in patch_names:
        model.set_model_patch(lambda *args, **kwargs: args, patch_name)

    with pytest.raises(ValueError) as raised:
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            model,
            _capabilities(family),
        )

    assert "existing MODEL attention patches" in str(raised.value)
    assert all(patch_name in str(raised.value) for patch_name in patch_names)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("model_options", None, "model_options must be a dictionary"),
        (
            "model_options",
            {"transformer_options": None},
            "transformer_options must be a dictionary",
        ),
        ("wrappers", None, "wrappers must be a dictionary"),
        ("object_patches", None, "object_patches must be a dictionary"),
    ],
)
def test_validator_rejects_malformed_patcher_state(
    attribute: str,
    value: object,
    message: str,
) -> None:
    """Fail closed before malformed dynamic host state reaches a clone."""

    model = _patcher()
    original = getattr(model, attribute)
    setattr(model, attribute, value)

    try:
        with pytest.raises(TypeError, match=message):
            REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
                model,
                _capabilities(RegionalModelFamily.ANIMA),
            )
    finally:
        setattr(model, attribute, original)


def test_validator_rejects_noncallable_override_and_wrapper_entries() -> None:
    """Do not preserve modifier slots whose installed behavior cannot execute."""

    override_model = _patcher()
    override_model.model_options["transformer_options"][
        "optimized_attention_override"
    ] = object()
    wrapper_model = _patcher()
    wrapper_model.wrappers = {WrappersMP.DIFFUSION_MODEL: {"broken": [object()]}}

    with pytest.raises(
        TypeError, match="optimized_attention_override must be callable"
    ):
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            override_model,
            _capabilities(RegionalModelFamily.ANIMA),
        )
    with pytest.raises(TypeError, match="wrapper callbacks must be callable"):
        REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
            wrapper_model,
            _capabilities(RegionalModelFamily.ANIMA),
        )


def _patcher() -> ModelPatcher:
    """Return one real installed patcher with focused CPU model state."""

    device = torch.device("cpu")
    return ModelPatcher(
        _FixtureModel(),
        load_device=device,
        offload_device=device,
    )


def _capabilities(family: RegionalModelFamily) -> RegionalModelCapabilities:
    """Build the exact family contract consumed by modifier admission."""

    if family is RegionalModelFamily.ANIMA:
        return RegionalModelCapabilities(
            model_family=family,
            attention_backend=RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
            latent_layout=RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW,
            spatial_patch_support=(RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS),
            control_gligen_policy=RegionalControlGligenPolicy.REJECT,
            reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
            known_patch_conflicts=(
                RegionalPatchConflict.DIFFUSION_MODEL_WRAPPER,
                RegionalPatchConflict.CROSS_ATTENTION_OBJECT_PATCH,
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
            ),
        )
    return RegionalModelCapabilities(
        model_family=family,
        attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
        latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
        spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
        control_gligen_policy=RegionalControlGligenPolicy.REJECT,
        reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
        known_patch_conflicts=(
            RegionalPatchConflict.ATTN2_INPUT_PATCH,
            RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
        ),
    )
