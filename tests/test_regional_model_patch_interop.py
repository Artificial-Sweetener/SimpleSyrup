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
    RegionalAttentionTopology,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from simple_syrup.runtime.ppm_negpip_interop import PpmNegpipSemantics
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

    def extra_conds(self, **_kwargs: object) -> dict[str, object]:
        """Expose the object path patched by Anima NegPiP."""

        return {}


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


def test_validator_reports_generic_model_weight_patches_without_mutation() -> None:
    """Publish populated MODEL weight-patch state without inspecting identity."""

    source = _patcher()
    source.patches["projection.weight"] = [object()]
    before = {key: values.copy() for key, values in source.patches.items()}

    report = REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
        source,
        _capabilities(RegionalModelFamily.STANDARD_UNET),
    )

    assert report.preserved_modifiers == (
        RegionalPreservedModelModifier.MODEL_WEIGHT_PATCH,
    )
    assert source.patches == before


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


def test_validator_admits_exact_standard_unet_negpip_without_mutation() -> None:
    """Retain PPM's exact split-K/V callback as typed interop evidence."""

    model = _patcher()
    callback = _identity_callback(
        "custom_nodes.ComfyUI-ppm.src.negpip.unet_negpip",
        "sdxl_attn2_negpip",
    )
    model.model_options["ppm_negpip"] = True
    model.set_model_attn2_patch(callback)

    report = REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
        model,
        _capabilities(RegionalModelFamily.STANDARD_UNET),
    )

    assert report.negpip is not None
    assert report.negpip.semantics is PpmNegpipSemantics.STANDARD_UNET_SPLIT_KEY_VALUE
    assert report.negpip.attention_patch is callback
    assert model.model_options["transformer_options"]["patches"]["attn2_patch"] == [
        callback
    ]


def test_validator_admits_exact_anima_negpip_without_mutation() -> None:
    """Retain PPM's complete Anima callback, wrapper, and object-patch family."""

    model = _patcher()
    callback = _identity_callback(
        "custom_nodes.ComfyUI-ppm.src.negpip.anima_negpip",
        "cosmos_attn2_negpip",
    )
    wrapper = _identity_callback(
        "custom_nodes.ComfyUI-ppm.src.negpip.anima_negpip",
        "cosmos_diffusion_negpip_wrapper",
    )
    extra_conds = _identity_callback(
        "custom_nodes.ComfyUI-ppm.src.negpip.anima_negpip",
        ("anima_extra_conds_negpip_wrapper.<locals>._anima_extra_conds_negpip_wrapper"),
    )
    model.model_options["ppm_negpip"] = True
    model.set_model_attn2_patch(callback)
    model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "ppm_negpip_anima",
        wrapper,
    )
    model.add_object_patch("extra_conds", extra_conds)

    report = REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR.validate(
        model,
        _capabilities(RegionalModelFamily.ANIMA),
    )

    assert report.negpip is not None
    assert report.negpip.semantics is PpmNegpipSemantics.ANIMA_VALUE_MASK
    assert report.negpip.attention_patch is callback
    assert model.wrappers[WrappersMP.DIFFUSION_MODEL]["ppm_negpip_anima"] == [wrapper]
    assert model.object_patches["extra_conds"] is extra_conds


@pytest.mark.parametrize(
    ("family", "configure", "message"),
    [
        (
            RegionalModelFamily.STANDARD_UNET,
            lambda model: model.model_options.__setitem__("ppm_negpip", True),
            "requires exactly its PPM split-K/V",
        ),
        (
            RegionalModelFamily.STANDARD_UNET,
            lambda model: model.set_model_attn2_patch(
                _identity_callback(
                    "custom_nodes.ComfyUI-ppm.src.negpip.unet_negpip",
                    "sdxl_attn2_negpip",
                )
            ),
            "incomplete NegPiP patch family",
        ),
        (
            RegionalModelFamily.ANIMA,
            lambda model: model.model_options.__setitem__("ppm_negpip", True),
            "requires exactly its PPM attention patch",
        ),
    ],
)
def test_validator_rejects_partial_or_foreign_negpip_families(
    family: RegionalModelFamily,
    configure: Callable[[ModelPatcher], object],
    message: str,
) -> None:
    """Fail closed before partial or identity-foreign NegPiP state is composed."""

    model = _patcher()
    configure(model)

    with pytest.raises(ValueError, match=message):
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


def _identity_callback(module: str, qualname: str) -> Callable[..., object]:
    """Build one executable callback carrying a stable PPM definition identity."""

    def callback(*args: object, **_kwargs: object) -> object:
        """Return callback inputs for model-state-only admission tests."""

        return args

    callback.__module__ = module
    callback.__qualname__ = qualname
    return callback


def _capabilities(family: RegionalModelFamily) -> RegionalModelCapabilities:
    """Build the exact family contract consumed by modifier admission."""

    if family is RegionalModelFamily.ANIMA:
        return RegionalModelCapabilities(
            model_family=family,
            attention_backend=RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
            attention_topology=(
                RegionalAttentionTopology.SINGLETON_FRAME_SPATIOTEMPORAL
            ),
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
        attention_topology=RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT,
        latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
        spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
        control_gligen_policy=RegionalControlGligenPolicy.REJECT,
        reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
        known_patch_conflicts=(
            RegionalPatchConflict.ATTN2_INPUT_PATCH,
            RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
        ),
    )
