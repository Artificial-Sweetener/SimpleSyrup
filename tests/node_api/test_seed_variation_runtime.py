# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for seed variation at ComfyUI's outer sampling boundary."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from comfy.patcher_extension import WrappersMP

from simple_syrup.domain.seed_variation import SeedVariationSettings
from simple_syrup.runtime.seed_variation import (
    SEED_VARIATION_WRAPPER_KEY,
    SeedVariationModelPatchBackend,
    SeedVariationNoiseInterpolator,
    SeedVariationOuterSampleWrapper,
)


def test_strength_zero_and_disabled_noise_are_exact_passthroughs() -> None:
    """Avoid allocation and preserve explicit sampler noise-disable behavior."""

    interpolator = SeedVariationNoiseInterpolator()
    base_noise = torch.randn((2, 4, 8, 8), generator=_generator(10))
    disabled_noise = torch.zeros_like(base_noise)

    unchanged = interpolator.interpolate(
        base_noise,
        SeedVariationSettings(variation_seed=20, strength=0.0),
    )
    disabled = interpolator.interpolate(
        disabled_noise,
        SeedVariationSettings(variation_seed=20, strength=1.0),
    )

    assert unchanged is base_noise
    assert disabled is disabled_noise


def test_strength_one_matches_comfy_noise_without_global_rng_mutation() -> None:
    """Use one local CPU generator and make the variation endpoint exact."""

    base_noise = torch.randn((2, 4, 3, 5), generator=_generator(11))
    expected = torch.randn(
        base_noise.shape,
        dtype=torch.float32,
        generator=_generator(29),
    )
    torch.manual_seed(101)
    expected_global_next = torch.randn((3,))
    torch.manual_seed(101)

    result = SeedVariationNoiseInterpolator().interpolate(
        base_noise,
        SeedVariationSettings(variation_seed=29, strength=1.0),
    )
    actual_global_next = torch.randn((3,))

    assert torch.equal(result, expected)
    assert torch.equal(actual_global_next, expected_global_next)


def test_interpolation_is_deterministic_per_batch_item_for_video_noise() -> None:
    """Preserve shape and dtype while applying canonical per-item SLERP."""

    base_noise = torch.randn(
        (2, 4, 3, 5, 7),
        dtype=torch.float64,
        generator=_generator(37),
    )
    settings = SeedVariationSettings(variation_seed=41, strength=0.35)
    interpolator = SeedVariationNoiseInterpolator()

    first = interpolator.interpolate(base_noise, settings)
    second = interpolator.interpolate(base_noise, settings)
    variation = torch.randn(
        base_noise.shape,
        dtype=torch.float32,
        generator=_generator(41),
    ).to(dtype=base_noise.dtype)
    expected = _reference_slerp(base_noise, variation, settings.strength)

    assert first.shape == base_noise.shape
    assert first.dtype == base_noise.dtype
    assert torch.equal(first, second)
    assert torch.allclose(first, expected, atol=1e-6, rtol=1e-6)
    assert not torch.equal(first[0], first[1])


def test_identical_noise_uses_stable_linear_fallback() -> None:
    """Avoid undefined spherical division for collinear seeded tensors."""

    variation_seed = 53
    base_noise = torch.randn(
        (1, 4, 6, 6),
        dtype=torch.float32,
        generator=_generator(variation_seed),
    )

    result = SeedVariationNoiseInterpolator().interpolate(
        base_noise,
        SeedVariationSettings(variation_seed=variation_seed, strength=0.5),
    )

    assert torch.all(torch.isfinite(result))
    assert torch.allclose(result, base_noise)


@pytest.mark.parametrize(
    ("noise", "message"),
    [
        (torch.ones((1, 2), dtype=torch.int64), "floating-point"),
        (torch.ones((4,), dtype=torch.float32), "batched"),
    ],
)
def test_interpolator_rejects_unsupported_noise_tensors(
    noise: torch.Tensor,
    message: str,
) -> None:
    """Fail clearly before unsupported sampler noise enters interpolation."""

    with pytest.raises((TypeError, ValueError), match=message):
        SeedVariationNoiseInterpolator().interpolate(
            noise,
            SeedVariationSettings(variation_seed=1, strength=0.5),
        )


def test_outer_sample_wrapper_changes_only_noise_argument() -> None:
    """Forward sampler, latent, callbacks, and base seed without modification."""

    base_noise = torch.randn((1, 4, 4, 4), generator=_generator(61))
    latent = torch.zeros_like(base_noise)
    sampler = object()
    sigmas = torch.tensor([1.0, 0.0])
    callback = object()
    calls: list[tuple[torch.Tensor, tuple[object, ...], dict[str, object]]] = []

    def executor(
        noise: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> str:
        """Record wrapper forwarding and return a stable sentinel."""

        calls.append((noise, args, kwargs))
        return "sampled"

    wrapper = SeedVariationOuterSampleWrapper(
        SeedVariationSettings(variation_seed=67, strength=1.0)
    )
    result = wrapper(
        executor,
        base_noise,
        latent,
        sampler,
        sigmas,
        None,
        callback,
        False,
        1234,
        latent_shapes=[tuple(latent.shape)],
    )

    assert result == "sampled"
    assert len(calls) == 1
    varied_noise, forwarded_args, forwarded_kwargs = calls[0]
    assert torch.equal(
        varied_noise,
        torch.randn(base_noise.shape, generator=_generator(67)),
    )
    assert forwarded_args == (
        latent,
        sampler,
        sigmas,
        None,
        callback,
        False,
        1234,
    )
    assert forwarded_kwargs == {"latent_shapes": [tuple(latent.shape)]}


def test_model_patch_backend_uses_one_collision_safe_outer_wrapper() -> None:
    """Preserve source state and direct lineage through the native wrapper surface."""

    source = _patcher()
    backend = SeedVariationModelPatchBackend()
    settings = SeedVariationSettings(variation_seed=71, strength=0.4)

    derived = cast(Any, backend.derive(source, settings))

    assert derived is not source
    assert derived.parent is source
    assert (
        source.get_wrappers(WrappersMP.OUTER_SAMPLE, SEED_VARIATION_WRAPPER_KEY) == []
    )
    wrappers = derived.get_wrappers(
        WrappersMP.OUTER_SAMPLE,
        SEED_VARIATION_WRAPPER_KEY,
    )
    assert len(wrappers) == 1
    assert isinstance(wrappers[0], SeedVariationOuterSampleWrapper)
    assert wrappers[0].settings == settings


def test_model_patch_backend_returns_source_when_variation_is_disabled() -> None:
    """Avoid a redundant MODEL clone when strength makes the operation a no-op."""

    source = _patcher()

    result = cast(
        Any,
        SeedVariationModelPatchBackend().derive(
            source,
            SeedVariationSettings(variation_seed=79, strength=0.0),
        ),
    )

    assert result is source
    assert source.wrappers == {}


def _generator(seed: int) -> torch.Generator:
    """Return an isolated CPU generator for deterministic expectations."""

    return torch.Generator(device="cpu").manual_seed(seed)


def _reference_slerp(
    base_noise: torch.Tensor,
    variation_noise: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Calculate independent canonical batch SLERP for behavior comparison."""

    outputs: list[torch.Tensor] = []
    for base_item, variation_item in zip(base_noise, variation_noise, strict=True):
        base_flat = base_item.float().flatten()
        variation_flat = variation_item.float().flatten()
        cosine = torch.dot(
            base_flat / torch.linalg.vector_norm(base_flat),
            variation_flat / torch.linalg.vector_norm(variation_flat),
        ).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        sine = torch.sin(angle)
        mixed = (
            torch.sin((1.0 - strength) * angle) / sine * base_flat
            + torch.sin(strength * angle) / sine * variation_flat
        )
        outputs.append(mixed.reshape_as(base_item).to(dtype=base_item.dtype))
    return torch.stack(outputs)


def _patcher() -> Any:
    """Create a real CPU Comfy MODEL patcher for wrapper integration."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(
        torch.nn.Linear(1, 1),
        load_device=device,
        offload_device=device,
    )
