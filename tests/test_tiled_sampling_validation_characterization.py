"""Characterize validation imported from the mixed tiled-sampling runtime."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    RegionalCapabilityAdmission,
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.runtime.tiled_sampling_validation import (
    reject_unsupported_conditioning,
    validate_latent_samples,
    validate_sampling_controls,
    validate_tensor_shape,
)


def test_sampling_controls_accept_all_boundary_values() -> None:
    """Preserve the inclusive denoise range and minimum integer controls."""

    for denoise in (0.0, 1.0):
        validate_sampling_controls(
            steps=1,
            denoise=denoise,
            latent_tile_width=4,
            latent_tile_height=4,
            latent_tile_batch_size=1,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"steps": 0}, "steps must be at least 1"),
        ({"denoise": -0.001}, "denoise must be between 0 and 1"),
        ({"denoise": 1.001}, "denoise must be between 0 and 1"),
        ({"latent_tile_width": 3}, "latent_tile_width must be at least 4"),
        ({"latent_tile_height": 3}, "latent_tile_height must be at least 4"),
        ({"latent_tile_batch_size": 0}, "latent_tile_batch_size must be at least 1"),
    ],
)
def test_sampling_controls_preserve_each_failure_boundary(
    override: dict[str, int | float], message: str
) -> None:
    """Fix each current control failure and its actionable field name."""

    controls: dict[str, int | float] = {
        "steps": 1,
        "denoise": 1.0,
        "latent_tile_width": 4,
        "latent_tile_height": 4,
        "latent_tile_batch_size": 1,
    }
    controls.update(override)

    with pytest.raises(ValueError, match=message):
        validate_sampling_controls(
            steps=int(controls["steps"]),
            denoise=float(controls["denoise"]),
            latent_tile_width=int(controls["latent_tile_width"]),
            latent_tile_height=int(controls["latent_tile_height"]),
            latent_tile_batch_size=int(controls["latent_tile_batch_size"]),
        )


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param(torch.zeros((2, 4, 8, 12)), id="bchw"),
        pytest.param(torch.zeros((2, 16, 1, 8, 12)), id="bcdhw-singleton-depth"),
    ],
)
def test_latent_validation_returns_supported_tensor_identity(
    samples: torch.Tensor,
) -> None:
    """Return the exact input tensor for both supported latent layouts."""

    assert (
        validate_latent_samples({"samples": samples}, sampler_label="Sampler")
        is samples
    )


@pytest.mark.parametrize("latent", [{}, {"samples": None}, {"samples": "tensor"}])
def test_latent_validation_rejects_missing_or_dynamic_samples(
    latent: dict[str, object],
) -> None:
    """Fail before shape access when the latent payload is not a tensor."""

    with pytest.raises(ValueError, match="latent samples must be a torch tensor"):
        validate_latent_samples(latent, sampler_label="Sampler")


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 8, 8), "requires latent samples shaped"),
        ((1, 4, 1, 1, 8, 8), "requires latent samples shaped"),
        ((1, 16, 2, 8, 8), "singleton third axis"),
    ],
)
def test_tensor_shape_rejects_every_unsupported_rank_or_depth(
    shape: tuple[int, ...], message: str
) -> None:
    """Preserve rank and singleton-depth admission with sampler context."""

    with pytest.raises(ValueError, match=message) as captured:
        validate_tensor_shape(torch.zeros(shape), sampler_label="Named Sampler")

    assert "Named Sampler" in str(captured.value)


def test_tensor_shape_rejects_nested_tensors_before_rank_policy() -> None:
    """Preserve the explicit non-nested invariant and sampler label."""

    with pytest.warns(UserWarning, match="nested tensors.*prototype stage"):
        samples = torch.nested.nested_tensor([torch.zeros((4, 8, 8))])

    with pytest.raises(ValueError, match="Named Sampler requires non-nested"):
        validate_tensor_shape(samples, sampler_label="Named Sampler")


@pytest.mark.parametrize("key", ["area", "control", "gligen"])
def test_conditioning_rejection_finds_each_unsupported_key_recursively(
    key: str,
) -> None:
    """Detect every current unsupported key through tuple/list/dict nesting."""

    conditioning = ([{"outer": ({key: object()},)}],)

    with pytest.raises(ValueError, match="Named Sampler does not support"):
        reject_unsupported_conditioning(conditioning, sampler_label="Named Sampler")


def test_conditioning_rejection_admits_supported_nonspatial_values() -> None:
    """Leave ordinary tensors, metadata, and hooks admitted."""

    conditioning = [
        [
            torch.ones((1, 2, 3)),
            {
                "hooks": object(),
                "strength": 0.5,
                "nested": {"values": (torch.ones((1,)), "metadata")},
            },
        ]
    ]

    reject_unsupported_conditioning(conditioning, sampler_label="Named Sampler")


@pytest.mark.parametrize(
    ("metadata", "admit_masks", "raises"),
    [
        ({"mask": torch.ones((1, 2, 2))}, False, True),
        ({"mask": torch.ones((1, 2, 2))}, True, True),
        (
            {"mask": torch.ones((1, 2, 2)), "set_area_to_bounds": True},
            True,
            True,
        ),
        (
            {"mask": torch.ones((1, 2, 2)), "set_area_to_bounds": False},
            False,
            True,
        ),
        (
            {"mask": torch.ones((1, 2, 2)), "set_area_to_bounds": False},
            True,
            False,
        ),
    ],
)
def test_full_context_mask_admission_requires_both_explicit_conditions(
    metadata: dict[str, object], admit_masks: bool, raises: bool
) -> None:
    """Admit only explicitly full-context masks on enabled runtime paths."""

    def operation() -> None:
        """Execute the selected mask-admission case."""

        reject_unsupported_conditioning(
            [[torch.ones((1, 2, 3)), metadata]],
            sampler_label="Named Sampler",
            capability_admission=_mask_admission(admit_masks),
        )

    if raises:
        with pytest.raises(ValueError, match="Named Sampler does not support"):
            operation()
    else:
        operation()


def _mask_admission(admit_masks: bool) -> RegionalCapabilityAdmission:
    """Return empty or successful full-context mask admission for one case."""

    if not admit_masks:
        return EMPTY_REGIONAL_CAPABILITY_ADMISSION
    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
    )
    return RegionalCapabilityAdmission(request, request.features, None)
