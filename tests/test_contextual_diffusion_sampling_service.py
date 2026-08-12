# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for contextual diffusion sampling orchestration."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_features import (
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.services import (
    contextual_diffusion_sampling_service as service_module,
)
from simple_syrup.services.contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingService,
)


def test_service_builds_global_context_and_segs_guided_tile_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected SEGS reach the runtime through the existing guided tile planner."""

    calls: list[dict[str, Any]] = []

    def fake_sample_contextual_diffusion(**kwargs: Any) -> dict[str, Any]:
        """Record the completed plan and return the input latent."""

        calls.append(kwargs)
        latent_image = kwargs["latent_image"]
        if not isinstance(latent_image, dict):
            raise TypeError("Test runtime expected a latent dictionary.")
        return latent_image

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        fake_sample_contextual_diffusion,
    )
    latent = {"samples": torch.zeros((1, 4, 64, 96))}

    result = ContextualDiffusionSamplingService().sample(
        **_sample_kwargs(latent=latent, segs=_segs(512, 768))
    )

    assert torch.equal(result.latent["samples"], latent["samples"])
    assert result.contexts[0] == (512, 768)
    assert len(result.contexts[1]) == len(calls[0]["plan"].tile_plan.tiles)
    assert not result.contexts[1].is_materialized
    assert all(segment.label.startswith("context_") for segment in result.contexts[1])
    assert result.contexts[1].is_materialized
    assert len(calls) == 1
    assert calls[0]["diffusion_mode"] == "mixture_of_diffusers"
    plan = calls[0]["plan"]
    assert (
        plan.global_view.model_width,
        plan.global_view.model_height,
    ) == (64, 44)
    assert len(plan.tile_plan.tiles) > 1
    assert all(tile.weight_mask is not None for tile in plan.tile_plan.tiles)


def test_service_composes_explicit_regional_planning_masks_with_segs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Attention Coupling masks in planning without conditioning re-entry."""

    calls: list[dict[str, Any]] = []

    def fake_sample_contextual_diffusion(**kwargs: Any) -> dict[str, Any]:
        """Record the region-and-SEGS plan and return its latent unchanged."""

        calls.append(kwargs)
        latent_image = kwargs["latent_image"]
        if not isinstance(latent_image, dict):
            raise TypeError("Test runtime expected a latent dictionary.")
        return latent_image

    class FakeCapabilityAdmissionService:
        """Admit the requested feature without requiring a real Comfy model."""

        class Admission:
            """Expose the feature query used by this orchestration test."""

            def __init__(self, request: RegionalFeatureRequest) -> None:
                """Retain the admitted request."""

                self.request = request

            def supports(self, feature: RegionalFeature) -> bool:
                """Return whether the fake admission contains one feature."""

                return feature in self.request.features

        def admit(
            self,
            *,
            request: RegionalFeatureRequest,
            sampler_capabilities: object,
            model: object,
        ) -> Admission:
            """Return an admission containing every requested feature."""

            del sampler_capabilities, model
            return self.Admission(request)

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        fake_sample_contextual_diffusion,
    )
    monkeypatch.setattr(
        ContextualDiffusionSamplingService,
        "capability_admission_service_class",
        FakeCapabilityAdmissionService,
    )
    latent = {"samples": torch.zeros((1, 4, 64, 96))}
    masks = torch.zeros((2, 64, 96))
    masks[0, :, :56] = 1.0
    masks[1, :, 40:] = 1.0

    ContextualDiffusionSamplingService().sample(
        **(
            _sample_kwargs(latent=latent, segs=_segs(512, 768))
            | {
                "positive": "base-positive",
                "negative": "base-negative",
                "planning_region_masks": masks,
                "feature_request": RegionalFeatureRequest(
                    frozenset({RegionalFeature.ATTENTION_COUPLING})
                ),
            }
        )
    )

    assert len(calls) == 1
    assert calls[0]["positive"] == "base-positive"
    assert calls[0]["negative"] == "base-negative"
    assert calls[0]["capability_admission"].supports(RegionalFeature.ATTENTION_COUPLING)
    plan = calls[0]["plan"].tile_plan
    assert len(plan.tiles) > 1
    assert all(tile.weight_mask is not None for tile in plan.tiles)


def test_service_rejects_explicit_and_legacy_regional_planning_masks() -> None:
    """Reject two competing regional tile-planning authorities."""

    masks = torch.ones((1, 64, 96))
    with pytest.raises(ValueError, match="cannot combine explicit planning masks"):
        ContextualDiffusionSamplingService().sample(
            **(
                _sample_kwargs(
                    latent={"samples": torch.zeros((1, 4, 64, 96))},
                    segs=None,
                )
                | {
                    "positive": ConditioningBatch(
                        (_conditioning("global"), _conditioning("region"))
                    ),
                    "negative": _conditioning("negative"),
                    "region_masks": masks,
                    "planning_region_masks": masks,
                }
            )
        )


def test_service_rejects_segs_batch_mismatch_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous per-image semantic guidance fails before model sampling."""

    called = False

    def fake_sample_contextual_diffusion(**kwargs: Any) -> dict[str, Any]:
        """Mark unexpected runtime entry."""

        del kwargs
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        fake_sample_contextual_diffusion,
    )
    latent = {"samples": torch.zeros((2, 4, 64, 96))}

    with pytest.raises(ValueError, match="one SEGS payload or one per latent"):
        ContextualDiffusionSamplingService().sample(
            **_sample_kwargs(
                latent=latent,
                segs=[_segs(512, 768), _segs(512, 768), _segs(512, 768)],
            )
        )

    assert not called


def test_service_selects_conditioning_batch_per_latent_with_shared_segs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contextual SEGS sampling keeps its per-latent conditioning contract."""

    calls: list[dict[str, Any]] = []

    def fake_sample_contextual_diffusion(**kwargs: Any) -> dict[str, Any]:
        """Record one item and return its latent unchanged."""

        calls.append(kwargs)
        latent_image = kwargs["latent_image"]
        if not isinstance(latent_image, dict):
            raise TypeError("Test runtime expected a latent dictionary.")
        return latent_image

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        fake_sample_contextual_diffusion,
    )
    latent = {"samples": torch.zeros((2, 4, 64, 96))}

    ContextualDiffusionSamplingService().sample(
        **(
            _sample_kwargs(latent=latent, segs=_segs(512, 768))
            | {
                "positive": ConditioningBatch(("positive-0", "positive-1")),
                "negative": ConditioningBatch(("negative-0", "negative-1")),
            }
        )
    )

    assert [call["positive"] for call in calls] == ["positive-0", "positive-1"]
    assert [call["negative"] for call in calls] == ["negative-0", "negative-1"]


def test_service_applies_regional_conditioning_to_contextual_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contextual local and global predictions share assembled regional inputs."""

    calls: list[dict[str, Any]] = []

    def fake_sample_contextual_diffusion(**kwargs: Any) -> dict[str, Any]:
        """Record one regional contextual request and return its latent."""

        calls.append(kwargs)
        latent_image = kwargs["latent_image"]
        if not isinstance(latent_image, dict):
            raise TypeError("Test runtime expected a latent dictionary.")
        return latent_image

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        fake_sample_contextual_diffusion,
    )
    masks = torch.zeros((2, 64, 96))
    masks[0, :, :48] = 1.0
    masks[1, :, 48:] = 1.0

    ContextualDiffusionSamplingService().sample(
        **(
            _sample_kwargs(
                latent={
                    "samples": torch.zeros((1, 4, 64, 96)),
                    "downscale_ratio_spacial": 8,
                },
                segs=None,
            )
            | {
                "positive": ConditioningBatch(
                    (
                        _conditioning("global"),
                        _conditioning("left"),
                        _conditioning("right"),
                    )
                ),
                "negative": _conditioning("negative"),
                "region_masks": masks,
            }
        )
    )

    assert len(calls) == 1
    assert [item[0] for item in calls[0]["positive"]] == [
        "global",
        "left",
        "right",
    ]
    assert calls[0]["capability_admission"].supports(
        RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING
    )
    assert len(calls[0]["plan"].tile_plan.tiles) >= 2
    assert all(
        tile.weight_mask is not None for tile in calls[0]["plan"].tile_plan.tiles
    )


def test_service_rejects_invalid_controls_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid context geometry fails without sampler side effects."""

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        lambda **_kwargs: pytest.fail("runtime should not be called"),
    )

    with pytest.raises(ValueError, match="latent_context_overlap"):
        ContextualDiffusionSamplingService().sample(
            **(
                _sample_kwargs(
                    latent={
                        "samples": torch.zeros((1, 4, 64, 96)),
                        "downscale_ratio_spacial": 8,
                    },
                    segs=None,
                )
                | {"latent_context_overlap": 64}
            )
        )


def test_service_rejects_unknown_diffusion_mode_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown tile blending policies fail before model sampling begins."""

    monkeypatch.setattr(
        service_module,
        "sample_contextual_diffusion",
        lambda **_kwargs: pytest.fail("runtime should not be called"),
    )

    with pytest.raises(ValueError, match="diffusion_mode must be one of"):
        ContextualDiffusionSamplingService().sample(
            **(
                _sample_kwargs(
                    latent={"samples": torch.zeros((1, 4, 64, 96))},
                    segs=None,
                )
                | {"diffusion_mode": "unknown"}
            )
        )


def _sample_kwargs(*, latent: dict[str, Any], segs: object | None) -> dict[str, Any]:
    """Return one valid service request."""

    return {
        "model": object(),
        "seed": 1,
        "steps": 4,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": [],
        "negative": [],
        "latent_image": latent,
        "denoise": 0.5,
        "diffusion_mode": "mixture_of_diffusers",
        "latent_context_size": 64,
        "latent_context_overlap": 8,
        "latent_context_batch_size": 2,
        "global_weight": 1.0,
        "global_steps": 1,
        "global_decay": 0.5,
        "segs": segs,
    }


def _segs(height: int, width: int) -> object:
    """Return one full-image Impact-compatible SEG payload."""

    crop = CropRegion(0, 0, width, height)
    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((height, width), dtype=torch.float32),
        confidence=1.0,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="subject",
    )
    return ((height, width), (segment,))


def _conditioning(name: str) -> list[list[object]]:
    """Return one structurally valid standard conditioning value."""

    return [[name, {}]]
