# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test observation-only selected-token attention capture."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionCaptureProfile,
    AttentionRegionControls,
    AttentionRegionRequest,
    AttentionRegionRequestKind,
)
from simple_syrup.domain.attention_region_maps import (
    AttentionTokenCatalog,
    AttentionTokenSpan,
    OpenVocabularyContext,
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_region_capture import AttentionRegionCaptureSession
from simple_syrup.runtime.attention_region_capture_backend import (
    ATTENTION_REGION_CAPTURE_BACKEND,
    OptimizedAttentionCaptureOverride,
)
from simple_syrup.runtime.attention_region_open_vocabulary import (
    OpenVocabularyKeyProjector,
)


def test_backend_clones_model_and_composes_existing_attention_override() -> None:
    """Keep source options intact while preserving an upstream override."""

    source = _patcher()

    def previous(
        original: object,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return a sentinel from an admitted upstream override."""

        del original, args, kwargs
        return torch.tensor(3.0)

    source.model_options["transformer_options"]["optimized_attention_override"] = (
        previous
    )

    derived: Any = ATTENTION_REGION_CAPTURE_BACKEND.derive(source, _session())

    assert derived.parent is source
    assert (
        source.model_options["transformer_options"]["optimized_attention_override"]
        is previous
    )
    installed = derived.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert isinstance(installed, OptimizedAttentionCaptureOverride)
    assert (
        installed(
            lambda *_args, **_kwargs: torch.tensor(1.0),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            1,
        ).item()
        == 3.0
    )


def test_open_vocabulary_query_projects_side_keys_without_changing_native_keys() -> (
    None
):
    """Reuse SDXL spatial queries for an absent phrase without conditioning edits."""

    controls = AttentionRegionControls(
        0.0,
        1.0,
        0.3,
        0.2,
        0.5,
        1,
        AttentionCaptureProfile.EXHAUSTIVE,
    )
    request = AttentionRegionRequest(
        "search",
        AttentionRegionRequestKind.CONCEPT_SEGS,
        ("head",),
        controls,
    )
    plan = AttentionCapturePlan(
        "sampler",
        "sampler",
        "model",
        ("model", 0),
        ("positive", 0),
        (request,),
        "1girl",
        ("loader", 1),
    )
    context = OpenVocabularyContext(
        "head",
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]),
        (1,),
    )
    session = AttentionRegionCaptureSession(
        plan=plan,
        model_family=RegionalModelFamily.STANDARD_UNET,
        token_catalog=AttentionTokenCatalog(
            3, (AttentionTokenSpan("1girl", 1, (1,)),), (1, 2, 3)
        ),
        request_spans={"search": ()},
        open_vocabulary_contexts=(context,),
    )
    projector = OpenVocabularyKeyProjector(torch.nn.Identity(), (context,), session)
    native_context = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])

    native_keys = projector(native_context)
    session.observe(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        native_keys,
        native_context,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert torch.equal(native_keys, native_context)
    maps = session.maps_for("search")
    assert len(maps) == 1
    assert maps[0].label == "head"
    assert maps[0].values[0].item() > maps[0].values[1].item()


def test_open_vocabulary_query_projection_is_cached_per_device_and_dtype() -> None:
    """Avoid repeating absent-query key projections at every denoising call."""

    class CountingProjection(torch.nn.Module):
        """Count native and side-projection calls while preserving values."""

        calls: int

        def __init__(self) -> None:
            """Initialize an unused projection counter."""

            super().__init__()
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Return the input after recording the projection."""

            self.calls += 1
            return value

    context = OpenVocabularyContext(
        "head",
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]),
        (1,),
    )
    projection = CountingProjection()
    projector = OpenVocabularyKeyProjector(projection, (context,), _session())
    native_context = torch.ones(1, 3, 2)

    assert torch.equal(projector(native_context), native_context)
    assert torch.equal(projector(native_context), native_context)
    assert projection.calls == 3


def test_sampled_denominator_bounds_unusually_strong_selected_keys() -> None:
    """Keep fast-profile maps finite when denominator sampling misses the peak key."""

    session = _session(
        profile=AttentionCaptureProfile.FAST,
        sequence_length=40,
    )
    query = torch.tensor([[[1000.0, 0.0], [0.0, 1.0]]])
    key = torch.zeros(1, 40, 2)
    key[0, 1, 0] = 1000.0

    session.observe(
        query,
        key,
        key,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    maps = session.maps_for("search")
    assert len(maps) == 1
    assert torch.isfinite(maps[0].values).all().item()
    assert maps[0].values.max().item() <= 1.0


def test_graph_source_aspect_orients_dit_geometry_without_transformer_metadata() -> (
    None
):
    """Use the selected sampler's portrait input when a DiT omits shape metadata."""

    session = _session(source_aspect=0.75)

    session.observe(
        torch.ones(1, 12, 2),
        torch.ones(1, 3, 2),
        torch.ones(1, 3, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    attention_map = session.maps_for("search")[0]
    assert (attention_map.spatial_height, attention_map.spatial_width) == (4, 3)


def test_anima_without_graph_or_runtime_geometry_fails_closed() -> None:
    """Return no maps instead of guessing an unprojectable Anima grid orientation."""

    session = _session(
        sequence_length=512,
        model_family=RegionalModelFamily.ANIMA,
    )

    session.observe(
        torch.ones(1, 12, 2),
        torch.ones(1, 512, 2),
        torch.ones(1, 512, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert session.maps_for("search") == ()
    assert "not graph-visible" in session.status_message


def _session(
    profile: AttentionCaptureProfile = AttentionCaptureProfile.EXHAUSTIVE,
    sequence_length: int = 3,
    source_aspect: float | None = None,
    model_family: RegionalModelFamily = RegionalModelFamily.STANDARD_UNET,
    token_indices: tuple[int, ...] = (1,),
    catalog_spans: tuple[AttentionTokenSpan, ...] | None = None,
) -> AttentionRegionCaptureSession:
    """Return one exact-native-query capture session."""

    controls = AttentionRegionControls(0.0, 1.0, 0.3, 0.2, 0.5, 1, profile)
    request = AttentionRegionRequest(
        "search",
        AttentionRegionRequestKind.CONCEPT_SEGS,
        ("pink hair",),
        controls,
    )
    plan = AttentionCapturePlan(
        "sampler",
        "sampler",
        "model",
        ("model", 0),
        ("positive", 0),
        (request,),
        "pink hair",
        ("loader", 1),
        source_aspect,
    )
    default_span = AttentionTokenSpan("pink hair", 1, token_indices)
    catalog = AttentionTokenCatalog(
        sequence_length,
        catalog_spans or (default_span,),
        tuple(range(sequence_length)),
    )
    return AttentionRegionCaptureSession(
        plan=plan,
        model_family=model_family,
        token_catalog=catalog,
        request_spans={"search": (catalog.spans[0],)},
    )


def _options(
    *,
    cond_or_uncond: list[int],
    sigma: float = 1.0,
    block_index: int = 0,
) -> dict[str, object]:
    """Return exact sampler metadata at the beginning of denoising."""

    return {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "sigmas": torch.tensor([sigma]),
        "cond_or_uncond": cond_or_uncond,
        "block": ("middle", 0),
        "block_index": block_index,
    }


def _patcher() -> Any:
    """Create a real CPU ModelPatcher with isolated transformer options."""

    from comfy.model_patcher import ModelPatcher

    base_model = torch.nn.Module()
    base_model.diffusion_model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    return ModelPatcher(base_model, load_device=device, offload_device=device)
