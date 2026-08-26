# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate downstream attention results for IMAGE and LATENT public nodes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_maps import CapturedAttentionMap
from ..domain.segs import ImpactSegs, to_impact_compatible_segs
from ..runtime.attention_region_status import ATTENTION_REGION_STATUS_PUBLISHER
from ..runtime.attention_region_store import (
    ATTENTION_REGION_CAPTURE_STORE,
    AttentionCaptureSession,
)
from .attention_region_rendering import ATTENTION_REGION_RENDERING_SERVICE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AttentionImageResult:
    """Return an unchanged image, batch SEGS, and batch mask."""

    image: torch.Tensor
    segs: list[ImpactSegs]
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class AttentionLatentResult:
    """Return an unchanged latent and attention-derived mask."""

    latent: dict[str, object]
    mask: torch.Tensor


class AttentionRegionNodeService:
    """Consume one prompt-scoped session and render public node outputs."""

    def for_image(
        self,
        *,
        request_node_id: str,
        image: object,
        controls: AttentionRegionControls,
    ) -> AttentionImageResult:
        """Render one SEGS payload per image batch member."""

        image_batch = _image_batch(image)
        session, maps = _consume_maps(request_node_id)
        height = int(image_batch.shape[1])
        width = int(image_batch.shape[2])
        segs: list[ImpactSegs] = []
        masks: list[torch.Tensor] = []
        for batch_index in range(int(image_batch.shape[0])):
            native, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
                maps=maps,
                controls=controls,
                height=height,
                width=width,
                image=image_batch[batch_index : batch_index + 1],
                batch_index=batch_index,
            )
            segs.append(to_impact_compatible_segs(native))
            masks.append(mask)
        _publish_status(request_node_id, session.status_message, maps)
        return AttentionImageResult(image_batch, segs, torch.cat(masks, dim=0))

    def for_latent(
        self,
        *,
        request_node_id: str,
        latent: object,
        controls: AttentionRegionControls,
    ) -> AttentionLatentResult:
        """Render one latent-resolution mask per latent batch member."""

        latent_value, samples = _latent_samples(latent)
        session, maps = _consume_maps(request_node_id)
        height = int(samples.shape[-2])
        width = int(samples.shape[-1])
        masks = tuple(
            ATTENTION_REGION_RENDERING_SERVICE.render(
                maps=maps,
                controls=controls,
                height=height,
                width=width,
                batch_index=batch_index,
            )[1]
            for batch_index in range(int(samples.shape[0]))
        )
        mask = torch.cat(masks, dim=0)
        _publish_status(request_node_id, session.status_message, maps)
        return AttentionLatentResult(latent_value, mask)

    def mask_conditioning(
        self,
        conditioning: object,
        mask: torch.Tensor,
        strength: float,
    ) -> object:
        """Attach one reusable mask through Comfy's generic conditioning policy."""

        if isinstance(strength, bool) or not isinstance(strength, int | float):
            raise TypeError("Attention conditioning strength must be numeric.")
        if not 0.0 <= float(strength) <= 10.0:
            raise ValueError("Attention conditioning strength must be within 0..10.")
        node_helpers = import_module("node_helpers")
        setter = getattr(node_helpers, "conditioning_set_values", None)
        if not callable(setter):
            raise RuntimeError("Comfy conditioning mask support is unavailable.")
        return setter(
            conditioning,
            {
                "mask": mask,
                "set_area_to_bounds": False,
                "mask_strength": float(strength),
            },
        )


def _consume_maps(
    request_node_id: str,
) -> tuple[AttentionCaptureSession, tuple[CapturedAttentionMap, ...]]:
    """Consume one request binding or create an observable empty session."""

    session = ATTENTION_REGION_CAPTURE_STORE.consume(request_node_id)
    if session is None:
        from .attention_capture_model_service import EmptyAttentionCaptureSession

        message = "No upstream attention capture was recovered"
        session = EmptyAttentionCaptureSession((request_node_id,), message)
        LOGGER.warning(
            "Attention-region public node executed without capture state",
            extra={"request_node_id": request_node_id},
        )
    return session, session.maps_for(request_node_id)


def _publish_status(
    request_node_id: str,
    status: str,
    maps: tuple[CapturedAttentionMap, ...],
) -> None:
    """Publish capture status with an exact observation count."""

    ATTENTION_REGION_STATUS_PUBLISHER.publish(
        request_node_id,
        f"{status}: {len(maps)} maps",
    )


def _image_batch(value: object) -> torch.Tensor:
    """Return a finite non-empty BHWC image tensor."""

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or int(value.shape[0]) < 1
        or int(value.shape[-1]) not in (1, 3, 4)
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError("Attention SEGS requires a finite non-empty BHWC IMAGE.")
    return value


def _latent_samples(value: object) -> tuple[dict[str, object], torch.Tensor]:
    """Return a latent mapping and supported BCHW or BCTHW sample tensor."""

    if not isinstance(value, dict):
        raise TypeError("Attention Region Mask requires a LATENT dictionary.")
    samples = value.get("samples")
    if (
        not isinstance(samples, torch.Tensor)
        or samples.ndim not in (4, 5)
        or int(samples.shape[0]) < 1
        or (samples.ndim == 5 and int(samples.shape[-3]) != 1)
    ):
        raise ValueError(
            "Attention Region Mask requires BCHW image latents or singleton-frame "
            "BCTHW Anima latents."
        )
    return value, samples


ATTENTION_REGION_NODE_SERVICE = AttentionRegionNodeService()
