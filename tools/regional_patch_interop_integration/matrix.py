# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the closed managed P9.7 modifier interoperability matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tools.anima_attention_coupling_prompts import PINNED_ADAPTER_A, RegionalLora

STEPS = 8
CFG = 4.0
FEATHER = 0
FULL_WIDTH = 1024
FULL_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
MASK_CASE_ID = "vertical-hard-50-50"


class PatchInteropModelFamily(StrEnum):
    """Identify the managed model family used by one graph."""

    ANIMA = "anima"
    SDXL = "sdxl"


class PatchInteropSpatialMode(StrEnum):
    """Identify the public sampler and latent traversal for one case."""

    FULL = "full"
    TILED = "tiled-1.5x"
    CONTEXTUAL = "contextual-1.5x"


class PatchInteropModifier(StrEnum):
    """Identify one public upstream MODEL modifier graph."""

    NONE = "none"
    EASYCACHE = "easycache"
    LAZYCACHE = "lazycache"
    OPTIMIZED_ATTENTION = "optimized-attention"
    NEGPIP = "negpip"


class PatchInteropOutcome(StrEnum):
    """Identify whether the public sampler must execute or reject."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RegionalPatchInteropCase:
    """Describe one exact public modifier and sampler interaction."""

    case_id: str
    label: str
    model_family: PatchInteropModelFamily
    spatial_mode: PatchInteropSpatialMode
    modifier: PatchInteropModifier
    outcome: PatchInteropOutcome
    scheduled_regional_lora: bool = False
    expected_error_fragments: tuple[str, ...] = ()
    cfg: float = CFG
    feather: int = FEATHER

    def __post_init__(self) -> None:
        """Require coherent success and rejection declarations."""

        if not self.case_id or not self.label:
            raise ValueError("P9.7 cases require identities and labels.")
        if self.outcome is PatchInteropOutcome.ACCEPTED:
            if self.expected_error_fragments:
                raise ValueError("Accepted P9.7 cases cannot expect errors.")
        elif not self.expected_error_fragments:
            raise ValueError("Rejected P9.7 cases require error fragments.")
        if (
            self.model_family is PatchInteropModelFamily.SDXL
            and self.spatial_mode is not PatchInteropSpatialMode.FULL
        ):
            raise ValueError("P9.7 SDXL modifier proof is full-context only.")

    @property
    def expect_success(self) -> bool:
        """Report whether the sampler must publish an image."""

        return self.outcome is PatchInteropOutcome.ACCEPTED

    @property
    def regional_loras(self) -> tuple[RegionalLora, ...]:
        """Return the exact static or scheduled regional ADAPTER_A declaration."""

        schedule = (0.25, 0.75) if self.scheduled_regional_lora else None
        return (RegionalLora(0, PINNED_ADAPTER_A, 0.75, schedule),)

    @property
    def image_size(self) -> tuple[int, int]:
        """Return the terminal image dimensions for accepted cases."""

        if self.spatial_mode is PatchInteropSpatialMode.FULL:
            return FULL_WIDTH, FULL_HEIGHT
        return TARGET_WIDTH, TARGET_HEIGHT


def cases() -> tuple[RegionalPatchInteropCase, ...]:
    """Return accepted and rejected cases in authoritative evidence order."""

    negpip_error = (
        "does not support NegPiP",
        "ordinary conditioning batch",
        "regional branch batch",
    )
    return (
        _accepted("anima-full-baseline", "Anima full static ADAPTER_A baseline"),
        _accepted(
            "anima-full-easycache",
            "Anima full static ADAPTER_A with core EasyCache",
            modifier=PatchInteropModifier.EASYCACHE,
        ),
        RegionalPatchInteropCase(
            "anima-full-lazycache-rejected",
            "Reject LazyCache because denoiser reuse violates exact LoRA execution",
            PatchInteropModelFamily.ANIMA,
            PatchInteropSpatialMode.FULL,
            PatchInteropModifier.LAZYCACHE,
            PatchInteropOutcome.REJECTED,
            False,
            (
                "does not support LazyCache",
                "complete denoiser evaluations",
                "exact regional conditioning or LoRA math",
            ),
        ),
        _accepted(
            "anima-full-optimized-attention",
            "Anima full static ADAPTER_A with optimized attention override",
            modifier=PatchInteropModifier.OPTIMIZED_ATTENTION,
        ),
        _accepted(
            "anima-tiled-optimized-attention",
            "Anima optimized-attention 1024 to 1536 tiled refinement",
            modifier=PatchInteropModifier.OPTIMIZED_ATTENTION,
            spatial_mode=PatchInteropSpatialMode.TILED,
        ),
        _accepted(
            "anima-contextual-optimized-attention",
            "Anima optimized-attention 1024 to 1536 Contextual refinement",
            modifier=PatchInteropModifier.OPTIMIZED_ATTENTION,
            spatial_mode=PatchInteropSpatialMode.CONTEXTUAL,
        ),
        RegionalPatchInteropCase(
            "anima-full-scheduled-easycache-rejected",
            "Reject EasyCache with scheduled regional ADAPTER_A",
            PatchInteropModelFamily.ANIMA,
            PatchInteropSpatialMode.FULL,
            PatchInteropModifier.EASYCACHE,
            PatchInteropOutcome.REJECTED,
            True,
            ("easycache", "scheduled regional execution"),
        ),
        RegionalPatchInteropCase(
            "anima-tiled-easycache-rejected",
            "Reject EasyCache for 1024 to 1536 tiled regional refinement",
            PatchInteropModelFamily.ANIMA,
            PatchInteropSpatialMode.TILED,
            PatchInteropModifier.EASYCACHE,
            PatchInteropOutcome.REJECTED,
            False,
            ("easycache", "tiled spatial views", "view coordinates"),
        ),
        RegionalPatchInteropCase(
            "anima-contextual-easycache-rejected",
            "Reject EasyCache for 1024 to 1536 Contextual regional refinement",
            PatchInteropModelFamily.ANIMA,
            PatchInteropSpatialMode.CONTEXTUAL,
            PatchInteropModifier.EASYCACHE,
            PatchInteropOutcome.REJECTED,
            False,
            ("easycache", "Contextual spatial views", "view coordinates"),
        ),
        RegionalPatchInteropCase(
            "anima-negpip-rejected",
            "Reject Anima NegPiP before regional branch packing",
            PatchInteropModelFamily.ANIMA,
            PatchInteropSpatialMode.FULL,
            PatchInteropModifier.NEGPIP,
            PatchInteropOutcome.REJECTED,
            False,
            negpip_error,
        ),
        RegionalPatchInteropCase(
            "sdxl-negpip-rejected",
            "Reject SDXL NegPiP before paired regional attention patches",
            PatchInteropModelFamily.SDXL,
            PatchInteropSpatialMode.FULL,
            PatchInteropModifier.NEGPIP,
            PatchInteropOutcome.REJECTED,
            False,
            negpip_error,
        ),
    )


def _accepted(
    case_id: str,
    label: str,
    *,
    modifier: PatchInteropModifier = PatchInteropModifier.NONE,
    spatial_mode: PatchInteropSpatialMode = PatchInteropSpatialMode.FULL,
) -> RegionalPatchInteropCase:
    """Return one accepted Anima static-ADAPTER_A case."""

    return RegionalPatchInteropCase(
        case_id,
        label,
        PatchInteropModelFamily.ANIMA,
        spatial_mode,
        modifier,
        PatchInteropOutcome.ACCEPTED,
    )
