# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable capability values for regional attention backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RegionalModelFamily(StrEnum):
    """Identify a defensively admitted regional model family."""

    ANIMA = "anima"
    STANDARD_UNET = "standard_unet"


class RegionalAttentionBackend(StrEnum):
    """Identify the model-specific attention patch backend."""

    ANIMA_OBJECT_PATCH = "anima_object_patch"
    UNET_ATTN2_PATCH = "unet_attn2_patch"


class RegionalLatentLayout(StrEnum):
    """Identify the latent rank and temporal layout admitted by a backend."""

    ANIMA_SINGLE_FRAME_BCTHW = "anima_single_frame_bcthw"
    STANDARD_IMAGE_BCHW = "standard_image_bchw"


class RegionalSpatialPatchSupport(StrEnum):
    """Report whether one backend can consume canonical spatial views."""

    FULL_AND_SPATIAL_VIEWS = "full_and_spatial_views"


class RegionalControlGligenPolicy(StrEnum):
    """Report control and GLIGEN admission for attention coupling."""

    REJECT = "reject"


class RegionalReferenceLatentPolicy(StrEnum):
    """Report reference-latent admission for attention coupling."""

    REJECT = "reject"


class RegionalPatchConflict(StrEnum):
    """Identify a patch surface that must be collision-free before mutation."""

    DIFFUSION_MODEL_WRAPPER = "diffusion_model_wrapper"
    CROSS_ATTENTION_OBJECT_PATCH = "cross_attention_object_patch"
    ATTN2_INPUT_PATCH = "attn2_input_patch"
    ATTN2_OUTPUT_PATCH = "attn2_output_patch"


@dataclass(frozen=True, slots=True)
class RegionalModelCapabilities:
    """Describe one admitted backend and every relevant compatibility policy."""

    model_family: RegionalModelFamily
    attention_backend: RegionalAttentionBackend
    latent_layout: RegionalLatentLayout
    spatial_patch_support: RegionalSpatialPatchSupport
    control_gligen_policy: RegionalControlGligenPolicy
    reference_latent_policy: RegionalReferenceLatentPolicy
    known_patch_conflicts: tuple[RegionalPatchConflict, ...]

    def __post_init__(self) -> None:
        """Reject mutable, duplicate, or internally inconsistent capabilities."""

        enum_fields = (
            ("model family", self.model_family, RegionalModelFamily),
            ("attention backend", self.attention_backend, RegionalAttentionBackend),
            ("latent layout", self.latent_layout, RegionalLatentLayout),
            (
                "spatial patch support",
                self.spatial_patch_support,
                RegionalSpatialPatchSupport,
            ),
            (
                "control/GLIGEN policy",
                self.control_gligen_policy,
                RegionalControlGligenPolicy,
            ),
            (
                "reference-latent policy",
                self.reference_latent_policy,
                RegionalReferenceLatentPolicy,
            ),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise TypeError(
                    f"Regional {name} must be a {enum_type.__name__} value."
                )
        if not isinstance(self.known_patch_conflicts, tuple):
            raise TypeError(
                "Known regional patch conflicts must be an immutable tuple."
            )
        if not self.known_patch_conflicts:
            raise ValueError("Regional capabilities require known patch conflicts.")
        if not all(
            isinstance(conflict, RegionalPatchConflict)
            for conflict in self.known_patch_conflicts
        ):
            raise TypeError(
                "Known regional patch conflicts must contain "
                "RegionalPatchConflict values."
            )
        if len(set(self.known_patch_conflicts)) != len(self.known_patch_conflicts):
            raise ValueError(
                "Known regional patch conflicts must be unique and ordered."
            )
        self._validate_family_contract()

    def _validate_family_contract(self) -> None:
        """Require the exact backend, layout, and conflict surface for a family."""

        expected: tuple[
            RegionalAttentionBackend,
            RegionalLatentLayout,
            tuple[RegionalPatchConflict, ...],
        ]
        if self.model_family is RegionalModelFamily.ANIMA:
            expected = (
                RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
                RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW,
                (
                    RegionalPatchConflict.DIFFUSION_MODEL_WRAPPER,
                    RegionalPatchConflict.CROSS_ATTENTION_OBJECT_PATCH,
                    RegionalPatchConflict.ATTN2_INPUT_PATCH,
                    RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
                ),
            )
        else:
            expected = (
                RegionalAttentionBackend.UNET_ATTN2_PATCH,
                RegionalLatentLayout.STANDARD_IMAGE_BCHW,
                (
                    RegionalPatchConflict.ATTN2_INPUT_PATCH,
                    RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
                ),
            )
        if (
            self.attention_backend,
            self.latent_layout,
            self.known_patch_conflicts,
        ) != expected:
            raise ValueError(
                "Regional model capabilities do not match the model-family contract."
            )
