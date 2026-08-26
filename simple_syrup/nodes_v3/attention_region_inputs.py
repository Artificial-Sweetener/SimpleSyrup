# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build shared Comfy v3 inputs and domain controls for attention-region nodes."""

from __future__ import annotations

from typing import Any

from ..domain.attention_region_capture import (
    AttentionCaptureProfile,
    AttentionRegionControls,
)


def attention_region_control_inputs(io: Any) -> list[object]:
    """Return the complete shared attention-native control schema."""

    return [
        io.Int.Input(
            "sampler_stage",
            default=1,
            min=-1,
            max=1024,
            step=1,
            tooltip=(
                "Selects the connected sampling stage: 1 is first, 2 is second, "
                "and 0 or -1 selects the last; oversized values use the last."
            ),
        ),
        io.Float.Input(
            "capture_start",
            default=0.0,
            min=0.0,
            max=0.99,
            step=0.01,
            tooltip=(
                "Start of denoising evidence to include; later values ignore more "
                "of the initial composition phase."
            ),
        ),
        io.Float.Input(
            "capture_end",
            default=1.0,
            min=0.01,
            max=1.0,
            step=0.01,
            tooltip=(
                "End of denoising evidence to include; earlier values ignore more "
                "late refinement attention."
            ),
        ),
        io.Float.Input(
            "minimum_strength",
            default=0.15,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=(
                "Minimum normalized attention association retained in a region; "
                "higher values narrow the silhouette toward its semantic core."
            ),
        ),
        io.Float.Input(
            "minimum_consensus",
            default=0.25,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=(
                "Fraction of selected observations that must support a pixel; "
                "higher values keep more persistent regions."
            ),
        ),
        io.Float.Input(
            "split_sensitivity",
            default=0.35,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=(
                "Sensitivity to separate peaks into instances without shrinking "
                "their combined silhouette; higher values cut stronger bridges."
            ),
        ),
        io.Int.Input(
            "minimum_region_size",
            default=512,
            min=1,
            max=1048576,
            step=1,
            tooltip="Discard attention components smaller than this many pixels.",
        ),
        io.Int.Input(
            "keep_only",
            default=1,
            min=0,
            max=1024,
            step=1,
            tooltip=(
                "Keep the best N instances per concept; 1 keeps the largest and "
                "0 keeps all."
            ),
        ),
        io.Combo.Input(
            "keep_by",
            options=["largest size", "highest confidence"],
            default="largest size",
            tooltip="Ranks retained instances by area or attention confidence.",
        ),
        io.Boolean.Input(
            "combine_segs",
            default=False,
            tooltip="Combines retained instances of each concept into one SEG.",
        ),
        io.Float.Input(
            "matte_solidity",
            default=0.75,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=(
                "Higher values flatten accepted interiors toward fully opaque alpha."
            ),
        ),
        io.Int.Input(
            "edge_feather",
            default=8,
            min=0,
            max=4096,
            step=1,
            tooltip="Width in output pixels of the matte boundary transition.",
        ),
        io.Combo.Input(
            "capture_profile",
            options=[profile.value for profile in AttentionCaptureProfile],
            default=AttentionCaptureProfile.FAST.value,
            tooltip=(
                "Controls observation density: fast minimizes overhead, balanced "
                "adds temporal evidence, and exhaustive retains every eligible call."
            ),
        ),
    ]


def attention_region_controls(
    *,
    capture_start: float,
    capture_end: float,
    minimum_strength: float,
    minimum_consensus: float,
    split_sensitivity: float,
    minimum_region_size: int,
    keep_only: int,
    keep_by: str,
    combine_segs: bool,
    matte_solidity: float,
    edge_feather: int,
    capture_profile: str,
) -> AttentionRegionControls:
    """Build validated domain controls from public node inputs."""

    return AttentionRegionControls(
        capture_start=capture_start,
        capture_end=capture_end,
        minimum_strength=minimum_strength,
        minimum_consensus=minimum_consensus,
        split_sensitivity=split_sensitivity,
        minimum_region_size=minimum_region_size,
        profile=AttentionCaptureProfile(capture_profile),
        keep_only=keep_only,
        keep_by=keep_by,
        combine_segs=combine_segs,
        matte_solidity=matte_solidity,
        edge_feather=edge_feather,
    )
