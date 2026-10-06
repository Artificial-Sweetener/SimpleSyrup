# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolve trusted learned checkpoints through registered Comfy model paths."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from ..domain.latent_upscale import recommended_family
from ..domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from .auto_model_resolver import AutoModelResolver
from .latent_upscale_catalog import LATENT_UPSCALE_ARTIFACTS
from .model_downloads import ComfyProgressReporter


def resolve_latent_upscale_model(
    latent: dict[str, Any], method: str
) -> tuple[str, Path]:
    """Resolve verified Auto identity or a checkpoint with a recognized architecture."""
    provenance = latent.get(PROVENANCE_KEY)
    if method == "Auto":
        if not isinstance(provenance, SamplingProvenance):
            raise ValueError(
                "Auto latent upscaling requires sampling provenance; "
                "connect a source sampler or select a compatible "
                "checkpoint explicitly."
            )
        family = recommended_family(provenance.latent_format)
        return family, AutoModelResolver().resolve(
            LATENT_UPSCALE_ARTIFACTS[family], ComfyProgressReporter()
        ).path
    for family, artifact in LATENT_UPSCALE_ARTIFACTS.items():
        if method.replace("\\", "/") == f"{family}/{artifact.filename}":
            if (
                isinstance(provenance, SamplingProvenance)
                and recommended_family(provenance.latent_format) != family
            ):
                raise ValueError(
                    "Selected latent upscale checkpoint does not match the "
                    "source latent format."
                )
            return family, AutoModelResolver().resolve(
                artifact, ComfyProgressReporter()
            ).path
    folders = import_module("folder_paths")
    path = Path(folders.get_full_path_or_raise("latent_upscale_models", method))
    state = import_module("comfy.utils").load_torch_file(str(path), safe_load=True)
    head = state.get("head_conv.0.weight")
    if (
        not isinstance(head, torch.Tensor)
        or head.ndim != 4
        or "upsample.base_taps" not in state
    ):
        raise ValueError(
            "Selected checkpoint is not a supported SesquiLSR latent upscaler."
        )
    named_family = next(
        (
            family
            for family, artifact in LATENT_UPSCALE_ARTIFACTS.items()
            if path.name.casefold() == artifact.filename.casefold()
        ),
        None,
    )
    if named_family is not None:
        expected_channels = {"sdxl": 4, "flux": 16, "wan": 16, "flux2": 32}[
            named_family
        ]
        if head.shape[1] != expected_channels:
            raise ValueError(
                "Selected checkpoint filename and latent channels disagree."
            )
    detected_family = named_family or {4: "sdxl", 32: "flux2"}.get(int(head.shape[1]))
    if detected_family is None and head.shape[1] == 16:
        if not isinstance(provenance, SamplingProvenance):
            raise ValueError(
                "A 16-channel checkpoint requires verified Flux or Wan "
                "sampling provenance."
            )
        detected_family = recommended_family(provenance.latent_format)
    if detected_family not in LATENT_UPSCALE_ARTIFACTS:
        raise ValueError("Selected checkpoint has unsupported latent channels.")
    if (
        isinstance(provenance, SamplingProvenance)
        and recommended_family(provenance.latent_format) != detected_family
    ):
        raise ValueError(
            "Selected latent upscale checkpoint does not match "
            "the source latent format."
        )
    return detected_family, path
