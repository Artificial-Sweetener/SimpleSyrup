# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep persisted public widget positions append-only for established nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import pytest

from simple_syrup.nodes.detail_segs_as_regions import DetailSEGSAsRegions
from simple_syrup.nodes.detail_segs_by_scale_factor import DetailSEGSByScaleFactor
from simple_syrup.nodes.detail_segs_by_scale_factor_tiled_diffusion import (
    DetailSEGSByScaleFactorTiledDiffusion,
)
from simple_syrup.nodes.detect_segs_with_ultralytics import DetectSEGSWithUltralytics
from simple_syrup.nodes.grounding_dino_model_loader import GroundingDINOModelLoader
from simple_syrup.nodes.image_resize_to_target import ResizeImageToTarget
from simple_syrup.nodes.ksampler_extras import KSamplerExtras
from simple_syrup.nodes.ksampler_tiled_diffusion import KSamplerTiledDiffusion
from simple_syrup.nodes.load_ultralytics_model import LoadUltralyticsModel
from simple_syrup.nodes.prompt_encode_style import PromptEncodeStyle
from simple_syrup.nodes.prompt_segs_with_sam import PromptSEGSWithSAM
from simple_syrup.nodes.sam_model_loader import SAMModelLoader
from simple_syrup.nodes.schedule_and_encode_prompts_with_prompt_control import (
    ScheduleAndEncodePromptsWithPromptControl,
)
from simple_syrup.nodes.seed import Seed
from simple_syrup.nodes.simple_load_anima import SimpleLoadAnima
from simple_syrup.nodes.simple_load_checkpoint import SimpleLoadCheckpoint
from simple_syrup.nodes.vae_options import VAEDecodeOptions, VAEEncodeOptions
from simple_syrup.nodes.vitmatte_model_loader import ViTMatteModelLoader

_WIDGET_TYPES = frozenset(
    {"BOOLEAN", "COMBO", "FLOAT", "INT", "LIST", "NUMBER", "STRING", "TEXT"}
)


class _ClassicNode(Protocol):
    """Describe the persisted classic Comfy input declaration contract."""

    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, object]:
        """Return the node's ordered classic Comfy inputs."""


_PERSISTED_WIDGET_PREFIXES: tuple[tuple[type[_ClassicNode], tuple[str, ...]], ...] = (
    (
        DetailSEGSAsRegions,
        (
            "global_prompt_weight",
            "scale_factor",
            "upscale_method",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "feather",
            "noise_mask",
            "noise_mask_feather",
            "tiled_encode",
            "tiled_decode",
        ),
    ),
    (
        DetailSEGSByScaleFactor,
        (
            "scale_factor",
            "upscale_method",
            "clamp_size",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "feather",
            "noise_mask",
            "noise_mask_feather",
            "tiled_encode",
            "tiled_decode",
        ),
    ),
    (
        DetailSEGSByScaleFactorTiledDiffusion,
        (
            "scale_factor",
            "upscale_method",
            "clamp_size",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "feather",
            "noise_mask",
            "noise_mask_feather",
            "tiled_encode",
            "tiled_decode",
            "diffusion_mode",
            "latent_tile_width",
            "latent_tile_height",
            "latent_tile_overlap",
            "latent_tile_batch_size",
        ),
    ),
    (
        DetectSEGSWithUltralytics,
        (
            "confidence_threshold",
            "size_threshold",
            "keep_only",
            "keep_by",
            "bbox_dilation",
            "sub_dilation",
            "post_dilation",
            "crop_factor",
            "sort_order",
            "combine_segs",
        ),
    ),
    (GroundingDINOModelLoader, ("grounding_dino_model", "text_encoder")),
    (
        KSamplerExtras,
        ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    ),
    (
        KSamplerTiledDiffusion,
        (
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "diffusion_mode",
            "latent_tile_width",
            "latent_tile_height",
            "latent_tile_overlap",
            "latent_tile_batch_size",
        ),
    ),
    (LoadUltralyticsModel, ("model_name",)),
    (PromptEncodeStyle, ("encode_style",)),
    (
        PromptSEGSWithSAM,
        (
            "positive_prompt",
            "negative_prompt",
            "confidence_threshold",
            "size_threshold",
            "keep_only",
            "keep_by",
            "bbox_dilation",
            "mask_dilation",
            "detail_method",
            "detail_erode",
            "detail_dilate",
            "black_point",
            "white_point",
            "refine_mask",
            "mask_refinement_max_size",
            "execution_device",
            "crop_factor",
            "sort_order",
            "combine_segs",
        ),
    ),
    (
        ResizeImageToTarget,
        (
            "width",
            "height",
            "resize_mode",
            "sampling",
            "processor",
            "divisible_by",
            "crop_position",
            "pad_color",
            "max_batch_size",
            "sinc_window",
            "precision",
        ),
    ),
    (SAMModelLoader, ("sam_model",)),
    (
        ScheduleAndEncodePromptsWithPromptControl,
        ("positive_prompt", "negative_prompt", "encode_style"),
    ),
    (Seed, ("seed",)),
    (
        SimpleLoadAnima,
        (
            "diffusion_model",
            "diffusion_weight_dtype",
            "text_encoder",
            "text_encoder_device",
            "vae",
        ),
    ),
    (SimpleLoadCheckpoint, ("ckpt_name", "vae_name", "clip_skip")),
    (
        VAEDecodeOptions,
        ("use_tiling", "tile_size", "overlap", "temporal_size", "temporal_overlap"),
    ),
    (
        VAEEncodeOptions,
        ("use_tiling", "tile_size", "overlap", "temporal_size", "temporal_overlap"),
    ),
    (ViTMatteModelLoader, ("vitmatte_model",)),
)


@pytest.mark.parametrize(("node_class", "persisted_prefix"), _PERSISTED_WIDGET_PREFIXES)
def test_established_widget_inputs_remain_an_append_only_prefix(
    node_class: type[_ClassicNode],
    persisted_prefix: tuple[str, ...],
) -> None:
    """Fail when an established widget is inserted, removed, or reordered."""

    actual = _widget_input_names(node_class.INPUT_TYPES())

    assert actual[: len(persisted_prefix)] == persisted_prefix


def _widget_input_names(input_types: Mapping[str, object]) -> tuple[str, ...]:
    """Return classic Comfy widget inputs in their persisted positional order."""

    names: list[str] = []
    for section_name in ("required", "optional"):
        section = input_types.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for input_name, field_spec in section.items():
            if _is_widget_field(field_spec):
                names.append(str(input_name))
    return tuple(names)


def _is_widget_field(field_spec: object) -> bool:
    """Return whether a classic Comfy field serializes a widget value."""

    if not isinstance(field_spec, Sequence) or isinstance(field_spec, str | bytes):
        return False
    if not field_spec:
        return False
    input_type = field_spec[0]
    return (
        isinstance(input_type, Sequence)
        and not isinstance(input_type, str | bytes)
        or isinstance(input_type, str)
        and input_type.upper() in _WIDGET_TYPES
    )
