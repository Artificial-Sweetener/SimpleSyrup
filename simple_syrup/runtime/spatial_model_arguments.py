# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Transform model arguments and conditioning for spatial model views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..domain.spatial_views import SpatialBatchLayout, SpatialView, SpatialViewKind
from ..domain.tiled_diffusion import LatentTile
from .spatial_tensor_projection import (
    resize_spatial_tensor,
    spatial_view_slicer,
)

SPATIAL_INVARIANT_CONDITIONING_KEYS = frozenset({"ref_latents"})
SIMPLE_SYRUP_TRANSFORMER_NAMESPACE = "simple_syrup"
SPATIAL_BATCH_LAYOUT_KEY = "spatial_batch_layout"


def make_tiled_model_args(
    *,
    args: dict[str, Any],
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> dict[str, Any]:
    """Create apply-model args for one spatial tile batch."""

    layout = tiled_batch_layout(
        tiles=tiles,
        input_batch_size=input_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    x = args["input"]
    timestep = args["timestep"]
    conditioning = args.get("c", {})
    if not isinstance(x, torch.Tensor):
        raise ValueError("tiled sampler model input must be a tensor.")
    if not isinstance(timestep, torch.Tensor):
        raise ValueError("tiled sampler timestep must be a tensor.")
    if not isinstance(conditioning, dict):
        raise ValueError("tiled sampler conditioning must be a dict.")

    tiled_x = torch.cat(
        [x[spatial_view_slicer(view, x.ndim)] for view in layout.views],
        dim=0,
    )
    tiled_timestep = torch.cat([timestep] * layout.view_count, dim=0)
    tiled_conditioning = spatial_view_conditioning(
        conditioning=conditioning,
        layout=layout,
        view_timestep=tiled_timestep,
    )
    tiled_args = args.copy()
    tiled_args["input"] = tiled_x
    tiled_args["timestep"] = tiled_timestep
    tiled_args["c"] = tiled_conditioning
    if "cond_or_uncond" in args:
        tiled_args["cond_or_uncond"] = repeat_sequence(
            args["cond_or_uncond"],
            layout.view_count,
        )
    validate_spatial_model_args(
        source_args=args,
        transformed_args=tiled_args,
        layout=layout,
    )
    return tiled_args


def tiled_batch_layout(
    *,
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> SpatialBatchLayout:
    """Return exact ordered TILE views for one authoritative tile batch."""

    views = tuple(
        SpatialView(
            kind=SpatialViewKind.TILE,
            source_x=tile.x,
            source_y=tile.y,
            source_width=tile.width,
            source_height=tile.height,
            model_width=tile.width,
            model_height=tile.height,
        )
        for tile in tiles
    )
    return SpatialBatchLayout(
        canvas_width=latent_width,
        canvas_height=latent_height,
        views=views,
        input_batch_size=input_batch_size,
    )


def make_spatial_view_model_args(
    *,
    args: dict[str, Any],
    layout: SpatialBatchLayout,
) -> dict[str, Any]:
    """Create apply-model arguments for equally shaped spatial views."""

    target_shape = (layout.views[0].model_height, layout.views[0].model_width)
    if any(
        (view.model_height, view.model_width) != target_shape for view in layout.views
    ):
        raise ValueError("Batched spatial views must use one model spatial shape.")
    x = args["input"]
    timestep = args["timestep"]
    conditioning = args.get("c", {})
    if not isinstance(x, torch.Tensor):
        raise ValueError("contextual sampler model input must be a tensor.")
    if not isinstance(timestep, torch.Tensor):
        raise ValueError("contextual sampler timestep must be a tensor.")
    if not isinstance(conditioning, dict):
        raise ValueError("contextual sampler conditioning must be a dict.")

    view_x = torch.cat(
        [
            resize_spatial_tensor(
                x[spatial_view_slicer(view, x.ndim)],
                height=view.model_height,
                width=view.model_width,
                mode="nearest-exact",
            )
            for view in layout.views
        ],
        dim=0,
    )
    view_timestep = torch.cat([timestep] * layout.view_count, dim=0)
    view_conditioning = spatial_view_conditioning(
        conditioning=conditioning,
        layout=layout,
        view_timestep=view_timestep,
    )
    view_args = args.copy()
    view_args["input"] = view_x
    view_args["timestep"] = view_timestep
    view_args["c"] = view_conditioning
    if "cond_or_uncond" in args:
        view_args["cond_or_uncond"] = repeat_sequence(
            args["cond_or_uncond"],
            layout.view_count,
        )
    validate_spatial_model_args(
        source_args=args,
        transformed_args=view_args,
        layout=layout,
    )
    return view_args


def spatial_view_conditioning(
    *,
    conditioning: dict[str, Any],
    layout: SpatialBatchLayout,
    view_timestep: torch.Tensor,
) -> dict[str, Any]:
    """Resize spatial conditioning alongside arbitrary latent views."""

    transformed: dict[str, Any] = {}
    for key, value in conditioning.items():
        if key == "transformer_options":
            continue
        if key in SPATIAL_INVARIANT_CONDITIONING_KEYS:
            transformed[key] = repeat_spatial_invariant_value(
                value,
                view_count=layout.view_count,
                input_batch_size=layout.input_batch_size,
                conditioning_key=key,
            )
            continue
        transformed[key] = spatial_view_value(
            value,
            views=layout.views,
            input_batch_size=layout.input_batch_size,
            latent_height=layout.canvas_height,
            latent_width=layout.canvas_width,
        )
    transformer_options = conditioning.get("transformer_options", {})
    if not isinstance(transformer_options, dict):
        raise TypeError("conditioning transformer_options must be a dictionary.")
    transformed["transformer_options"] = spatial_transformer_options(
        transformer_options,
        layout=layout,
        view_timestep=view_timestep,
    )
    return transformed


def spatial_view_value(
    value: Any,
    *,
    views: Sequence[SpatialView],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> Any:
    """Transform tensors nested inside one spatial-view conditioning value."""

    if isinstance(value, torch.Tensor):
        if value.ndim >= 4 and value.shape[-2:] == (
            latent_height,
            latent_width,
        ):
            return torch.cat(
                [
                    resize_spatial_tensor(
                        value[spatial_view_slicer(view, value.ndim)],
                        height=view.model_height,
                        width=view.model_width,
                        mode="nearest-exact",
                    )
                    for view in views
                ],
                dim=0,
            )
        if value.ndim >= 1 and value.shape[0] == input_batch_size:
            return torch.cat([value] * len(views), dim=0)
        if value.ndim >= 1 and value.shape[0] == 1:
            repeats = [input_batch_size * len(views)] + [1] * (value.ndim - 1)
            return value.repeat(repeats)
        return value
    if isinstance(value, list):
        return [
            spatial_view_value(
                item,
                views=views,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            spatial_view_value(
                item,
                views=views,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        )
    return value


def repeat_spatial_invariant_value(
    value: Any,
    *,
    view_count: int,
    input_batch_size: int,
    conditioning_key: str,
) -> Any:
    """Repeat non-spatial conditioning without cropping its tensor contents."""

    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError(
                f"{conditioning_key} tensors must include a batch dimension."
            )
        if value.shape[0] == input_batch_size:
            return torch.cat([value] * view_count, dim=0)
        if value.shape[0] == 1:
            repeats = [input_batch_size * view_count] + [1] * (value.ndim - 1)
            return value.repeat(repeats)
        raise ValueError(
            f"{conditioning_key} tensor batch size must be 1 or match the model "
            f"input batch size {input_batch_size}; received {value.shape[0]}."
        )
    if isinstance(value, list):
        return [
            repeat_spatial_invariant_value(
                item,
                view_count=view_count,
                input_batch_size=input_batch_size,
                conditioning_key=conditioning_key,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            repeat_spatial_invariant_value(
                item,
                view_count=view_count,
                input_batch_size=input_batch_size,
                conditioning_key=conditioning_key,
            )
            for item in value
        )
    return value


def spatial_transformer_options(
    options: dict[str, Any],
    *,
    layout: SpatialBatchLayout,
    view_timestep: torch.Tensor,
) -> dict[str, Any]:
    """Repeat batch metadata and publish one layout through copied mappings."""

    transformed_options = options.copy()
    existing_namespace = options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
    if existing_namespace is None:
        simple_syrup_options: dict[str, Any] = {}
    elif isinstance(existing_namespace, dict):
        simple_syrup_options = existing_namespace.copy()
    else:
        raise TypeError("transformer_options.simple_syrup must be a dictionary.")
    simple_syrup_options[SPATIAL_BATCH_LAYOUT_KEY] = layout
    transformed_options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] = simple_syrup_options
    if "cond_or_uncond" in options:
        transformed_options["cond_or_uncond"] = repeat_sequence(
            options["cond_or_uncond"],
            layout.view_count,
        )
    if "uuids" in options:
        transformed_options["uuids"] = repeat_sequence(
            options["uuids"],
            layout.view_count,
        )
    if "sigmas" in options:
        transformed_options["sigmas"] = view_timestep
    return transformed_options


def repeat_sequence(value: object, times: int) -> object:
    """Repeat a metadata sequence once per tile."""

    if isinstance(value, list):
        return value * times
    if isinstance(value, tuple):
        return value * times
    return value


def validate_spatial_model_args(
    *,
    source_args: dict[str, Any],
    transformed_args: dict[str, Any],
    layout: SpatialBatchLayout,
) -> None:
    """Validate transformed tensor batches, metadata, and view-major ordering."""

    model_input = transformed_args.get("input")
    if not isinstance(model_input, torch.Tensor):
        raise TypeError("Spatial model input must be a torch.Tensor.")
    _validate_tensor_batch(
        model_input,
        expected=layout.expanded_batch_size,
        value_name="Spatial model input",
    )
    timestep = transformed_args.get("timestep")
    if not isinstance(timestep, torch.Tensor):
        raise TypeError("Spatial model timestep must be a torch.Tensor.")
    _validate_tensor_batch(
        timestep,
        expected=layout.expanded_batch_size,
        value_name="Spatial model timestep",
    )

    _validate_repeated_sequence(
        source_args,
        transformed_args,
        key="cond_or_uncond",
        layout=layout,
        value_name="Top-level cond_or_uncond",
    )
    source_conditioning = source_args.get("c", {})
    transformed_conditioning = transformed_args.get("c")
    if not isinstance(source_conditioning, dict) or not isinstance(
        transformed_conditioning,
        dict,
    ):
        raise TypeError("Spatial model conditioning must be a dictionary.")
    source_options = source_conditioning.get("transformer_options", {})
    transformed_options = transformed_conditioning.get("transformer_options")
    if not isinstance(source_options, dict) or not isinstance(
        transformed_options,
        dict,
    ):
        raise TypeError("Spatial transformer options must be a dictionary.")
    namespace = transformed_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
    if not isinstance(namespace, dict):
        raise TypeError("Spatial transformer options require SimpleSyrup metadata.")
    if namespace.get(SPATIAL_BATCH_LAYOUT_KEY) is not layout:
        raise ValueError("Spatial transformer layout does not match the model call.")

    for key, value_name in (
        ("cond_or_uncond", "Transformer cond_or_uncond"),
        ("uuids", "Transformer UUIDs"),
    ):
        _validate_repeated_sequence(
            source_options,
            transformed_options,
            key=key,
            layout=layout,
            value_name=value_name,
        )
    if "sigmas" in source_options:
        sigmas = transformed_options.get("sigmas")
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError("Transformer sigmas must be a torch.Tensor.")
        _validate_tensor_batch(
            sigmas,
            expected=layout.expanded_batch_size,
            value_name="Transformer sigmas",
        )
        if not torch.equal(sigmas, timestep):
            raise ValueError("Transformer sigmas must align with model timesteps.")


def _validate_tensor_batch(
    tensor: torch.Tensor,
    *,
    expected: int,
    value_name: str,
) -> None:
    """Require one leading tensor entry per expanded layout item."""

    if tensor.ndim < 1 or int(tensor.shape[0]) != expected:
        actual = 0 if tensor.ndim < 1 else int(tensor.shape[0])
        raise ValueError(f"{value_name} batch must be {expected}; received {actual}.")


def _validate_repeated_sequence(
    source: dict[str, Any],
    transformed: dict[str, Any],
    *,
    key: str,
    layout: SpatialBatchLayout,
    value_name: str,
) -> None:
    """Require exact source-sequence repetition in view-major order."""

    if key not in source:
        return
    source_value = source[key]
    transformed_value = transformed.get(key)
    if not isinstance(source_value, list | tuple) or not isinstance(
        transformed_value,
        list | tuple,
    ):
        raise TypeError(f"{value_name} must be a list or tuple.")
    expected = source_value * layout.view_count
    if transformed_value != expected:
        raise ValueError(
            f"{value_name} is not in view-major order for "
            f"{layout.view_count} spatial views over input batch "
            f"{layout.input_batch_size}; expected the source sequence repeated "
            "once per view."
        )
