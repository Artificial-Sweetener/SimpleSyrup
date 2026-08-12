"""Validate the exact live P6.9 tiled public-node metadata contract."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject

from .matrix import PUBLIC_NODE_ID

EXPECTED_INPUTS = (
    "model",
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "positive",
    "negative",
    "region_masks",
    "regional_prompt_weight",
    "region_mask_feather",
    "latent_image",
    "denoise",
    "diffusion_mode",
    "latent_tile_width",
    "latent_tile_height",
    "latent_tile_overlap",
    "latent_tile_batch_size",
)


def validate_public_node_metadata(metadata: JsonObject) -> None:
    """Reject metadata that does not prove the shipped tiled schema."""

    if metadata.get("name") != PUBLIC_NODE_ID:
        raise ValueError("Live tiled Attention Coupling node ID does not match P6.9.")
    if metadata.get("display_name") != (
        "KSampler (Attention Coupling + Tiled Diffusion)"
    ):
        raise ValueError("Live tiled Attention Coupling display name does not match.")
    if metadata.get("category") != "SimpleSyrup/Sampling":
        raise ValueError("Live tiled Attention Coupling category does not match.")
    order = _object(metadata.get("input_order"), "input_order").get("required")
    if not isinstance(order, list) or tuple(order) != EXPECTED_INPUTS:
        raise ValueError("Live tiled Attention Coupling input order does not match.")
    required = _object(
        _object(metadata.get("input"), "input").get("required"),
        "required",
    )
    if tuple(required) != EXPECTED_INPUTS:
        raise ValueError("Live tiled Attention Coupling inputs do not match.")
    expected_types = {
        "model": "MODEL",
        "positive": "CONDITIONING,CONDITIONING_BATCH",
        "negative": "CONDITIONING,CONDITIONING_BATCH",
        "region_masks": "MASK",
        "latent_image": "LATENT",
    }
    for name, expected_type in expected_types.items():
        descriptor = required[name]
        if not isinstance(descriptor, list) or descriptor[0] != expected_type:
            raise ValueError(f"Live input {name!r} does not expose {expected_type}.")
    mode = required["diffusion_mode"]
    if (
        not isinstance(mode, list)
        or len(mode) != 2
        or mode[0] != "COMBO"
        or _object(mode[1], "diffusion_mode options").get("options")
        != ["multidiffusion", "mixture_of_diffusers"]
    ):
        raise ValueError("Live tiled Attention Coupling fusion modes do not match.")
    if metadata.get("output") != ["LATENT"]:
        raise ValueError("Live tiled Attention Coupling output type does not match.")
    description = metadata.get("description")
    if not isinstance(description, str) or not all(
        word in description.lower()
        for word in ("lora", "schedule", "multidiffusion", "mixture")
    ):
        raise ValueError("Live tiled description omits required LoRA guidance.")
    for name in ("model", "positive", "negative", "region_masks"):
        descriptor = cast(list[object], required[name])
        tooltip = _object(descriptor[1], f"{name} options").get("tooltip")
        if not isinstance(tooltip, str) or "lora" not in tooltip.lower():
            raise ValueError(f"Live input {name!r} omits regional LoRA guidance.")


def _object(value: object, label: str) -> JsonObject:
    """Narrow one dynamic metadata value to a string-keyed object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Live tiled Attention Coupling {label} must be an object.")
    return cast(JsonObject, value)
