"""Validate the exact live P7.8 combined-node metadata contract."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject

from .matrix import PUBLIC_NODE_ID

EXPECTED_REQUIRED_INPUTS = (
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
    "latent_context_size",
    "latent_context_overlap",
    "latent_context_batch_size",
    "global_weight",
    "global_steps",
    "global_decay",
)
EXPECTED_OPTIONAL_INPUTS = ("segs",)


def validate_public_node_metadata(metadata: JsonObject) -> None:
    """Reject metadata that does not prove the shipped combined schema."""

    if metadata.get("name") != PUBLIC_NODE_ID:
        raise ValueError("Live Contextual Attention Coupling node ID does not match.")
    if metadata.get("display_name") != (
        "KSampler (Attention Coupling + Contextual Diffusion)"
    ):
        raise ValueError("Live Contextual Attention Coupling display name differs.")
    if metadata.get("category") != "SimpleSyrup/Sampling":
        raise ValueError("Live Contextual Attention Coupling category differs.")
    input_order = _object(metadata.get("input_order"), "input_order")
    if tuple(_list(input_order.get("required"), "required input order")) != (
        EXPECTED_REQUIRED_INPUTS
    ):
        raise ValueError("Live Contextual Attention Coupling input order differs.")
    if tuple(_list(input_order.get("optional"), "optional input order")) != (
        EXPECTED_OPTIONAL_INPUTS
    ):
        raise ValueError("Live Contextual optional input order differs.")
    inputs = _object(metadata.get("input"), "input")
    required = _object(inputs.get("required"), "required inputs")
    optional = _object(inputs.get("optional"), "optional inputs")
    if tuple(required) != EXPECTED_REQUIRED_INPUTS or tuple(optional) != (
        EXPECTED_OPTIONAL_INPUTS
    ):
        raise ValueError("Live Contextual Attention Coupling inputs differ.")
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
    segs = optional["segs"]
    if not isinstance(segs, list) or segs[0] != "SEGS":
        raise ValueError("Live optional SEGS input type differs.")
    mode = required["diffusion_mode"]
    if (
        not isinstance(mode, list)
        or len(mode) != 2
        or mode[0] != "COMBO"
        or _object(mode[1], "diffusion mode options").get("options")
        != ["multidiffusion", "mixture_of_diffusers"]
    ):
        raise ValueError("Live Contextual fusion modes differ.")
    if metadata.get("output") != ["LATENT", "SEGS"]:
        raise ValueError("Live Contextual Attention Coupling outputs differ.")
    description = metadata.get("description")
    if not isinstance(description, str) or not all(
        word in description.lower()
        for word in ("lora", "schedule", "local", "reduced-global", "segs")
    ):
        raise ValueError("Live Contextual description omits required guidance.")
    for name in ("model", "positive", "negative", "region_masks"):
        descriptor = cast(list[object], required[name])
        tooltip = _object(descriptor[1], f"{name} options").get("tooltip")
        if not isinstance(tooltip, str) or "lora" not in tooltip.lower():
            raise ValueError(f"Live input {name!r} omits regional LoRA guidance.")


def _object(value: object, label: str) -> JsonObject:
    """Narrow one dynamic metadata value to a string-keyed object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Live Contextual {label} must be an object.")
    return cast(JsonObject, value)


def _list(value: object, label: str) -> list[object]:
    """Narrow one dynamic metadata value to a list."""

    if not isinstance(value, list):
        raise TypeError(f"Live Contextual {label} must be a list.")
    return value
