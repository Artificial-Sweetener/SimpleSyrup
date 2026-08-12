"""Validate the exact live P5.8 public-node metadata contract."""

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
)


def validate_public_node_metadata(metadata: JsonObject) -> None:
    """Reject metadata that does not prove the shipped public schema."""

    if metadata.get("name") != PUBLIC_NODE_ID:
        raise ValueError("Live Attention Coupling node ID does not match P5.8.")
    if metadata.get("display_name") != "KSampler (Attention Coupling)":
        raise ValueError("Live Attention Coupling display name does not match P5.8.")
    if metadata.get("category") != "SimpleSyrup/Sampling":
        raise ValueError("Live Attention Coupling category does not match P5.8.")
    order = _object(metadata.get("input_order"), "input_order").get("required")
    if not isinstance(order, list) or tuple(order) != EXPECTED_INPUTS:
        raise ValueError("Live Attention Coupling input order does not match P5.8.")
    required = _object(
        _object(metadata.get("input"), "input").get("required"), "required"
    )
    if tuple(required) != EXPECTED_INPUTS:
        raise ValueError("Live Attention Coupling required inputs do not match P5.8.")
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
    if metadata.get("output") != ["LATENT"]:
        raise ValueError("Live Attention Coupling output type does not match P5.8.")
    description = metadata.get("description")
    if not isinstance(description, str) or not all(
        word in description.lower() for word in ("lora", "schedule", "overlap")
    ):
        raise ValueError("Live description omits required regional LoRA guidance.")
    for name in ("model", "positive", "negative", "region_masks"):
        descriptor = cast(list[object], required[name])
        options = _object(descriptor[1], f"{name} options")
        tooltip = options.get("tooltip")
        if not isinstance(tooltip, str) or "lora" not in tooltip.lower():
            raise ValueError(f"Live input {name!r} omits regional LoRA guidance.")


def _object(value: object, label: str) -> JsonObject:
    """Narrow one dynamic metadata value to a string-keyed object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Live Attention Coupling {label} must be an object.")
    return cast(JsonObject, value)
