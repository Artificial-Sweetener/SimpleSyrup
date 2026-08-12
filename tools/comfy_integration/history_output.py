"""Validate completed Comfy history and extract one saved image reference."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import ImageReference, JsonObject


def extract_saved_image(history: JsonObject, save_node_id: str) -> ImageReference:
    """Return the sole saved image from one successful completed prompt."""

    status = history.get("status")
    if not isinstance(status, dict):
        raise ValueError("Comfy history is missing its status object.")
    status_text = status.get("status_str")
    if status_text != "success":
        messages = status.get("messages")
        raise RuntimeError(
            f"Comfy prompt failed with status {status_text!r}: {messages!r}"
        )
    if status.get("completed") is not True:
        raise ValueError("Comfy history does not describe a completed prompt.")

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Comfy history is missing its outputs object.")
    saved_output = outputs.get(save_node_id)
    if not isinstance(saved_output, dict):
        raise ValueError(f"Comfy history is missing SaveImage output {save_node_id!r}.")
    images = saved_output.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Baseline SaveImage output must contain exactly one image.")
    image = images[0]
    if not isinstance(image, dict):
        raise TypeError("Saved image history entry must be an object.")
    narrowed = cast(dict[object, object], image)
    filename = narrowed.get("filename")
    subfolder = narrowed.get("subfolder")
    output_type = narrowed.get("type")
    if not isinstance(filename, str):
        raise TypeError("Saved image filename must be a string.")
    if not isinstance(subfolder, str):
        raise TypeError("Saved image subfolder must be a string.")
    if not isinstance(output_type, str):
        raise TypeError("Saved image output type must be a string.")
    return ImageReference(filename, subfolder, output_type)
